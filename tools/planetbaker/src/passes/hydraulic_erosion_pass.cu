#include "passes/hydraulic_erosion_pass.h"

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/neighbor_table.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pb {

namespace {

//====================================
//Units convention for this pass:
//  bedrock_elevation     : km
//  sediment_thickness    : m
//  water_column          : m
//  suspended (pass-local): m of sediment equivalent
//  flux (pass-local)     : m of water column transferred per iteration
//                          across one of the four pipes (+i, -i, +j, -j)
//
//Surface elevation in meters used by the flow kernels is computed as
//bedrock_elevation * 1000 + sediment_thickness + water_column.
//====================================

__device__ __forceinline__ int cell_idx(int face, int i, int j, int stride, int halo) {
    return face * stride * stride + (j + halo) * stride + (i + halo);
}

__device__ __forceinline__ float surf_total_m(float bed_km, float sed_m, float water_m) {
    return bed_km * 1000.0f + sed_m + water_m;
}

//K1: add a constant rainfall to every interior cell.
__global__ void hyd_rain_kernel(float* __restrict__ water,
                                int n, int stride, int halo,
                                float rain_m) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c = cell_idx(face, i, j, stride, halo);
    float w = water[c] + rain_m;
    if (w < 0.0f) w = 0.0f;
    water[c] = w;
}

//K2: per-cell outflow flux to the four neighbours, computed from total
//surface elevation. Outflow per direction is k_flow * max(0, dh); the four
//are uniformly scaled so total outflow never exceeds the cell's available
//water column.
__global__ void hyd_flux_kernel(const float* __restrict__ bed,
                                const float* __restrict__ sed,
                                const float* __restrict__ water,
                                float4* __restrict__ flux,
                                int n, int stride, int halo,
                                float k_flow) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c   = cell_idx(face, i, j, stride, halo);
    int pi  = c + 1;
    int mi  = c - 1;
    int pj  = c + stride;
    int mj  = c - stride;

    float w_c = water[c];
    if (w_c <= 0.0f) {
        flux[c] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    float surf_c  = surf_total_m(bed[c],  sed[c],  water[c]);
    float surf_pi = surf_total_m(bed[pi], sed[pi], water[pi]);
    float surf_mi = surf_total_m(bed[mi], sed[mi], water[mi]);
    float surf_pj = surf_total_m(bed[pj], sed[pj], water[pj]);
    float surf_mj = surf_total_m(bed[mj], sed[mj], water[mj]);

    float dh_pi = surf_c - surf_pi;
    float dh_mi = surf_c - surf_mi;
    float dh_pj = surf_c - surf_pj;
    float dh_mj = surf_c - surf_mj;

    float out_pi = (dh_pi > 0.0f) ? k_flow * dh_pi : 0.0f;
    float out_mi = (dh_mi > 0.0f) ? k_flow * dh_mi : 0.0f;
    float out_pj = (dh_pj > 0.0f) ? k_flow * dh_pj : 0.0f;
    float out_mj = (dh_mj > 0.0f) ? k_flow * dh_mj : 0.0f;

    float total = out_pi + out_mi + out_pj + out_mj;
    if (total > w_c && total > 0.0f) {
        float scl = w_c / total;
        out_pi *= scl;
        out_mi *= scl;
        out_pj *= scl;
        out_mj *= scl;
    }
    flux[c] = make_float4(out_pi, out_mi, out_pj, out_mj);
}

//K3: update water column. water_out = water_in - sum(outflow_self) +
//sum(outflow_neighbour_towards_self). The neighbour-towards-self channel for
//each direction is the OPPOSITE component on the neighbour cell (e.g. the
//+i neighbour's -i channel flows back to us).
__global__ void hyd_water_update_kernel(const float*  __restrict__ water_in,
                                        const float4* __restrict__ flux,
                                        float*        __restrict__ water_out,
                                        int n, int stride, int halo) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c  = cell_idx(face, i, j, stride, halo);
    int pi = c + 1;
    int mi = c - 1;
    int pj = c + stride;
    int mj = c - stride;

    float4 f = flux[c];
    float total_out = f.x + f.y + f.z + f.w;

    float in_pi = flux[pi].y;   // +i neighbour sending in its -i direction
    float in_mi = flux[mi].x;
    float in_pj = flux[pj].w;
    float in_mj = flux[mj].z;
    float total_in = in_pi + in_mi + in_pj + in_mj;

    float w = water_in[c] + total_in - total_out;
    if (w < 0.0f) w = 0.0f;
    water_out[c] = w;
}

//K4: per-cell erosion / deposition based on flow speed proxy and slope.
//Capacity is K_cap * slope * speed * water_depth, where slope is the
//central-difference gradient magnitude of surface elevation (bedrock +
//sediment, in metres) and speed is the dimensionless fraction of the water
//column moving horizontally per iteration (net flux / water column).
//
//If suspended < capacity the cell erodes: sediment_thickness is consumed
//first, then bedrock once sediment is gone. If suspended > capacity the
//excess deposits to sediment_thickness. The per-iteration erosion is
//capped by max_erode_m to keep the explicit Euler step stable on very
//steep cells.
__global__ void hyd_erode_kernel(float* __restrict__ bed,
                                 float* __restrict__ sed,
                                 const float*  __restrict__ water,
                                 const float4* __restrict__ flux,
                                 float* __restrict__ suspended,
                                 int n, int stride, int halo,
                                 float capacity_k,
                                 float erode_k,
                                 float deposit_k,
                                 float min_slope_m,
                                 float max_erode_m) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c  = cell_idx(face, i, j, stride, halo);
    int pi = c + 1;
    int mi = c - 1;
    int pj = c + stride;
    int mj = c - stride;

    float w = water[c];
    if (w <= 1.0e-6f) return;

    //Slope is the central-difference gradient magnitude of (bed_km*1000 +
    //sed_m). We never need the self-cell surface value here - the gradient
    //terms only use the neighbours - so it's omitted to keep the kernel
    //register-light.
    float surf_pi = bed[pi] * 1000.0f + sed[pi];
    float surf_mi = bed[mi] * 1000.0f + sed[mi];
    float surf_pj = bed[pj] * 1000.0f + sed[pj];
    float surf_mj = bed[mj] * 1000.0f + sed[mj];

    float gx = 0.5f * (surf_pi - surf_mi);
    float gy = 0.5f * (surf_pj - surf_mj);
    float slope = sqrtf(gx * gx + gy * gy);
    if (slope < min_slope_m) slope = min_slope_m;

    float4 f = flux[c];
    float net_x = f.x - f.y;
    float net_y = f.z - f.w;
    float speed = sqrtf(net_x * net_x + net_y * net_y) / w;

    float capacity = capacity_k * slope * speed * w;
    float susp     = suspended[c];

    if (susp < capacity) {
        float erode = erode_k * (capacity - susp);
        if (erode > max_erode_m) erode = max_erode_m;

        float s = sed[c];
        if (s >= erode) {
            sed[c] = s - erode;
        } else {
            float from_bed_m = erode - s;
            sed[c] = 0.0f;
            bed[c] = bed[c] - from_bed_m * 0.001f;
        }
        suspended[c] = susp + erode;
    } else {
        float deposit = deposit_k * (susp - capacity);
        suspended[c]  = susp - deposit;
        sed[c]        = sed[c] + deposit;
    }
}

//K5: transport suspended sediment with the water flow. Sediment moves in the
//same proportion as water does, so cell C loses suspended * (total_out/w)
//to its neighbours and gains, per neighbour, suspended_n * (flux_to_c / w_n).
//Gather form to avoid scatter-write races.
__global__ void hyd_transport_kernel(const float*  __restrict__ susp_in,
                                     const float*  __restrict__ water,
                                     const float4* __restrict__ flux,
                                     float*        __restrict__ susp_out,
                                     int n, int stride, int halo) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c  = cell_idx(face, i, j, stride, halo);
    int pi = c + 1;
    int mi = c - 1;
    int pj = c + stride;
    int mj = c - stride;

    float w_c = water[c];
    float4 f_c = flux[c];
    float total_out = f_c.x + f_c.y + f_c.z + f_c.w;
    float frac_lost = (w_c > 1.0e-6f) ? (total_out / w_c) : 0.0f;
    if (frac_lost > 1.0f) frac_lost = 1.0f;
    float retained = susp_in[c] * (1.0f - frac_lost);

    float w_pi = water[pi];
    float w_mi = water[mi];
    float w_pj = water[pj];
    float w_mj = water[mj];

    float in_pi = (w_pi > 1.0e-6f) ? susp_in[pi] * (flux[pi].y / w_pi) : 0.0f;
    float in_mi = (w_mi > 1.0e-6f) ? susp_in[mi] * (flux[mi].x / w_mi) : 0.0f;
    float in_pj = (w_pj > 1.0e-6f) ? susp_in[pj] * (flux[pj].w / w_pj) : 0.0f;
    float in_mj = (w_mj > 1.0e-6f) ? susp_in[mj] * (flux[mj].z / w_mj) : 0.0f;

    float new_s = retained + in_pi + in_mi + in_pj + in_mj;
    if (new_s < 0.0f) new_s = 0.0f;
    susp_out[c] = new_s;
}

//K6: scalar evaporation, w *= (1 - rate).
__global__ void hyd_evaporate_kernel(float* __restrict__ water,
                                     int n, int stride, int halo,
                                     float factor) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c = cell_idx(face, i, j, stride, halo);
    float w = water[c] * factor;
    if (w < 0.0f) w = 0.0f;
    water[c] = w;
}

//Final-frame drain: anything still in suspension when the run finishes is
//deposited into sediment so we don't lose mass. Single kernel, runs once.
__global__ void hyd_drain_suspended_kernel(float* __restrict__ sed,
                                            const float* __restrict__ suspended,
                                            int n, int stride, int halo) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c = cell_idx(face, i, j, stride, halo);
    sed[c] = sed[c] + suspended[c];
}

}

void HydraulicErosionPass::declare_params(ParamRegistry& reg) const {
    const char* o = name();

    reg.declare_int  ("hydraulic.iterations",         0, 0, 10000,
                      "iterations",
                      "Outer simulation iterations. 0 = pass is a no-op "
                      "(the default, so the dry-rocky pipeline stays "
                      "untouched). 500 carves shallow valleys, 2000 "
                      "produces a fully developed drainage network.",
                      "", o);
    reg.declare_float("hydraulic.rain_per_iter_m",    0.05f, 0.0f, 1.0f,
                      "rain / iter",
                      "Water depth added uniformly per iteration. Couples "
                      "with evaporation to set the equilibrium water "
                      "column. Default ~ 0.05 m gives ~2.5 m equilibrium "
                      "at evap 0.02, enough to carry sediment in channels.",
                      "m", o);
    reg.declare_float("hydraulic.evaporation_rate",   0.02f, 0.0f, 0.5f,
                      "evaporation",
                      "Fraction of water lost per iteration. The retained "
                      "fraction is (1 - rate). Higher rates dry channels "
                      "faster between rain events so deposition wins.",
                      "", o);
    reg.declare_float("hydraulic.flow_constant",      0.4f, 0.0f, 2.0f,
                      "flow constant",
                      "Outflow per pipe per iteration as a fraction of the "
                      "surface-height difference (in metres). Higher "
                      "values are more aggressive but get capped to the "
                      "available water column so the system stays stable.",
                      "", o);
    reg.declare_float("hydraulic.capacity_k",         1.0f, 0.0f, 8.0f,
                      "capacity K",
                      "Sediment carrying capacity = K * slope_m * speed * "
                      "water_m. Higher capacity = sharper, deeper canyons "
                      "(more sediment can be carried away from steep cells). "
                      "Default 1.0 puts capacity well above the per-iter "
                      "erosion cap on mountain slopes, so max_erode_per_iter_m "
                      "is what actually limits carving speed.",
                      "", o);
    reg.declare_float("hydraulic.erode_k",            0.4f, 0.0f, 1.0f,
                      "erode K",
                      "Fraction of (capacity - suspended) eroded per "
                      "iteration when the channel is under-saturated. "
                      "Capped per-iter by max_erode_m.",
                      "", o);
    reg.declare_float("hydraulic.deposit_k",          0.4f, 0.0f, 1.0f,
                      "deposit K",
                      "Fraction of (suspended - capacity) deposited per "
                      "iteration when the channel is over-saturated. "
                      "Higher = sediment drops out faster, creating "
                      "fans / deltas at channel outlets.",
                      "", o);
    reg.declare_float("hydraulic.min_slope_m",        1.0f, 0.0f, 100.0f,
                      "min slope",
                      "Floor for the slope term so flat-ish but flowing "
                      "channels still carry sediment instead of dropping "
                      "everything immediately.",
                      "m/cell", o);
    reg.declare_float("hydraulic.max_erode_per_iter_m", 0.5f, 0.0f, 10.0f,
                      "max erode / iter",
                      "Per-iteration cap on bedrock + sediment erosion in "
                      "any one cell. This is the actual bottleneck on "
                      "canyon depth: at 0.5 m/iter, 1000 iters gives up "
                      "to ~500 m of cumulative bedrock removal in active "
                      "channels. Drop this if explicit-Euler instability "
                      "(oscillation, NaNs) shows up on the steepest cells.",
                      "m", o);
}

void HydraulicErosionPass::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    const int   iters         = reg.get_int  ("hydraulic.iterations");
    const float rain_m        = reg.get_float("hydraulic.rain_per_iter_m");
    const float evap_rate     = reg.get_float("hydraulic.evaporation_rate");
    const float k_flow        = reg.get_float("hydraulic.flow_constant");
    const float capacity_k    = reg.get_float("hydraulic.capacity_k");
    const float erode_k       = reg.get_float("hydraulic.erode_k");
    const float deposit_k     = reg.get_float("hydraulic.deposit_k");
    const float min_slope_m   = reg.get_float("hydraulic.min_slope_m");
    const float max_erode_m   = reg.get_float("hydraulic.max_erode_per_iter_m");

    const auto& g      = state.grid();
    const auto& nbt    = state.neighbors();
    const int   n      = g.n;
    const int   stride = g.stride();
    const int   halo   = CubedSphereGrid::HALO;
    if (n <= 0 || iters <= 0) return;

    //Pass-local buffers. ping-pong water and suspended; flux is recomputed
    //each iteration so a single buffer is enough.
    Field<float>  water_new (g);
    Field<float>  susp_a    (g);
    Field<float>  susp_b    (g);
    Field<float4> flux      (g);
    water_new.zero();
    susp_a.zero();
    susp_b.zero();
    flux.zero();

    Field<float>* water_in_field  = &state.water_column;
    Field<float>* water_out_field = &water_new;
    Field<float>* susp_in_field   = &susp_a;
    Field<float>* susp_out_field  = &susp_b;

    dim3 block(16, 16, 1);
    dim3 grid_dims((n + block.x - 1) / block.x,
                   (n + block.y - 1) / block.y,
                   6);

    float evap_factor = 1.0f - evap_rate;
    if (evap_factor < 0.0f) evap_factor = 0.0f;

    progress.stage("hydraulic erosion");
    progress.fraction(0.0f);

    for (int it = 0; it < iters; ++it) {
        //----- 1. Halo exchange of all read-modified scalar fields -----
        state.bedrock_elevation.halo_exchange(nbt);
        state.sediment_thickness.halo_exchange(nbt);
        water_in_field->halo_exchange(nbt);

        //----- 2. Add rainfall (interior only) -----
        hyd_rain_kernel<<<grid_dims, block>>>(water_in_field->data(),
                                              n, stride, halo, rain_m);
        CUDA_CHECK(cudaGetLastError());

        //Rain only writes interior so re-exchange water halos before flux.
        water_in_field->halo_exchange(nbt);

        //----- 3. Compute per-cell outflow flux -----
        hyd_flux_kernel<<<grid_dims, block>>>(state.bedrock_elevation.data(),
                                              state.sediment_thickness.data(),
                                              water_in_field->data(),
                                              flux.data(),
                                              n, stride, halo, k_flow);
        CUDA_CHECK(cudaGetLastError());
        flux.halo_exchange(nbt);

        //----- 4. Update water column by gather of net flux -----
        hyd_water_update_kernel<<<grid_dims, block>>>(water_in_field->data(),
                                                      flux.data(),
                                                      water_out_field->data(),
                                                      n, stride, halo);
        CUDA_CHECK(cudaGetLastError());

        //----- 5. Erosion / deposition using pre-update water + flux -----
        hyd_erode_kernel<<<grid_dims, block>>>(state.bedrock_elevation.data(),
                                               state.sediment_thickness.data(),
                                               water_in_field->data(),
                                               flux.data(),
                                               susp_in_field->data(),
                                               n, stride, halo,
                                               capacity_k, erode_k, deposit_k,
                                               min_slope_m, max_erode_m);
        CUDA_CHECK(cudaGetLastError());

        //----- 6. Transport suspended sediment with the water flow -----
        susp_in_field->halo_exchange(nbt);
        hyd_transport_kernel<<<grid_dims, block>>>(susp_in_field->data(),
                                                    water_in_field->data(),
                                                    flux.data(),
                                                    susp_out_field->data(),
                                                    n, stride, halo);
        CUDA_CHECK(cudaGetLastError());

        //Promote new water / suspended to the live buffers for next iter.
        std::swap(water_in_field, water_out_field);
        std::swap(susp_in_field,  susp_out_field);

        //----- 7. Evaporation -----
        hyd_evaporate_kernel<<<grid_dims, block>>>(water_in_field->data(),
                                                    n, stride, halo, evap_factor);
        CUDA_CHECK(cudaGetLastError());

        if ((it & 0x1f) == 0) {
            progress.fraction(static_cast<float>(it) / static_cast<float>(iters));
        }
    }

    //If the final live water buffer is not the PlanetState one, copy it back
    //so the field viewer (and any downstream pass) sees the latest result.
    if (water_in_field != &state.water_column) {
        CUDA_CHECK(cudaMemcpy(state.water_column.data(),
                              water_in_field->data(),
                              state.water_column.bytes(),
                              cudaMemcpyDeviceToDevice));
    }
    //Drop remaining suspended sediment into the sediment field so we don't
    //lose mass when the pass finishes. The final suspended buffer is what
    //didn't deposit during the last iter; treat it as deposited now.
    hyd_drain_suspended_kernel<<<grid_dims, block>>>(state.sediment_thickness.data(),
                                                      susp_in_field->data(),
                                                      n, stride, halo);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    progress.fraction(1.0f);

    PB_LOG_INFO("hydraulic",
                "%d iterations (rain=%.3f m, evap=%.3f, kflow=%.2f, "
                "cap=%.2f, erode=%.2f, deposit=%.2f, max_erode=%.3f m)",
                iters,
                static_cast<double>(rain_m),
                static_cast<double>(evap_rate),
                static_cast<double>(k_flow),
                static_cast<double>(capacity_k),
                static_cast<double>(erode_k),
                static_cast<double>(deposit_k),
                static_cast<double>(max_erode_m));
}

}
