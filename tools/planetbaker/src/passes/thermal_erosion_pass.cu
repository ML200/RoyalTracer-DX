#include "passes/thermal_erosion_pass.h"

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/neighbor_table.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pb {

namespace {

constexpr float kPi = 3.14159265358979323846f;

//Effective horizontal scale for slope evaluation. Using a fixed value
//(rather than physical cell size) lets the PLAN-spec repose angles bite at
//our coarse cubed-sphere resolutions. Tunable via params if needed.
constexpr float kSlopeScaleKm = 1.0f;

//Sediment threshold above which the surface is treated as sand-like
//(lower repose). PLAN defines this implicitly via "loose sand ~30 deg".
constexpr float kSandThresholdM = 1.0f;

__device__ __forceinline__ int cell_idx(int face, int i, int j, int stride, int halo) {
    return face * stride * stride + (j + halo) * stride + (i + halo);
}

__device__ __forceinline__ float surface_km(float bed_km, float sed_m) {
    return bed_km + sed_m * 0.001f;
}

__device__ __forceinline__ float repose_tan_for_sed(float sed_m, float tan_sand, float tan_rock) {
    return (sed_m > kSandThresholdM) ? tan_sand : tan_rock;
}

__global__ void thermal_step_kernel(const float* __restrict__ bed,
                                    const float* __restrict__ sed_in,
                                    float* __restrict__ sed_out,
                                    int n, int stride, int halo,
                                    float tan_sand, float tan_rock,
                                    float rate, float scale_km) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;

    int idx_c = cell_idx(face, i, j, stride, halo);
    int idx_l = idx_c - 1;
    int idx_r = idx_c + 1;
    int idx_d = idx_c - stride;
    int idx_u = idx_c + stride;

    float bed_c  = bed[idx_c];
    float sed_c  = sed_in[idx_c];
    float surf_c = surface_km(bed_c, sed_c);
    float tan_c  = repose_tan_for_sed(sed_c, tan_sand, tan_rock);

    int   neighbors[4] = {idx_l, idx_r, idx_d, idx_u};

    //Outflows from c down to lower neighbours (in km).
    float outflow[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float total_outflow_km = 0.0f;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int idx_n = neighbors[k];
        float surf_n = surface_km(bed[idx_n], sed_in[idx_n]);
        float dh = surf_c - surf_n;
        if (dh > 0.0f) {
            float slope = dh / scale_km;
            if (slope > tan_c) {
                float excess = dh - tan_c * scale_km;
                outflow[k]   = rate * excess;
                total_outflow_km += outflow[k];
            }
        }
    }

    //Cap outflow to available sediment.
    float available_km = sed_c * 0.001f;
    if (total_outflow_km > available_km && total_outflow_km > 0.0f) {
        float scl = available_km / total_outflow_km;
        #pragma unroll
        for (int k = 0; k < 4; ++k) outflow[k] *= scl;
        total_outflow_km = available_km;
    }

    //Inflows from higher neighbours (capped by 1/4 of neighbour's sediment).
    float total_inflow_km = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int   idx_n  = neighbors[k];
        float bed_n  = bed[idx_n];
        float sed_n  = sed_in[idx_n];
        float surf_n = surface_km(bed_n, sed_n);
        float dh     = surf_n - surf_c;
        if (dh > 0.0f) {
            float tan_n = repose_tan_for_sed(sed_n, tan_sand, tan_rock);
            float slope = dh / scale_km;
            if (slope > tan_n) {
                float excess = dh - tan_n * scale_km;
                float amount = rate * excess;
                float cap    = sed_n * 0.001f * 0.25f;
                if (amount > cap) amount = cap;
                total_inflow_km += amount;
            }
        }
    }

    float new_sed_m = sed_c + (total_inflow_km - total_outflow_km) * 1000.0f;
    if (new_sed_m < 0.0f) new_sed_m = 0.0f;
    sed_out[idx_c] = new_sed_m;
}

}

void ThermalErosionPass::declare_params(ParamRegistry& reg) const {
    const char* o = name();
    reg.declare_int  ("thermal.iterations",            200,  0,  2000,
                      "iterations", "Relaxation sweeps", "", o);
    reg.declare_float("thermal.angle_repose_sand_deg", 32.0f, 5.0f, 89.0f,
                      "sand repose", "Repose angle for sandy cells", "deg", o);
    reg.declare_float("thermal.angle_repose_rock_deg", 55.0f, 5.0f, 89.0f,
                      "rock repose", "Repose angle for rocky cells", "deg", o);
    reg.declare_float("thermal.transfer_rate",         0.3f,  0.0f, 1.0f,
                      "transfer rate", "Fraction of excess transferred per iteration", "", o);
}

void ThermalErosionPass::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    const int   iters          = reg.get_int  ("thermal.iterations");
    const float sand_deg       = reg.get_float("thermal.angle_repose_sand_deg");
    const float rock_deg       = reg.get_float("thermal.angle_repose_rock_deg");
    const float rate           = reg.get_float("thermal.transfer_rate");

    const float tan_sand = std::tan(sand_deg * kPi / 180.0f);
    const float tan_rock = std::tan(rock_deg * kPi / 180.0f);

    const auto& g      = state.grid();
    const auto& nbt    = state.neighbors();
    const int   n      = g.n;
    const int   stride = g.stride();
    const int   halo   = CubedSphereGrid::HALO;
    if (n <= 0 || iters <= 0) return;

    dim3 block(16, 16, 1);
    dim3 grid_dims((n + block.x - 1) / block.x,
                   (n + block.y - 1) / block.y,
                   6);

    //Bedrock is read-only this pass; one halo exchange covers all iterations.
    state.bedrock_elevation.halo_exchange(nbt);

    //Sediment ping-pong with the existing PlanetState field as the entry
    //point; result is left in state.sediment_thickness at the end.
    Field<float> sed_other(g);
    sed_other.zero();
    Field<float>* a = &state.sediment_thickness;
    Field<float>* b = &sed_other;

    progress.stage("thermal erosion");
    progress.fraction(0.0f);
    for (int it = 0; it < iters; ++it) {
        a->halo_exchange(nbt);
        thermal_step_kernel<<<grid_dims, block>>>(
            state.bedrock_elevation.data(),
            a->data(),
            b->data(),
            n, stride, halo,
            tan_sand, tan_rock, rate, kSlopeScaleKm);
        CUDA_CHECK(cudaGetLastError());
        std::swap(a, b);
        if ((it & 0x1f) == 0) {
            progress.fraction(static_cast<float>(it) / static_cast<float>(iters));
        }
    }

    //If the final result landed in sed_other, copy back into PlanetState.
    if (a != &state.sediment_thickness) {
        CUDA_CHECK(cudaMemcpy(state.sediment_thickness.data(),
                              a->data(),
                              a->bytes(),
                              cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    progress.fraction(1.0f);

    PB_LOG_INFO("thermal", "%d iterations (sand=%.1fdeg, rock=%.1fdeg, rate=%.2f)",
                iters, sand_deg, rock_deg, rate);
}

}
