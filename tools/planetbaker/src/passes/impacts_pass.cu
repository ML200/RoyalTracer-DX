#include "passes/impacts_pass.h"

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace pb {

namespace {

constexpr float kPlanetRadiusKm = 6371.0f;       //Earth-ish; only matters as a length scale
constexpr float kBedrockFloorKm = -25.0f;        //prevents pathological accumulated depression
constexpr int   kCraterCountCap = 10000;
constexpr float kBaseCountPerFlux = 1500.0f;

//(CraterMorphology enum lives in impacts_pass.h now - exposed so callers
//like bake_pass can count basins / multi-ring craters and filter results.)

//====================================
//SFD sampler helpers
//====================================

float random_uniform(std::mt19937_64& rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    return u(rng);
}

void random_unit_vec(std::mt19937_64& rng, float& x, float& y, float& z) {
    //Marsaglia: uniform sample on unit sphere via two uniforms.
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    while (true) {
        float a = u(rng);
        float b = u(rng);
        float s = a * a + b * b;
        if (s < 1.0f && s > 1e-6f) {
            float k = 2.0f * std::sqrt(1.0f - s);
            x = a * k;
            y = b * k;
            z = 1.0f - 2.0f * s;
            return;
        }
    }
}

std::uint32_t classify_morphology(float D, float t_sc, float t_basin, float t_multi) {
    if (D < t_sc)    return MorphSimple;
    if (D < t_basin) return MorphComplex;
    if (D < t_multi) return MorphBasin;
    return MorphMultiring;
}

} //namespace

//====================================
//Public SFD sampler (declared in impacts_pass.h)
//====================================
std::vector<Crater> sample_craters(const ImpactsParams& p) {
    int n_craters = static_cast<int>(p.flux_multiplier * kBaseCountPerFlux);
    if (n_craters < 0) n_craters = 0;
    if (n_craters > kCraterCountCap) n_craters = kCraterCountCap;

    std::vector<Crater> out(n_craters);
    if (n_craters == 0) return out;

    std::mt19937_64 rng(p.seed);

    const float b      = p.size_distribution_exponent;
    const float Dmin_p = std::pow(p.min_diameter_km, -b);
    const float Dmax_p = std::pow(p.max_diameter_km, -b);

    for (auto& c : out) {
        random_unit_vec(rng, c.pos_x, c.pos_y, c.pos_z);

        float u = random_uniform(rng, 0.0f, 1.0f);
        float inv = Dmin_p - u * (Dmin_p - Dmax_p);
        c.diameter_km = std::pow(inv, -1.0f / b);
        c.diameter_km = std::clamp(c.diameter_km, p.min_diameter_km, p.max_diameter_km);

        float ua = random_uniform(rng, 0.0f, 1.0f);
        c.age_myr = p.epoch_start_myr - ua * (p.epoch_start_myr - p.epoch_end_myr);

        c.morphology = classify_morphology(c.diameter_km,
                                           p.simple_complex_threshold_km,
                                           p.basin_threshold_km,
                                           p.multiring_threshold_km);
        c.ejecta_extent = p.ejecta_extent_radii;
    }

    //Oldest first; newer craters splat on top and partially erase older ones.
    std::sort(out.begin(), out.end(),
              [](const Crater& a, const Crater& b) { return a.age_myr > b.age_myr; });

    return out;
}

//====================================
//Public param loader (declared in impacts_pass.h)
//====================================
ImpactsParams load_impacts_params(const ParamRegistry& reg) {
    ImpactsParams p;
    p.flux_multiplier             = reg.get_float("impacts.flux_multiplier");
    p.size_distribution_exponent  = reg.get_float("impacts.size_distribution_exponent");
    p.min_diameter_km             = reg.get_float("impacts.min_diameter_km");
    p.max_diameter_km             = reg.get_float("impacts.max_diameter_km");
    p.epoch_start_myr             = reg.get_float("impacts.epoch_start_myr");
    p.epoch_end_myr               = reg.get_float("impacts.epoch_end_myr");
    p.simple_complex_threshold_km = reg.get_float("impacts.simple_complex_threshold_km");
    p.basin_threshold_km          = reg.get_float("impacts.basin_threshold_km");
    p.multiring_threshold_km      = reg.get_float("impacts.multiring_threshold_km");
    p.ejecta_extent_radii         = reg.get_float("impacts.ejecta_extent_radii");
    p.seed                        = static_cast<std::uint64_t>(static_cast<std::uint32_t>(reg.get_int("impacts.seed")));
    return p;
}

namespace {

//====================================
//Splat kernel
//====================================

//Depth coefficients are tuned to match Pike's law and observed planetary
//crater depth/diameter ratios:
//  Simple   d/D ~ 0.10  (Earth/Moon small craters)
//  Complex  d/D ~ 0.025 (transitions to ~0.04 for D~100km, lower for bigger)
//  Basin    d/D ~ 0.008 (Hellas: 2300 km / 9 km deep, ratio 0.004)
//  Multiring d/D ~ 0.004
//Earlier coefficients (0.20 / 0.10 / 0.04 / 0.03) gave Complex craters
//deeper than the tallest mountains, which read as wrong relative to the
//mountain layer's typical 4-8 km peaks.
__device__ __forceinline__ float crater_depth_at(float D, float r_norm, std::uint32_t morph) {
    if (morph == MorphSimple) {
        float bowl = D * 0.10f * (1.0f - r_norm * r_norm);
        return bowl;
    }
    if (morph == MorphComplex) {
        float bowl = D * 0.025f * (1.0f - r_norm * r_norm);
        float peak = D * 0.010f * expf(-(r_norm * 5.0f) * (r_norm * 5.0f));
        return bowl - peak;
    }
    if (morph == MorphBasin) {
        float bowl = D * 0.008f * (1.0f - r_norm * r_norm);
        //One inner ring (sub-basin bump)
        float ring = D * 0.001f * cosf(r_norm * 6.2831853f * 2.0f);
        return bowl - ring;
    }
    //Multi-ring basin: very shallow with 2 rings
    float bowl  = D * 0.004f * (1.0f - r_norm * r_norm);
    float ring1 = D * 0.0006f * cosf(r_norm * 6.2831853f * 2.5f);
    float ring2 = D * 0.0004f * cosf(r_norm * 6.2831853f * 4.0f);
    return bowl - ring1 - ring2;
}

__device__ __forceinline__ float crater_rim_height(float D, std::uint32_t morph) {
    if (morph == MorphSimple)    return D * 0.020f;
    if (morph == MorphComplex)   return D * 0.005f;
    if (morph == MorphBasin)     return D * 0.002f;
    return D * 0.0007f;
}

__device__ __forceinline__ float ejecta_thickness_at(float D, float e_norm, std::uint32_t morph) {
    float max_e = crater_rim_height(D, morph) * 0.5f;
    float falloff = 1.0f - e_norm;
    if (falloff < 0.0f) falloff = 0.0f;
    return max_e * falloff * falloff * falloff;
}

__global__ void impacts_kernel(float* __restrict__ bedrock,
                               float* __restrict__ sediment,
                               std::uint8_t* __restrict__ crust_type,
                               int n, int stride, int halo,
                               const Crater* __restrict__ craters,
                               int crater_count,
                               float planet_radius_km) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;

    float u = ((static_cast<float>(i) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    float v = ((static_cast<float>(j) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    Vec3f p = face_uv_to_sphere(face, u, v);

    int idx = face * stride * stride + (j + halo) * stride + (i + halo);
    float bed  = bedrock[idx];
    float sed  = sediment[idx];
    std::uint8_t type = crust_type[idx];

    for (int k = 0; k < crater_count; ++k) {
        Crater c = craters[k];
        float dot_p = p.x * c.pos_x + p.y * c.pos_y + p.z * c.pos_z;
        if (dot_p > 1.0f) dot_p = 1.0f;
        if (dot_p < -1.0f) dot_p = -1.0f;
        float ang_dist = acosf(dot_p);

        float r_rad     = c.diameter_km * 0.5f / planet_radius_km;
        float ejecta_rad = r_rad * c.ejecta_extent;
        if (ang_dist > ejecta_rad) continue;

        if (ang_dist <= r_rad) {
            float r_norm = ang_dist / r_rad;
            float depth  = crater_depth_at(c.diameter_km, r_norm, c.morphology);
            bed -= depth;
            if (r_norm < 0.7f) type = 3;  //CrustImpactMelt
        } else {
            float e_norm = (ang_dist - r_rad) / (ejecta_rad - r_rad);
            float rim    = crater_rim_height(c.diameter_km, c.morphology);
            float rim_decay = expf(-e_norm * 3.0f);
            bed += rim * rim_decay * 0.5f;
            sed += ejecta_thickness_at(c.diameter_km, e_norm, c.morphology);
        }

        if (bed < kBedrockFloorKm) bed = kBedrockFloorKm;
    }

    bedrock[idx]    = bed;
    sediment[idx]   = sed;
    crust_type[idx] = type;
}

//====================================
//Bake-only single-face splat. Identical per-pixel evaluation to impacts_kernel,
//but writes to a tight n*n bedrock buffer (no halo, no stride, no padding)
//for ONE face, and folds ejecta into the same bedrock buffer (since the
//direct-synthesis bake path doesn't carry a separate sediment field).
//Bedrock is read-modify-written; the buffer must already contain the bedrock
//noise output that this kernel will stamp craters on top of.
//====================================
__global__ void bake_impacts_kernel(float* __restrict__ dst,
                                    int n, int face,
                                    const Crater* __restrict__ craters,
                                    int crater_count,
                                    float planet_radius_km) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    float u = ((static_cast<float>(i) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    float v = ((static_cast<float>(j) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    Vec3f p = face_uv_to_sphere(face, u, v);

    const std::size_t idx = static_cast<std::size_t>(j) * n + i;
    float bed = dst[idx];

    for (int k = 0; k < crater_count; ++k) {
        Crater c = craters[k];
        float dot_p = p.x * c.pos_x + p.y * c.pos_y + p.z * c.pos_z;
        if (dot_p > 1.0f) dot_p = 1.0f;
        if (dot_p < -1.0f) dot_p = -1.0f;
        float ang_dist = acosf(dot_p);

        float r_rad      = c.diameter_km * 0.5f / planet_radius_km;
        float ejecta_rad = r_rad * c.ejecta_extent;
        if (ang_dist > ejecta_rad) continue;

        if (ang_dist <= r_rad) {
            float r_norm = ang_dist / r_rad;
            float depth  = crater_depth_at(c.diameter_km, r_norm, c.morphology);
            bed -= depth;
        } else {
            float e_norm = (ang_dist - r_rad) / (ejecta_rad - r_rad);
            float rim    = crater_rim_height(c.diameter_km, c.morphology);
            float rim_decay = expf(-e_norm * 3.0f);
            //Fold ejecta into bedrock directly: the direct-synthesis bake
            //has no sediment field, so the ejecta thickness becomes part of
            //the bedrock elevation. Renderer doesn't care which "field"
            //the rim came from - only the final elevation matters.
            bed += rim * rim_decay * 0.5f;
            bed += ejecta_thickness_at(c.diameter_km, e_norm, c.morphology);
        }

        if (bed < kBedrockFloorKm) bed = kBedrockFloorKm;
    }

    dst[idx] = bed;
}

}

//====================================
//Public bake-only host wrapper (declared in impacts_pass.h)
//====================================
void bake_impacts_face(int face, int n,
                       const Crater* craters_device,
                       int crater_count,
                       float planet_radius_km,
                       float* dst_device) {
    if (face < 0 || face >= 6) return;
    if (n <= 0 || !dst_device)  return;
    if (crater_count <= 0)      return;

    dim3 block(16, 16, 1);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y,
              1);
    bake_impacts_kernel<<<grid, block>>>(dst_device, n, face,
                                         craters_device, crater_count,
                                         planet_radius_km);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//====================================
//Pass implementation
//====================================

void ImpactsPass::declare_params(ParamRegistry& reg) const {
    const char* o = name();
    //v3 defaults (2026-05-26): the previous defaults (flux=1, b=2, Dmin=1,
    //Dmax=1500) produced a power-law SFD heavily biased to sub-2 km craters
    //that are invisible at the bake's effective sub-km/pixel preview
    //resolution AND statistically yield zero basins per 1500 samples.
    //New defaults target a visible-from-preview population: flatter SFD
    //(b=1.8) over a narrower diameter range (Dmin=15..Dmax=800) so the
    //sampled mass lands in the basin-visible band, and a 2x flux for
    //~3000 craters total. Expected counts: ~36 basins, ~90 craters
    //100-150 km, ~265 craters 50-100 km, rest smaller but still pixel-sized.
    reg.declare_float("impacts.flux_multiplier",             2.0f,  0.0f, 10.0f,
                      "flux multiplier",
                      "Scales total crater count. 1.0 = 1500 craters, 2.0 = 3000, "
                      "etc., capped at kCraterCountCap (10000).", "", o);
    reg.declare_float("impacts.size_distribution_exponent",  1.8f,  1.0f, 4.0f,
                      "SFD exponent",
                      "Power-law slope of crater diameters. P(D > X) ~ X^-b. "
                      "b=2.0 = Mars/Moon realistic but heavily biased to small "
                      "craters that are sub-pixel in the 4k preview. b=1.5..1.8 "
                      "flattens the SFD so visible-sized craters dominate the "
                      "stamped population.", "", o);
    reg.declare_float("impacts.min_diameter_km",             15.0f, 0.1f, 100.0f,
                      "min diameter",
                      "Smallest crater sampled. Default raised from 1 km to 15 km "
                      "so the sampled population skips the sub-pixel band on a "
                      "4k preview (~2.4 km/pixel at 90deg face).", "km", o);
    reg.declare_float("impacts.max_diameter_km",             800.0f, 100.0f, 3000.0f,
                      "max diameter",
                      "Largest crater sampled. Default lowered from 1500 to 800 "
                      "so the SFD concentrates more samples in the visible 50-300 "
                      "km band rather than spreading thin to 1500 km giants that "
                      "almost never get sampled with b>=1.8.", "km", o);
    reg.declare_float("impacts.epoch_start_myr",             4000.0f, 100.0f, 4500.0f,
                      "epoch start", "Oldest crater age", "Myr", o);
    reg.declare_float("impacts.epoch_end_myr",               100.0f,  0.0f, 4000.0f,
                      "epoch end", "Youngest crater age", "Myr", o);
    reg.declare_float("impacts.simple_complex_threshold_km", 3.0f,    0.5f, 50.0f,
                      "simple/complex threshold", "Diameter at which simple becomes complex", "km", o);
    reg.declare_float("impacts.basin_threshold_km",          150.0f,  20.0f, 500.0f,
                      "basin threshold", "Diameter at which complex becomes basin", "km", o);
    reg.declare_float("impacts.multiring_threshold_km",      1000.0f, 200.0f, 2000.0f,
                      "multi-ring threshold", "Diameter at which basin becomes multi-ring", "km", o);
    reg.declare_float("impacts.ejecta_extent_radii",         2.5f,    1.0f, 5.0f,
                      "ejecta extent", "Ejecta reach in crater radii", "", o);
    reg.declare_int  ("impacts.seed",                        424242,
                      std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max(),
                      "seed", "RNG seed (cast to uint64)", "", o);
}

void ImpactsPass::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    const ImpactsParams p = load_impacts_params(reg);

    progress.stage("sample craters");
    progress.fraction(0.0f);
    std::vector<Crater> craters = sample_craters(p);
    int basin_count = 0;
    int multi_count = 0;
    for (const auto& c : craters) {
        if (c.morphology == MorphBasin)     ++basin_count;
        if (c.morphology == MorphMultiring) ++multi_count;
    }
    PB_LOG_INFO("impacts", "%zu craters (basins=%d, multi-ring=%d)",
                craters.size(), basin_count, multi_count);

    if (craters.empty()) {
        progress.fraction(1.0f);
        return;
    }

    progress.stage("splat");
    progress.fraction(0.0f);

    DeviceBuffer<Crater> d_craters(craters.size());
    d_craters.upload(craters);

    const auto& g = state.bedrock_elevation.grid();
    const int n      = g.n;
    const int stride = g.stride();
    const int halo   = CubedSphereGrid::HALO;

    dim3 block(16, 16, 1);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y,
              6);

    impacts_kernel<<<grid, block>>>(state.bedrock_elevation.data(),
                                    state.sediment_thickness.data(),
                                    state.crust_type.data(),
                                    n, stride, halo,
                                    d_craters.data(),
                                    static_cast<int>(craters.size()),
                                    kPlanetRadiusKm);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    progress.fraction(1.0f);
}

}
