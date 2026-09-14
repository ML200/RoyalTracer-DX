#include "passes/surface_color_pass.h"

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace pb {

namespace {

struct ColorParams {
    //Base palette anchors. Linear-space sRGB-ish (no gamma). Tuned so the
    //blended planet reads as Mars from orbit: cool dark mare in low areas,
    //orange-red standard plains, dusty tan highlands.
    float low_r,  low_g,  low_b;
    float mid_r,  mid_g,  mid_b;
    float high_r, high_g, high_b;

    float low_elev_km;     // anchor for "low" colour
    float mid_elev_km;     // anchor for "mid" colour
    float high_elev_km;    // anchor for "high" colour

    //Slope -> exposed rock blend. Slope here is m-of-surface per cell;
    //the conversion is purely numerical and rebalanced via these knobs.
    float rock_r, rock_g, rock_b;
    float slope_lo_m_per_cell;
    float slope_hi_m_per_cell;

    //Sediment -> ejecta / dust blend.
    float ejecta_r, ejecta_g, ejecta_b;
    float sediment_lo_m;
    float sediment_hi_m;
    float sediment_max_blend;

    //Impact-melt darkening (crust_type == 1).
    float impact_melt_factor;   // 0 = unchanged, 1 = full impact_melt_color
    float impact_r, impact_g, impact_b;

    //Polar ice and altitude ice.
    float ice_r, ice_g, ice_b;
    float polar_lat_lo_deg;
    float polar_lat_hi_deg;
    float ice_alt_lo_km;
    float ice_alt_hi_km;

    //Border noise: perturbs the polar-lat / mountain-alt threshold inputs
    //so the resulting cap edge is a wandering scalloped line rather than a
    //clean latitude / elevation contour. Different seeds keep polar caps
    //and mountain caps decorrelated.
    float polar_noise_freq;
    float polar_noise_amp_deg;
    int   polar_noise_octaves;
    float alt_noise_freq;
    float alt_noise_amp_km;
    int   alt_noise_octaves;
    std::uint32_t border_seed;

    //Large-scale albedo provinces (v3): dust-mantled bright vs basaltic dark.
    float province_freq;
    int   province_octaves;
    float province_lo;
    float province_hi;
    float province_dark_mul;
    float province_bright_mul;
    float province_warm;
};

//Minimal noise primitives. Inlined rather than depending on bedrock_noise's
//anonymous namespace so this pass stays self-contained.
__device__ __forceinline__ std::uint32_t d_pcg32(std::uint32_t x) {
    std::uint32_t state = x * 747796405u + 2891336453u;
    std::uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ std::uint32_t d_h3(int x, int y, int z, std::uint32_t seed) {
    std::uint32_t h = seed;
    h = d_pcg32(h ^ static_cast<std::uint32_t>(x * 0x27d4eb2du));
    h = d_pcg32(h ^ static_cast<std::uint32_t>(y * 0x165667b1u));
    h = d_pcg32(h ^ static_cast<std::uint32_t>(z * 0x9e3779b9u));
    return h;
}

__device__ __forceinline__ float d_quint(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ float d_g3(std::uint32_t h, float x, float y, float z) {
    std::uint32_t k = h & 15u;
    float u = (k < 8u) ? x : y;
    float v = (k < 4u) ? y : ((k == 12u || k == 14u) ? x : z);
    float gu = ((k & 1u) == 0u) ? u : -u;
    float gv = ((k & 2u) == 0u) ? v : -v;
    return gu + gv;
}

__device__ float d_perlin3(float x, float y, float z, std::uint32_t seed) {
    int   x0 = static_cast<int>(floorf(x));
    int   y0 = static_cast<int>(floorf(y));
    int   z0 = static_cast<int>(floorf(z));
    float fx = x - static_cast<float>(x0);
    float fy = y - static_cast<float>(y0);
    float fz = z - static_cast<float>(z0);
    float sx = d_quint(fx);
    float sy = d_quint(fy);
    float sz = d_quint(fz);

    float n000 = d_g3(d_h3(x0,     y0,     z0,     seed), fx,        fy,        fz);
    float n100 = d_g3(d_h3(x0 + 1, y0,     z0,     seed), fx - 1.0f, fy,        fz);
    float n010 = d_g3(d_h3(x0,     y0 + 1, z0,     seed), fx,        fy - 1.0f, fz);
    float n110 = d_g3(d_h3(x0 + 1, y0 + 1, z0,     seed), fx - 1.0f, fy - 1.0f, fz);
    float n001 = d_g3(d_h3(x0,     y0,     z0 + 1, seed), fx,        fy,        fz - 1.0f);
    float n101 = d_g3(d_h3(x0 + 1, y0,     z0 + 1, seed), fx - 1.0f, fy,        fz - 1.0f);
    float n011 = d_g3(d_h3(x0,     y0 + 1, z0 + 1, seed), fx,        fy - 1.0f, fz - 1.0f);
    float n111 = d_g3(d_h3(x0 + 1, y0 + 1, z0 + 1, seed), fx - 1.0f, fy - 1.0f, fz - 1.0f);

    float a = n000 + sx * (n100 - n000);
    float b = n010 + sx * (n110 - n010);
    float c = n001 + sx * (n101 - n001);
    float d = n011 + sx * (n111 - n011);
    float e = a   + sy * (b   - a  );
    float f = c   + sy * (d   - c  );
    return e + sz * (f - e);
}

//Signed fBm in roughly [-1, +1]. Caller scales by an amplitude.
__device__ float d_fbm_signed(float x, float y, float z,
                              int octaves, float base_freq,
                              std::uint32_t seed) {
    if (octaves <= 0) return 0.0f;
    float sum = 0.0f, amp = 1.0f, norm = 0.0f, f = base_freq;
    std::uint32_t s = seed;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * d_perlin3(x * f, y * f, z * f, s);
        norm += amp;
        amp  *= 0.5f;
        f    *= 2.0f;
        s     = s * 1664525u + 1013904223u;
    }
    return (norm > 0.0f) ? (sum / norm) : 0.0f;
}

__device__ __forceinline__ int cell_idx(int face, int i, int j, int stride, int halo) {
    return face * stride * stride + (j + halo) * stride + (i + halo);
}

__device__ __forceinline__ float d_smooth(float a, float b, float t) {
    float w = b - a;
    if (w <= 0.0f) return (t >= b) ? 1.0f : 0.0f;
    float k = (t - a) / w;
    if (k < 0.0f) k = 0.0f;
    if (k > 1.0f) k = 1.0f;
    return k * k * (3.0f - 2.0f * k);
}

__device__ __forceinline__ float d_mix(float a, float b, float t) {
    return a + (b - a) * t;
}

__device__ __forceinline__ void mix3(float& r, float& g, float& b,
                                     float dr, float dg, float db,
                                     float t) {
    r = d_mix(r, dr, t);
    g = d_mix(g, dg, t);
    b = d_mix(b, db, t);
}

__global__ void surface_color_kernel(float4* __restrict__ out,
                                     const float*        __restrict__ bed,
                                     const float*        __restrict__ sed,
                                     const std::uint8_t* __restrict__ crust_type,
                                     int n, int stride, int halo,
                                     ColorParams P) {
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;
    int c  = cell_idx(face, i, j, stride, halo);
    int pi = c + 1;
    int mi = c - 1;
    int pj = c + stride;
    int mj = c - stride;

    float h_km = bed[c];
    float s_m  = sed[c];

    //Base palette: piecewise blend through low/mid/high anchors.
    float t_low_mid = d_smooth(P.low_elev_km, P.mid_elev_km, h_km);
    float r = d_mix(P.low_r, P.mid_r, t_low_mid);
    float g = d_mix(P.low_g, P.mid_g, t_low_mid);
    float b = d_mix(P.low_b, P.mid_b, t_low_mid);
    float t_mid_hi  = d_smooth(P.mid_elev_km, P.high_elev_km, h_km);
    r = d_mix(r, P.high_r, t_mid_hi);
    g = d_mix(g, P.high_g, t_mid_hi);
    b = d_mix(b, P.high_b, t_mid_hi);

    //Slope from central diff of surface = bed*1000 + sed (meters).
    float surf_pi = bed[pi] * 1000.0f + sed[pi];
    float surf_mi = bed[mi] * 1000.0f + sed[mi];
    float surf_pj = bed[pj] * 1000.0f + sed[pj];
    float surf_mj = bed[mj] * 1000.0f + sed[mj];
    float gx = 0.5f * (surf_pi - surf_mi);
    float gy = 0.5f * (surf_pj - surf_mj);
    float slope = sqrtf(gx * gx + gy * gy);

    float rock_blend = d_smooth(P.slope_lo_m_per_cell,
                                P.slope_hi_m_per_cell,
                                slope);
    mix3(r, g, b, P.rock_r, P.rock_g, P.rock_b, rock_blend);

    //Sediment -> ejecta/dust. The blend is gated by sediment_max_blend so
    //even heavy sediment cover doesn't entirely whiteout the surface tint.
    float sed_blend = d_smooth(P.sediment_lo_m, P.sediment_hi_m, s_m)
                    * P.sediment_max_blend;
    mix3(r, g, b, P.ejecta_r, P.ejecta_g, P.ejecta_b, sed_blend);

    //Impact-melt darkening for cells the ImpactsPass tagged as in-cavity.
    //ImpactsPass writes crust_type = 3 ("impact-melt") in the inner 0.7r of
    //each crater; outer ejecta is captured by the sediment-tint above.
    if (crust_type[c] == 3u) {
        mix3(r, g, b, P.impact_r, P.impact_g, P.impact_b, P.impact_melt_factor);
    }

    //Latitude from the cubed-sphere cell direction.
    float u = ((static_cast<float>(i) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    float v = ((static_cast<float>(j) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    Vec3f p = face_uv_to_sphere(face, u, v);
    float lat_deg = asinf(fmaxf(-1.0f, fminf(1.0f, p.y))) * (180.0f / 3.14159265358979323846f);
    float abs_lat = fabsf(lat_deg);

    //Large-scale albedo provinces (v3). After craters, Mars's dominant orbital
    //signal is its albedo map - bright dust-mantled regions (Arabia, Tharsis)
    //vs dark wind-scoured basaltic regions (Syrtis Major, Acidalia) - and it is
    //largely DECORRELATED from elevation, so the elevation palette above cannot
    //produce it. A low-freq fBm on the cell direction scales surface value
    //(dust = brighter, basalt = darker) and shifts warmth (dust warm, basalt
    //cool). Multiplicative, so it works at any base albedo level; applied
    //before the ice mix so caps still paint white on top.
    float prov_raw  = 0.5f + 0.5f * d_fbm_signed(p.x, p.y, p.z,
                                                 P.province_octaves,
                                                 P.province_freq,
                                                 P.border_seed + 911u);
    float province  = d_smooth(P.province_lo, P.province_hi, prov_raw);
    float prov_mul  = d_mix(P.province_dark_mul, P.province_bright_mul, province);
    r *= prov_mul; g *= prov_mul; b *= prov_mul;
    float prov_warm = (province - 0.5f) * 2.0f * P.province_warm;
    r *= (1.0f + prov_warm);
    b *= (1.0f - prov_warm);

    //Border noise. Two independent 3D fBm samples on the cell direction,
    //decorrelated via different seed offsets. Push the THRESHOLD INPUT
    //instead of the smoothstep output so a tight band still produces a
    //sharp edge - the edge just wanders inward / outward along the noise.
    float polar_noise = d_fbm_signed(p.x, p.y, p.z,
                                     P.polar_noise_octaves,
                                     P.polar_noise_freq,
                                     P.border_seed + 17u);
    float alt_noise   = d_fbm_signed(p.x, p.y, p.z,
                                     P.alt_noise_octaves,
                                     P.alt_noise_freq,
                                     P.border_seed + 251u);

    float lat_perturbed = abs_lat + polar_noise * P.polar_noise_amp_deg;
    float h_perturbed   = h_km    + alt_noise   * P.alt_noise_amp_km;

    float polar_ice = d_smooth(P.polar_lat_lo_deg, P.polar_lat_hi_deg, lat_perturbed);
    float alt_ice   = d_smooth(P.ice_alt_lo_km,    P.ice_alt_hi_km,    h_perturbed);
    float ice = fmaxf(polar_ice, alt_ice);
    mix3(r, g, b, P.ice_r, P.ice_g, P.ice_b, ice);

    //Clamp + store.
    if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
    if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
    if (b < 0.0f) b = 0.0f; if (b > 1.0f) b = 1.0f;
    out[c] = make_float4(r, g, b, 1.0f);
}

}

void SurfaceColorPass::declare_params(ParamRegistry& reg) const {
    const char* o = name();

    reg.declare_float("color.low_elev_km",   -3.0f, -10.0f, 10.0f,
                      "low elev anchor",
                      "Elevation at which the palette reaches the low-anchor "
                      "colour. Values below this saturate to that anchor.",
                      "km", o);
    reg.declare_float("color.mid_elev_km",    1.0f, -10.0f, 10.0f,
                      "mid elev anchor", "Elevation for the mid-anchor colour.",
                      "km", o);
    reg.declare_float("color.high_elev_km",   6.0f, -10.0f, 20.0f,
                      "high elev anchor", "Elevation for the high-anchor colour.",
                      "km", o);

    //Mars-like palette, linear-ish. Dark mare-red -> standard red-brown ->
    //dusty highland tan.
    //v3: muted true-colour Mars brown (less orange than the v2 anchors).
    //Saturation pulled down toward MRO/HiRISE white-balanced tan; the new
    //albedo-province layer below re-adds the large-scale bright/dark variation.
    reg.declare_float("color.low_r",  0.36f, 0.0f, 1.0f, "low R",  "Low-anchor red",   "", o);
    reg.declare_float("color.low_g",  0.22f, 0.0f, 1.0f, "low G",  "Low-anchor green", "", o);
    reg.declare_float("color.low_b",  0.16f, 0.0f, 1.0f, "low B",  "Low-anchor blue",  "", o);
    reg.declare_float("color.mid_r",  0.52f, 0.0f, 1.0f, "mid R",  "Mid-anchor red",   "", o);
    reg.declare_float("color.mid_g",  0.38f, 0.0f, 1.0f, "mid G",  "Mid-anchor green", "", o);
    reg.declare_float("color.mid_b",  0.29f, 0.0f, 1.0f, "mid B",  "Mid-anchor blue",  "", o);
    reg.declare_float("color.high_r", 0.64f, 0.0f, 1.0f, "high R", "High-anchor red",  "", o);
    reg.declare_float("color.high_g", 0.54f, 0.0f, 1.0f, "high G", "High-anchor green","", o);
    reg.declare_float("color.high_b", 0.45f, 0.0f, 1.0f, "high B", "High-anchor blue", "", o);

    //Rock (exposed bedrock on steep slopes).
    reg.declare_float("color.rock_r", 0.38f, 0.0f, 1.0f, "rock R", "Steep-rock red",   "", o);
    reg.declare_float("color.rock_g", 0.32f, 0.0f, 1.0f, "rock G", "Steep-rock green", "", o);
    reg.declare_float("color.rock_b", 0.28f, 0.0f, 1.0f, "rock B", "Steep-rock blue",  "", o);
    reg.declare_float("color.slope_lo_m_per_cell", 200.0f, 0.0f, 5000.0f,
                      "slope rock-blend lo",
                      "Slope (m surface-height per cell) where the exposed-"
                      "rock blend begins.",
                      "m", o);
    reg.declare_float("color.slope_hi_m_per_cell", 1200.0f, 0.0f, 8000.0f,
                      "slope rock-blend hi",
                      "Slope where the exposed-rock blend reaches full strength.",
                      "m", o);

    //Ejecta / dust (driven by sediment_thickness).
    reg.declare_float("color.ejecta_r", 0.62f, 0.0f, 1.0f, "ejecta R", "Ejecta blanket red",   "", o);
    reg.declare_float("color.ejecta_g", 0.52f, 0.0f, 1.0f, "ejecta G", "Ejecta blanket green", "", o);
    reg.declare_float("color.ejecta_b", 0.43f, 0.0f, 1.0f, "ejecta B", "Ejecta blanket blue",  "", o);
    reg.declare_float("color.sediment_lo_m", 0.2f, 0.0f, 50.0f,
                      "sediment blend lo",
                      "Sediment depth (m) at which the ejecta tint begins to show.",
                      "m", o);
    reg.declare_float("color.sediment_hi_m", 6.0f, 0.0f, 100.0f,
                      "sediment blend hi",
                      "Sediment depth at which the ejecta tint reaches its cap.",
                      "m", o);
    reg.declare_float("color.sediment_max_blend", 0.65f, 0.0f, 1.0f,
                      "sediment max blend",
                      "Maximum proportion of ejecta tint to mix in even at "
                      "saturating sediment depth; prevents craters from going "
                      "fully tan and overpowering the base palette.",
                      "", o);

    //Impact-melt darkening.
    reg.declare_float("color.impact_r", 0.22f, 0.0f, 1.0f, "impact-melt R", "Crater-floor melt red",   "", o);
    reg.declare_float("color.impact_g", 0.14f, 0.0f, 1.0f, "impact-melt G", "Crater-floor melt green", "", o);
    reg.declare_float("color.impact_b", 0.10f, 0.0f, 1.0f, "impact-melt B", "Crater-floor melt blue",  "", o);
    reg.declare_float("color.impact_melt_factor", 0.6f, 0.0f, 1.0f,
                      "impact melt mix",
                      "Blend strength of the impact-melt colour where the "
                      "ImpactsPass tagged crust_type == 1.",
                      "", o);

    //Ice (polar caps + mountain caps).
    reg.declare_float("color.ice_r", 0.94f, 0.0f, 1.0f, "ice R", "Ice cap red",   "", o);
    reg.declare_float("color.ice_g", 0.95f, 0.0f, 1.0f, "ice G", "Ice cap green", "", o);
    reg.declare_float("color.ice_b", 0.98f, 0.0f, 1.0f, "ice B", "Ice cap blue",  "", o);
    reg.declare_float("color.polar_lat_lo_deg", 66.0f, 0.0f, 90.0f,
                      "polar ice lat lo",
                      "Absolute latitude (deg) where polar ice begins to mix "
                      "in. Bands narrower than the noise amplitude give a "
                      "sharp scalloped edge; wide bands airbrush the cap.",
                      "deg", o);
    reg.declare_float("color.polar_lat_hi_deg", 70.0f, 0.0f, 90.0f,
                      "polar ice lat hi",
                      "Absolute latitude where polar ice is fully opaque.",
                      "deg", o);
    reg.declare_float("color.ice_alt_lo_km", 8.0f, 0.0f, 20.0f,
                      "mountain ice alt lo",
                      "Elevation (km) where mountain-top ice begins to mix in.",
                      "km", o);
    reg.declare_float("color.ice_alt_hi_km", 9.5f, 0.0f, 25.0f,
                      "mountain ice alt hi",
                      "Elevation where mountain-top ice is fully opaque.",
                      "km", o);

    //Border perturbation. Adds 3D fBm noise to the threshold INPUT (not the
    //output), so the cap edge wanders along the noise instead of being a
    //clean latitude / elevation contour. Keeping the smoothstep band tight
    //(few degrees / hundreds of m) means the wandering edge still reads as
    //sharp, just irregular - lobate / scalloped, like Mars cap edges.
    reg.declare_float("color.polar_noise_freq", 6.0f, 0.1f, 50.0f,
                      "polar border noise freq",
                      "3D fBm base frequency on the cell direction. Higher = "
                      "finer scallops along the cap edge. 6 gives a few "
                      "wavelengths per quadrant of latitude.",
                      "", o);
    reg.declare_float("color.polar_noise_amp_deg", 7.0f, 0.0f, 30.0f,
                      "polar border noise amp",
                      "Degrees of latitude that the noise can push the cap "
                      "edge in or out. Combined with the tight smoothstep "
                      "band, this is where the 'random border' comes from.",
                      "deg", o);
    reg.declare_int  ("color.polar_noise_octaves", 4, 1, 8,
                      "polar border noise octaves",
                      "Number of fBm octaves on the polar-border noise.",
                      "", o);
    reg.declare_float("color.alt_noise_freq", 8.0f, 0.1f, 80.0f,
                      "mountain ice noise freq",
                      "3D fBm base frequency for the mountain-ice border "
                      "perturbation.",
                      "", o);
    reg.declare_float("color.alt_noise_amp_km", 1.5f, 0.0f, 8.0f,
                      "mountain ice noise amp",
                      "Kilometres of elevation perturbation on the mountain-"
                      "cap threshold input. With a 1.5 km amp on an 8.0-9.5 "
                      "smoothstep band you get ice patches well below the "
                      "main cap and bare-rock pockets above it.",
                      "km", o);
    reg.declare_int  ("color.alt_noise_octaves", 4, 1, 8,
                      "mountain ice noise octaves",
                      "Number of fBm octaves on the mountain-ice noise.",
                      "", o);
    reg.declare_int  ("color.border_seed", 0x2D7Du,
                      std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max(),
                      "border noise seed",
                      "Seed for the polar + mountain border noises. Different "
                      "seed = different ragged-edge pattern at the same "
                      "smoothstep + amp settings.",
                      "", o);

    //Large-scale albedo provinces (v3). A low-freq fBm decorrelated from
    //elevation that scales surface value + warmth, so the planet reads as
    //bright dust-mantled regions vs dark basaltic regions from orbit - the
    //dominant Mars albedo signal an elevation-only palette can't make.
    reg.declare_float("color.province_freq", 2.0f, 0.1f, 20.0f,
                      "province freq",
                      "Base frequency of the albedo-province fBm on the cell "
                      "direction. ~2 gives several continent-scale provinces "
                      "per face; lower = bigger provinces.",
                      "", o);
    reg.declare_int  ("color.province_octaves", 4, 1, 8,
                      "province octaves",
                      "fBm octaves for the province field. More = craggier "
                      "province borders.",
                      "", o);
    reg.declare_float("color.province_lo", 0.42f, 0.0f, 1.0f,
                      "province lo",
                      "Smoothstep low edge on the [0,1] province field. The gap "
                      "to province_hi sets how sharp the dark/bright boundary is.",
                      "", o);
    reg.declare_float("color.province_hi", 0.60f, 0.0f, 1.0f,
                      "province hi",
                      "Smoothstep high edge on the province field.",
                      "", o);
    reg.declare_float("color.province_dark_mul", 0.55f, 0.0f, 2.0f,
                      "province dark mul",
                      "Brightness multiplier in dark (basaltic) provinces. "
                      "0.55 = dark regions sit at ~55% of the base albedo.",
                      "", o);
    reg.declare_float("color.province_bright_mul", 1.18f, 0.0f, 2.0f,
                      "province bright mul",
                      "Brightness multiplier in bright (dust-mantled) provinces.",
                      "", o);
    reg.declare_float("color.province_warm", 0.10f, 0.0f, 0.5f,
                      "province warmth",
                      "Warm/cool shift between provinces: dusty provinces gain "
                      "red and lose blue, basaltic the reverse. 0 = brightness "
                      "variation only, no hue shift.",
                      "", o);
}

void SurfaceColorPass::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    ColorParams P;
    P.low_r  = reg.get_float("color.low_r");
    P.low_g  = reg.get_float("color.low_g");
    P.low_b  = reg.get_float("color.low_b");
    P.mid_r  = reg.get_float("color.mid_r");
    P.mid_g  = reg.get_float("color.mid_g");
    P.mid_b  = reg.get_float("color.mid_b");
    P.high_r = reg.get_float("color.high_r");
    P.high_g = reg.get_float("color.high_g");
    P.high_b = reg.get_float("color.high_b");
    P.low_elev_km  = reg.get_float("color.low_elev_km");
    P.mid_elev_km  = reg.get_float("color.mid_elev_km");
    P.high_elev_km = reg.get_float("color.high_elev_km");
    P.rock_r = reg.get_float("color.rock_r");
    P.rock_g = reg.get_float("color.rock_g");
    P.rock_b = reg.get_float("color.rock_b");
    P.slope_lo_m_per_cell = reg.get_float("color.slope_lo_m_per_cell");
    P.slope_hi_m_per_cell = reg.get_float("color.slope_hi_m_per_cell");
    P.ejecta_r = reg.get_float("color.ejecta_r");
    P.ejecta_g = reg.get_float("color.ejecta_g");
    P.ejecta_b = reg.get_float("color.ejecta_b");
    P.sediment_lo_m = reg.get_float("color.sediment_lo_m");
    P.sediment_hi_m = reg.get_float("color.sediment_hi_m");
    P.sediment_max_blend = reg.get_float("color.sediment_max_blend");
    P.impact_r = reg.get_float("color.impact_r");
    P.impact_g = reg.get_float("color.impact_g");
    P.impact_b = reg.get_float("color.impact_b");
    P.impact_melt_factor = reg.get_float("color.impact_melt_factor");
    P.ice_r = reg.get_float("color.ice_r");
    P.ice_g = reg.get_float("color.ice_g");
    P.ice_b = reg.get_float("color.ice_b");
    P.polar_lat_lo_deg = reg.get_float("color.polar_lat_lo_deg");
    P.polar_lat_hi_deg = reg.get_float("color.polar_lat_hi_deg");
    P.ice_alt_lo_km    = reg.get_float("color.ice_alt_lo_km");
    P.ice_alt_hi_km    = reg.get_float("color.ice_alt_hi_km");
    P.polar_noise_freq    = reg.get_float("color.polar_noise_freq");
    P.polar_noise_amp_deg = reg.get_float("color.polar_noise_amp_deg");
    P.polar_noise_octaves = reg.get_int  ("color.polar_noise_octaves");
    P.alt_noise_freq      = reg.get_float("color.alt_noise_freq");
    P.alt_noise_amp_km    = reg.get_float("color.alt_noise_amp_km");
    P.alt_noise_octaves   = reg.get_int  ("color.alt_noise_octaves");
    P.border_seed         = static_cast<std::uint32_t>(reg.get_int("color.border_seed"));
    P.province_freq       = reg.get_float("color.province_freq");
    P.province_octaves    = reg.get_int  ("color.province_octaves");
    P.province_lo         = reg.get_float("color.province_lo");
    P.province_hi         = reg.get_float("color.province_hi");
    P.province_dark_mul   = reg.get_float("color.province_dark_mul");
    P.province_bright_mul = reg.get_float("color.province_bright_mul");
    P.province_warm       = reg.get_float("color.province_warm");

    const auto& g      = state.grid();
    const auto& nbt    = state.neighbors();
    const int   n      = g.n;
    const int   stride = g.stride();
    const int   halo   = CubedSphereGrid::HALO;
    if (n <= 0) return;

    //Slope reads need cross-face neighbours; same for sediment/bedrock so
    //the per-cell kernel can central-difference cleanly.
    state.bedrock_elevation.halo_exchange(nbt);
    state.sediment_thickness.halo_exchange(nbt);
    state.crust_type.halo_exchange(nbt);

    progress.stage("surface color");
    progress.fraction(0.0f);

    dim3 block(16, 16, 1);
    dim3 grid_dims((n + block.x - 1) / block.x,
                   (n + block.y - 1) / block.y,
                   6);
    surface_color_kernel<<<grid_dims, block>>>(
        state.surface_color.data(),
        state.bedrock_elevation.data(),
        state.sediment_thickness.data(),
        state.crust_type.data(),
        n, stride, halo, P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    PB_LOG_INFO("surface_color",
                "tint computed (low=%.2f mid=%.2f high=%.2f km, ice from "
                "lat>=%.0f or alt>=%.1f km)",
                static_cast<double>(P.low_elev_km),
                static_cast<double>(P.mid_elev_km),
                static_cast<double>(P.high_elev_km),
                static_cast<double>(P.polar_lat_lo_deg),
                static_cast<double>(P.ice_alt_lo_km));
    progress.fraction(1.0f);
}

}
