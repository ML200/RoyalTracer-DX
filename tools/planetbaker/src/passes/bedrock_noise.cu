#include "passes/bedrock_noise.h"
#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace pb {

namespace {

//====================================
//Rocky-body bedrock synthesis. The planet is one contiguous rocky surface
//(no oceans, no continent/ocean dichotomy). Regional elevation varies
//smoothly from lowland plains to highlands, with localised mountain belts
//and high plateaus stacked on top. Suitable as Mars, Mercury, the Moon,
//or any solid airless / single-rocky-surface body. Crust is uniform
//continental rock everywhere.
//
//Layers (all added per pixel):
//  1. Regional fBm                  - smooth low-frequency elevation
//                                     dichotomy (highlands <-> lowlands).
//                                     Domain-warped at low frequency.
//  2. Plateau lift                  - smoothstep at top tail of regional
//                                     fBm: flat-topped highland regions.
//  2b. Mesoscale                    - mid-freq fBm (freq ~6, ~600-1000 km
//                                     scale on Earth) decorrelated from
//                                     the regional dichotomy. Adds sub-
//                                     continent basins / rises and is the
//                                     gating field for dune placement.
//  3. Tectonic plate-boundary mask  - procedural Voronoi plates via
//                                     Worley F2-F1; mountains form
//                                     along the boundaries of jittered
//                                     plate cells, gated by a separate
//                                     convergence noise so only ~half
//                                     the boundaries are mountain-
//                                     building.
//  3b. Terrain-variation roughness  - low-freq fBm decorrelated from
//                                     the regional dichotomy; scales
//                                     every small-scale amplitude
//                                     (hills, fine, mountain, peak,
//                                     lowland chaos) so different
//                                     regions have different surface
//                                     roughness signatures. Also drives
//                                     mountain freq / sharpness variation
//                                     and the dune regime mask.
//  4c. Dune regime                  - in low-terrain_var regions,
//                                     mountains and peaks are suppressed
//                                     and replaced with a sin-based
//                                     directional ripple (dune-field
//                                     texture). Curving crests via a
//                                     multi-octave position warp.
//  4. Mountain ridges               - Musgrave ridged multifractal at the
//                                     warped position, scaled by belt mask
//                                     and modulated by regional elevation
//                                     so peaks cluster in highlands.
//  4b. Singular peaks               - Worley-distance compact peaks at
//                                     jittered cell centers, stacked on
//                                     ridges. Adds Matterhorn / K2-style
//                                     isolated summits that pure ridged
//                                     mfbm cannot produce.
//  5. Rolling hills                 - mid-freq fBm everywhere.
//  6. Lowland chaos                 - extra ridged noise in low regions
//                                     for the cratered/chaotic-terrain
//                                     look (Martian Vastitas / Noachian).
//  7. Fine roughness                - high-freq fBm, universal.
//
//Dendritic valleys (the branching-tree drainage networks real mountains
//have) are NOT produced procedurally here: ridged Perlin only generates
//continuous ridge lines, not flow-network valleys. They come from the
//downstream HydraulicErosionPass when its iterations are nonzero.
//====================================

struct BedrockNoiseParams {
    //Planet radius. All frequency knobs below are tuned for Earth scale
    //(6371 km); at other radii, run() multiplies every freq param by
    //(planet_radius_km / 6371) before the kernel launch, so the same freq
    //value produces the same km-spacing of features regardless of body
    //size. Earth=6371, Mars=3389, Moon=1737, Mercury=2440. Amplitudes
    //(*_km) and warp amounts are NOT scaled - those are absolute or
    //unit-sphere angular and stay the user's responsibility per body.
    float planet_radius_km      = 6371.0f;

    //Regional elevation extremes. Mean rocky-body radius is the reference;
    //elevation is the offset from that. Mars's dichotomy is roughly -2 to
    //+3 km between northern lowlands and southern highlands, so the
    //defaults are tuned to that range.
    float lowland_base_km       = -1.8f;
    float highland_base_km      = 2.4f;

    //Regional fBm: large-scale highland/lowland dichotomy.
    //v4: base freq 2.5 (was 1.0) gives ~15 cells around the equator at the
    //base octave so the macro shape has real structure. octaves 7 (was 5)
    //adds two more cascade levels. warp is multi-octave fBm with a low
    //base freq and 3 octaves so coastlines crinkle at multiple scales.
    float regional_warp_freq    = 0.9f;
    float regional_warp_amount  = 0.30f;
    int   regional_warp_octaves = 3;
    float regional_freq         = 2.5f;
    int   regional_octaves      = 7;

    //Plateau lift on the top tail of regional fBm (Tharsis-style raised
    //flat-topped highland patches that sit above the average highland).
    float plateau_threshold     = 0.68f;
    float plateau_softness      = 0.05f;
    float plateau_lift_km       = 2.0f;

    //Mesoscale elevation (v11). Mid-frequency fBm bridging the gap between
    //the regional dichotomy (freq ~1.5) and the hill scale (freq 45). Adds
    //sub-continent basins and rises at ~600-1000 km scale on Earth so
    //plateaus and lowlands have internal variation instead of reading as
    //uniformly tall/short slabs. Also drives the dune mask: dune zones
    //form where the mesoscale field dips negative (local depressions),
    //so dunes never appear on plateaus and never tie to the macro
    //regional dichotomy.
    float mesoscale_freq        = 6.0f;
    int   mesoscale_octaves     = 5;
    float mesoscale_amp_km      = 1.5f;

    //Procedural plate-boundary mask (v8). Replaces v6's belt fBm with a
    //Worley F2-F1 'distance to nearest plate edge' field, gated by a
    //convergence noise so only ~half the boundaries become mountain
    //ranges (the collision / subduction ones). Plates are jittered
    //Voronoi cells in 3D space; the unit sphere passes through ~10
    //plates at plate_freq=0.8 (Earth-like, scaled by planet radius).
    //plate_warp_* curves the boundaries away from polygonal Voronoi
    //edges so they look like real subduction arcs, not flat seams.
    //boundary_sharpness controls the exp falloff width: 12 -> mountain
    //belt is ~250 km wide on Earth.
    float plate_freq            = 0.8f;
    float plate_warp_freq       = 0.4f;
    float plate_warp_amount     = 0.18f;
    int   plate_warp_octaves    = 2;
    float boundary_sharpness    = 12.0f;
    //Which boundaries are mountain-building (convergent) vs flat
    //(transform / divergent). Low-freq fBm so a single plate boundary
    //is coherently mountain-building or coherently flat along its arc.
    float convergence_freq      = 1.5f;
    int   convergence_octaves   = 3;
    float convergence_threshold = 0.55f;
    float convergence_softness  = 0.08f;

    //Mountain ridges (Musgrave ridged multifractal).
    //
    //CRITICAL: mountain_freq controls ridge spacing. The visible-ridge
    //spacing is ~half the base wavelength (zero-crossings of the noise).
    //At mountain_freq = 120 -> base wavelength 0.052 rad ~ 3 deg ~ 330 km,
    //so ridges are ~165 km apart. At preview N=1024 (~20 km/px) that's
    //~8 px between ridges - clearly visible as line features.
    //Going lower (e.g. 30) gives blob-sized features that look like
    //'smeared bubbles' at preview resolution.
    //
    //v4: warp is multi-octave fBm (mountain_warp_octaves) so even with
    //a low warp base freq the highest warp octave reaches the ridge
    //frequency and breaks the closed contours of ridged Perlin into
    //chain-like fragments. Default warp_octaves=4 with base_freq=30
    //gives warp components at 30, 60, 120, 240 - the freq=120 octave
    //matches the ridge freq and dissolves the bubble pattern.
    float mountain_warp_freq    = 30.0f;
    float mountain_warp_amount  = 0.04f;
    int   mountain_warp_octaves = 4;
    float mountain_freq         = 120.0f;
    int   mountain_octaves      = 6;
    float mountain_amp_km       = 4.5f;
    float mountain_exponent     = 2.6f;
    float mountain_lacunarity   = 2.0f;
    float mountain_gain         = 0.55f;
    //Boost mountains in highlands, attenuate in lowlands.
    float mountain_regional_lo  = 0.30f;
    float mountain_regional_hi  = 1.00f;

    //Singular peaks (layer 4b). Worley-distance peaks at jittered cell
    //centers, stacked on top of the continuous ridge network. Models the
    //isolated tall summits real mountains have (the Matterhorn, the
    //Eiger, Olympus Mons, K2) that ridged multifractal alone cannot
    //produce because it only makes continuous ridge lines.
    //
    //peak_freq at 80 with planet_radius_km=6371 gives plant-point spacing
    //~80 km, so peaks are roughly 80-150 km apart in clusters.
    //peak_radius is the falloff radius in cell-size units; 0.5 means the
    //peak's influence dies within half a cell of its center. Peaks are
    //gated by belt_mask * reg_boost, same as ridges, so they only appear
    //in mountain country.
    float peak_freq             = 80.0f;
    float peak_radius           = 0.55f;
    float peak_sharpness        = 4.0f;
    float peak_amp_km           = 3.0f;

    //Rolling hills (mid-freq fBm everywhere). Wavelength ~5 deg (560 km,
    //~28 px at N=1024) so hills are visible as broad undulation.
    //v6: hill_amp_km 0.40 (was 0.55) so plains read as plains, not as
    //busy textured noise that competes with the mountain belt signal.
    float hill_freq             = 45.0f;
    int   hill_octaves          = 4;
    float hill_amp_km           = 0.40f;

    //Lowland chaos: extra ridged noise in low regions (Martian chaos
    //terrain look). Gated by inverse regional value so it's strongest
    //in basins.
    float lowland_rough_freq    = 70.0f;
    int   lowland_rough_octaves = 4;
    float lowland_rough_amp_km  = 0.45f;

    //Fine roughness everywhere. Wavelength ~1.2 deg (130 km, ~7 px at
    //N=1024) - close to the Nyquist limit, gives the surface a busy
    //textured appearance without being entirely aliased.
    float fine_freq             = 280.0f;
    int   fine_octaves          = 3;
    float fine_amp_km           = 0.15f;

    //Terrain-variation modulator (v9). A low-freq fBm field sampled per
    //pixel, used to scale every small-scale amplitude (hills, fine, lowland
    //chaos, mountain belts, peaks) by `mix(1-amp, 1+amp, var_n)`. Without
    //this, the same hill amplitude / mountain amplitude applies to the
    //ENTIRE planet so every region has the same roughness signature -
    //a smooth plain looks identical to a heavily-textured plain. With this
    //layer, ~half the planet reads as smoother-than-average, the other half
    //as rougher-than-average; the boundary is a continuous curve at the
    //variation frequency. Decorrelated from the regional dichotomy so a
    //smooth zone can sit on either highland or lowland.
    float terrain_var_freq      = 1.8f;
    int   terrain_var_octaves   = 4;
    float terrain_var_amp       = 0.7f;

    //Mountain shape variation (v10). The terrain_var_n field also modulates
    //mountain ridge FREQUENCY and SHARPNESS, so different regions have
    //different ridge spacings and ridge profiles instead of the v9 uniform
    //look. High-roughness regions get fine sharp ridges (Alps); low-
    //roughness regions get coarser softer ridges (older eroded ranges).
    float mountain_freq_var     = 0.5f;  //freq scales by mix(1-var, 1+var, var_n)
    float mountain_exp_var      = 0.3f;  //exp likewise

    //Dune regime (v10). Where terrain_var_n is low (smooth regions),
    //mountains and peaks are suppressed and replaced with a sin-based
    //directional ripple that reads as a dune field at planet view. The
    //ripple direction varies across the planet via a multi-octave warp
    //on the sample position, so dune crests curve realistically over long
    //distances (think Olympia Undae on Mars). Hard-suppresses mountain_
    //contrib and peak_contrib inside the dune mask so dune zones don't
    //also have mountains poking through. Set dune_amp_km = 0 or
    //dune_threshold = 0 to disable.
    float dune_threshold        = 0.25f;  //terrain_var_n below this -> dune zone
    float dune_softness         = 0.08f;
    float dune_freq             = 90.0f;  //~4 deg wavelength -> ~440 km ripple
    float dune_warp_freq        = 3.0f;
    float dune_warp_amount      = 0.07f;
    float dune_amp_km           = 0.4f;

    //Crust age variation (single density rocky body, but age varies).
    float age_freq              = 2.8f;
    float age_mid_myr           = 2500.0f;
    float age_var_myr           = 1500.0f;

    std::uint32_t seed          = 0xC0FFEE17u;
};

//====================================
//Device noise primitives.
//====================================

__device__ __forceinline__ std::uint32_t d_pcg(std::uint32_t x) {
    std::uint32_t state = x * 747796405u + 2891336453u;
    std::uint32_t word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ __forceinline__ std::uint32_t d_hash3(int x, int y, int z, std::uint32_t seed) {
    std::uint32_t h = seed;
    h = d_pcg(h ^ static_cast<std::uint32_t>(x * 0x27d4eb2du));
    h = d_pcg(h ^ static_cast<std::uint32_t>(y * 0x165667b1u));
    h = d_pcg(h ^ static_cast<std::uint32_t>(z * 0x9e3779b9u));
    return h;
}

__device__ __forceinline__ float d_h2u(std::uint32_t h) {
    return static_cast<float>(h) * (1.0f / 4294967296.0f);
}

//Quintic smoothstep with zero first and second derivatives at endpoints.
//Removes the soft lattice-aligned banding you get with cubic Hermite.
__device__ __forceinline__ float d_quintic(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

//Gradient lookup for 3D Perlin: 16 gradient directions encoded via hash
//bits. Returns the dot product of the gradient with the offset (x,y,z).
//Standard Ken Perlin 2002 reference set.
__device__ __forceinline__ float d_grad3(std::uint32_t h,
                                          float x, float y, float z) {
    std::uint32_t k = h & 15u;
    float u = (k < 8u) ? x : y;
    float v = (k < 4u) ? y : ((k == 12u || k == 14u) ? x : z);
    float gu = ((k & 1u) == 0u) ? u : -u;
    float gv = ((k & 2u) == 0u) ? v : -v;
    return gu + gv;
}

//3D Perlin gradient noise. Returns approximately [-1, 1]. Bubble-free:
//the underlying field has directional structure because each cube corner
//projects an angular bias, not a scalar bump.
__device__ float d_perlin(float x, float y, float z, std::uint32_t seed) {
    int   x0 = static_cast<int>(floorf(x));
    int   y0 = static_cast<int>(floorf(y));
    int   z0 = static_cast<int>(floorf(z));
    float fx = x - static_cast<float>(x0);
    float fy = y - static_cast<float>(y0);
    float fz = z - static_cast<float>(z0);
    float sx = d_quintic(fx);
    float sy = d_quintic(fy);
    float sz = d_quintic(fz);

    float n000 = d_grad3(d_hash3(x0,     y0,     z0,     seed),
                         fx,        fy,        fz);
    float n100 = d_grad3(d_hash3(x0 + 1, y0,     z0,     seed),
                         fx - 1.0f, fy,        fz);
    float n010 = d_grad3(d_hash3(x0,     y0 + 1, z0,     seed),
                         fx,        fy - 1.0f, fz);
    float n110 = d_grad3(d_hash3(x0 + 1, y0 + 1, z0,     seed),
                         fx - 1.0f, fy - 1.0f, fz);
    float n001 = d_grad3(d_hash3(x0,     y0,     z0 + 1, seed),
                         fx,        fy,        fz - 1.0f);
    float n101 = d_grad3(d_hash3(x0 + 1, y0,     z0 + 1, seed),
                         fx - 1.0f, fy,        fz - 1.0f);
    float n011 = d_grad3(d_hash3(x0,     y0 + 1, z0 + 1, seed),
                         fx,        fy - 1.0f, fz - 1.0f);
    float n111 = d_grad3(d_hash3(x0 + 1, y0 + 1, z0 + 1, seed),
                         fx - 1.0f, fy - 1.0f, fz - 1.0f);

    float a = n000 + sx * (n100 - n000);
    float b = n010 + sx * (n110 - n010);
    float c = n001 + sx * (n101 - n001);
    float d = n011 + sx * (n111 - n011);
    float e = a   + sy * (b   - a  );
    float f = c   + sy * (d   - c  );
    return e + sz * (f - e);
}

//fBm on Perlin gradient noise. Returns approximately [0, 1].
__device__ float d_fbm(float x, float y, float z,
                       int octaves, float freq,
                       float lacunarity, float gain,
                       std::uint32_t seed) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f, f = freq;
    std::uint32_t s = seed;
    for (int o = 0; o < octaves; ++o) {
        sum  += amp * d_perlin(x * f, y * f, z * f, s);
        norm += amp;
        amp  *= gain;
        f    *= lacunarity;
        s     = s * 1664525u + 1013904223u;
    }
    //Map signed sum -> [0, 1]. The signed range of 3D Perlin is roughly
    //[-0.866, +0.866] per octave; the sum/norm of n octaves stays in
    //approximately the same range so the *0.5+0.5 mapping is robust.
    float r = (sum / norm) * 0.5f + 0.5f;
    if (r < 0.0f) r = 0.0f;
    if (r > 1.0f) r = 1.0f;
    return r;
}

//Musgrave ridged multifractal on Perlin gradient noise. Each octave's
//contribution is multiplied by the PREVIOUS octave's value so detail
//concentrates in already-high regions - the "mountains get rougher,
//valleys stay smooth" behaviour. Because the underlying primitive is
//directional, the ridges form long sinuous chains rather than ridged
//bumps over a bubble pattern.
__device__ float d_ridged_mfbm(float x, float y, float z,
                               int octaves, float freq,
                               float lacunarity, float gain,
                               std::uint32_t seed) {
    float sum  = 0.0f;
    float amp  = 1.0f;
    float norm = 0.0f;
    float f    = freq;
    float prev = 1.0f;
    std::uint32_t s = seed;
    for (int o = 0; o < octaves; ++o) {
        float v = d_perlin(x * f, y * f, z * f, s);    //~[-1, 1]
        v = 1.0f - fabsf(v);                            //ridge: peak at v=0
        v = v * v;                                       //sharpen
        sum  += amp * v * prev;
        norm += amp;
        prev  = v;
        amp  *= gain;
        f    *= lacunarity;
        s     = s * 1664525u + 1013904223u;
    }
    return sum / norm;
}

//Worley F1 distance. Plants one jittered point per integer lattice cell
//and returns the (unsquared) distance from the query position to the
//nearest planted point. Scan window is the 3x3x3 cube of cells around
//the query cell. Output is in cell-size units (typical range ~0 to ~1.4
//for jittered grids; exceeds 1 only at cell corners). Multiply input by
//freq before calling for a desired plant-point density.
//
//Use case here: a Worley-distance peak layer for SINGULAR mountain peaks.
//Ridged Perlin produces a network of continuous ridge LINES; it cannot
//produce isolated peaks (the Matterhorn, Olympus Mons). Worley distance
//inverted into a Gaussian-style falloff places compact peaks at jittered
//cell centers, exactly what we want for that look.
__device__ float d_worley_f1(float x, float y, float z, std::uint32_t seed) {
    int   x0 = static_cast<int>(floorf(x));
    int   y0 = static_cast<int>(floorf(y));
    int   z0 = static_cast<int>(floorf(z));
    float min_d2 = 1.0e9f;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int           cx = x0 + dx;
        int           cy = y0 + dy;
        int           cz = z0 + dz;
        std::uint32_t h  = d_hash3(cx, cy, cz, seed);
        float         jx = d_h2u(h);
        float         jy = d_h2u(d_pcg(h));
        float         jz = d_h2u(d_pcg(d_pcg(h)));
        float         px = static_cast<float>(cx) + jx;
        float         py = static_cast<float>(cy) + jy;
        float         pz = static_cast<float>(cz) + jz;
        float         ex = x - px;
        float         ey = y - py;
        float         ez = z - pz;
        float         d2 = ex * ex + ey * ey + ez * ez;
        if (d2 < min_d2) min_d2 = d2;
    }
    return sqrtf(min_d2);
}

//Worley F2 - F1: distance from the query point to the nearest VORONOI
//EDGE in cell-size units. Same 3x3x3 scan as F1 but tracks both the
//nearest and the second-nearest plant points; the locus where d1 == d2
//(i.e. equidistant from two plants) is the cell boundary, and F2-F1
//grows from 0 there toward each cell's interior.
//
//Use case here: 'procedural plate tectonics'. Each jittered plant point
//defines a plate; F2-F1 lights up along plate-plate boundaries, giving
//a coherent line-pattern instead of the random blob mask v6 produced
//from belt-fBm. Per-pixel and mesh-free, so it sidesteps the resampling
//and hex-Voronoi artifacts of the abandoned Goldberg-mesh tectonic sim.
__device__ float d_worley_f2_minus_f1(float x, float y, float z, std::uint32_t seed) {
    int   x0 = static_cast<int>(floorf(x));
    int   y0 = static_cast<int>(floorf(y));
    int   z0 = static_cast<int>(floorf(z));
    float min1_d2 = 1.0e9f;
    float min2_d2 = 1.0e9f;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int           cx = x0 + dx;
        int           cy = y0 + dy;
        int           cz = z0 + dz;
        std::uint32_t h  = d_hash3(cx, cy, cz, seed);
        float         jx = d_h2u(h);
        float         jy = d_h2u(d_pcg(h));
        float         jz = d_h2u(d_pcg(d_pcg(h)));
        float         px = static_cast<float>(cx) + jx;
        float         py = static_cast<float>(cy) + jy;
        float         pz = static_cast<float>(cz) + jz;
        float         ex = x - px;
        float         ey = y - py;
        float         ez = z - pz;
        float         d2 = ex * ex + ey * ey + ez * ez;
        if (d2 < min1_d2) {
            min2_d2 = min1_d2;
            min1_d2 = d2;
        } else if (d2 < min2_d2) {
            min2_d2 = d2;
        }
    }
    return sqrtf(min2_d2) - sqrtf(min1_d2);
}

//Multi-octave fBm domain warp. Each component (dx, dy, dz) is its own fBm
//of signed Perlin, so the displacement field varies at many scales at once:
//low octaves bend continent-scale features, high octaves break individual
//ridges. Total displacement magnitude is bounded by `amount` because the
//per-component sum is normalized by the amplitude total.
//
//This replaces v3's single-octave d_warp. With one octave the warp could
//only do one thing - either smear large patches uniformly (low freq) or
//jitter at the ridge scale (high freq) - never both. Ridged-Perlin's peaks
//trace zero-crossings of the underlying noise, which are closed contours;
//breaking those requires distortion AT the ridge scale, but bending
//continents requires displacement at a much coarser scale. Multi-octave
//warp gives both for the same total displacement budget.
__device__ Vec3f d_fbm_warp(Vec3f p, float base_freq, float amount,
                            int octaves, float lacunarity, float gain,
                            std::uint32_t seed) {
    if (amount <= 0.0f || octaves <= 0) return p;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    float amp  = 1.0f;
    float norm = 0.0f;
    float f    = base_freq;
    std::uint32_t s = seed;
    for (int o = 0; o < octaves; ++o) {
        float x = p.x * f, y = p.y * f, z = p.z * f;
        dx  += amp * d_perlin(x, y, z, s);
        dy  += amp * d_perlin(x, y, z, s + 17u);
        dz  += amp * d_perlin(x, y, z, s + 113u);
        norm += amp;
        amp  *= gain;
        f    *= lacunarity;
        s     = s * 1664525u + 1013904223u;
    }
    float inv = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    Vec3f q   = {p.x + (dx * inv) * amount,
                 p.y + (dy * inv) * amount,
                 p.z + (dz * inv) * amount};
    return vec3f_normalize(q);
}

__device__ __forceinline__ float d_smoothstep(float a, float b, float t) {
    float w = (b - a);
    if (w <= 0.0f) return (t >= b) ? 1.0f : 0.0f;
    float k = (t - a) / w;
    if (k < 0.0f) k = 0.0f;
    if (k > 1.0f) k = 1.0f;
    return k * k * (3.0f - 2.0f * k);
}

__device__ __forceinline__ float d_mix(float a, float b, float t) {
    return a + (b - a) * t;
}

//====================================
//Rocky-body synthesis kernel.
//====================================

__global__ void synth_kernel(
    float*               __restrict__ out_bedrock,
    float*               __restrict__ out_thickness,
    float*               __restrict__ out_density,
    float*               __restrict__ out_age,
    std::uint8_t*        __restrict__ out_type,
    int n, int stride, int halo,
    BedrockNoiseParams P)
{
    int i    = blockIdx.x * blockDim.x + threadIdx.x;
    int j    = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;
    if (i >= n || j >= n) return;

    float u = ((static_cast<float>(i) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    float v = ((static_cast<float>(j) + 0.5f) / static_cast<float>(n)) * 2.0f - 1.0f;
    Vec3f p = face_uv_to_sphere(face, u, v);

    //========================================
    //1. Regional fBm: large-scale dichotomy controller. Continuous in
    //   [0, 1]; 0 = deep lowland, 1 = high highland. No threshold; the
    //   value modulates everything else.
    //========================================
    Vec3f p_reg = d_fbm_warp(p, P.regional_warp_freq, P.regional_warp_amount,
                             P.regional_warp_octaves, 2.0f, 0.5f, P.seed + 11u);
    float regional_n = d_fbm(p_reg.x, p_reg.y, p_reg.z,
                             P.regional_octaves, P.regional_freq,
                             2.0f, 0.5f, P.seed + 23u);
    //v7: sigmoid contrast on regional fBm, sharpened. fBm output is roughly
    //Gaussian around 0.5 with std ~0.15 (octaves dependent), so 95% of
    //cells fall in [0.20, 0.80]. v4's smoothstep(0.25, 0.75, n) mapped
    //that band linearly across [0, 1] - i.e. most of the planet ended up
    //in the soft middle of the curve, looking mid-tone everywhere with
    //no clear highland/lowland regime. v7 tightens to (0.40, 0.60) so
    //cells slightly above 0.5 saturate to highland and cells slightly
    //below saturate to lowland - producing a real bimodal dichotomy.
    //Boundary cells still smoothstep, no hard edges.
    float regional_curve = d_smoothstep(0.40f, 0.60f, regional_n);
    float base_elev = d_mix(P.lowland_base_km, P.highland_base_km, regional_curve);

    //========================================
    //2. Plateau lift on the top tail of regional fBm.
    //========================================
    float plateau_mask = d_smoothstep(P.plateau_threshold - P.plateau_softness,
                                      P.plateau_threshold + P.plateau_softness,
                                      regional_n);
    float plateau_contrib = plateau_mask * P.plateau_lift_km;

    //========================================
    //2b. Mesoscale elevation (v11). Mid-freq fBm decorrelated from the
    //    regional dichotomy. Provides sub-continent basins and rises so
    //    plateaus and lowlands aren't internally uniform.
    //========================================
    float mesoscale_n = d_fbm(p.x, p.y, p.z,
                              P.mesoscale_octaves, P.mesoscale_freq,
                              2.0f, 0.5f, P.seed + 271u);
    float mesoscale_contrib = (mesoscale_n - 0.5f) * 2.0f * P.mesoscale_amp_km;

    //========================================
    //3. Procedural plate-tectonic belt mask. Two stages:
    //   (a) plate boundaries via Worley F2-F1 on jittered plant points.
    //       Each plant defines a "plate" cell; F2-F1 is the distance to
    //       the nearest plate-plate edge, exp'd to a compact strip
    //       along that edge. Plant positions are pre-warped by an fBm
    //       so the boundary arcs CURVE (Andes/Himalayas-shaped) rather
    //       than form polygonal Voronoi seams.
    //   (b) convergence gate: a low-freq fBm decides which boundary
    //       arcs are convergent (mountain-building) vs transform /
    //       divergent (flat). Without this, every plate edge would
    //       have mountains, which doesn't match real planets.
    //   This replaces v6's belt fBm threshold which placed mountain
    //   belts at noise PEAKS (random blobs). Plate boundaries are
    //   LINES so the new mask gives coherent arc-shaped ranges.
    //========================================
    Vec3f p_plate = d_fbm_warp(p, P.plate_warp_freq, P.plate_warp_amount,
                               P.plate_warp_octaves, 2.0f, 0.5f, P.seed + 137u);
    float plate_edge_dist = d_worley_f2_minus_f1(p_plate.x * P.plate_freq,
                                                 p_plate.y * P.plate_freq,
                                                 p_plate.z * P.plate_freq,
                                                 P.seed + 149u);
    float boundary_mask = expf(-plate_edge_dist * P.boundary_sharpness);
    float convergence_n = d_fbm(p_plate.x, p_plate.y, p_plate.z,
                                P.convergence_octaves, P.convergence_freq,
                                2.0f, 0.5f, P.seed + 163u);
    float convergence_mask = d_smoothstep(P.convergence_threshold - P.convergence_softness,
                                          P.convergence_threshold + P.convergence_softness,
                                          convergence_n);
    float belt_mask = boundary_mask * convergence_mask;

    //========================================
    //3b. Terrain-variation roughness + shape + dune regime. A single
    //    low-freq fBm field drives three correlated axes of variation:
    //      - amplitude scaling (roughness factor on all detail amps)
    //      - mountain shape (freq and sharpness vary per region)
    //      - dune regime (low-var regions become directional ripple
    //                     plains instead of having mountains/peaks)
    //========================================
    float terrain_var_n = d_fbm(p.x, p.y, p.z,
                                P.terrain_var_octaves, P.terrain_var_freq,
                                2.0f, 0.5f, P.seed + 191u);
    float roughness = d_mix(1.0f - P.terrain_var_amp,
                            1.0f + P.terrain_var_amp,
                            terrain_var_n);
    if (roughness < 0.0f) roughness = 0.0f;

    //Mountain shape variation. Higher terrain_var -> finer + sharper ridges.
    float mtn_freq_scale = d_mix(1.0f - P.mountain_freq_var,
                                 1.0f + P.mountain_freq_var,
                                 terrain_var_n);
    float mtn_exp_scale  = d_mix(1.0f - P.mountain_exp_var,
                                 1.0f + P.mountain_exp_var,
                                 terrain_var_n);
    float mtn_freq_local = P.mountain_freq     * mtn_freq_scale;
    float mtn_exp_local  = P.mountain_exponent * mtn_exp_scale;

    //Dune regime mask (v11). Driven by MESOSCALE_N, not terrain_var_n. The
    //mesoscale field is decorrelated from both the coarse regional fBm and
    //the roughness mask, so dune patches are independent of macro elevation
    //AND smaller (mesoscale_freq=6 vs terrain_var_freq=1.8). Inverted
    //smoothstep: 1 where the mesoscale field is low (= a local depression),
    //0 where the mesoscale is high (= a local rise / plateau-like).
    //Geophysical reading: dunes accumulate in low spots, never on plateaus.
    float dune_mask = 1.0f - d_smoothstep(P.dune_threshold - P.dune_softness,
                                          P.dune_threshold + P.dune_softness,
                                          mesoscale_n);

    //========================================
    //4. Mountain ridges. Ridged multifractal at high frequency on a
    //   modestly warped position. Amplitude modulated by belt_mask AND
    //   regional elevation, so peaks cluster in highland belts and are
    //   muted in lowland belts (where you only get foothills). Also
    //   scaled by `roughness` so some belts are taller than others.
    //========================================
    Vec3f p_mnt = d_fbm_warp(p, P.mountain_warp_freq, P.mountain_warp_amount,
                             P.mountain_warp_octaves, 2.0f, 0.5f, P.seed + 53u);
    float mountain_n = d_ridged_mfbm(p_mnt.x, p_mnt.y, p_mnt.z,
                                     P.mountain_octaves, mtn_freq_local,
                                     P.mountain_lacunarity, P.mountain_gain,
                                     P.seed + 67u);
    float mountain_v = powf(fmaxf(mountain_n, 0.0f), mtn_exp_local);
    float reg_boost  = d_mix(P.mountain_regional_lo, P.mountain_regional_hi, regional_curve);
    float mountain_contrib = mountain_v * P.mountain_amp_km * belt_mask * reg_boost
                           * roughness * (1.0f - dune_mask);

    //========================================
    //4b. Singular peaks. Worley F1 distance to jittered cell centers,
    //    inverted into a sharp Gaussian-style falloff. Stacks isolated
    //    summit-style peaks on top of the continuous ridge network.
    //    Sampled at the mountain-warped position so peaks share the
    //    same ridge curvature; gated by belt_mask * reg_boost so they
    //    only appear in mountain belts.
    //========================================
    float peak_d = d_worley_f1(p_mnt.x * P.peak_freq,
                               p_mnt.y * P.peak_freq,
                               p_mnt.z * P.peak_freq,
                               P.seed + 83u);
    float peak_t = peak_d / fmaxf(P.peak_radius, 1.0e-4f);
    if (peak_t > 1.0f) peak_t = 1.0f;
    float peak_v = powf(1.0f - peak_t, P.peak_sharpness);
    float peak_contrib = peak_v * P.peak_amp_km * belt_mask * reg_boost
                       * roughness * (1.0f - dune_mask);

    //========================================
    //4c. Dune regime. A directional sin ripple gated by dune_mask, replaces
    //    peaks/mountains in smooth (low-terrain_var) regions. The position
    //    is multi-octave fbm-warped so dune crests curve naturally instead
    //    of running straight - same trick as mountain warp.
    //========================================
    Vec3f p_dune = d_fbm_warp(p, P.dune_warp_freq, P.dune_warp_amount,
                              2, 2.0f, 0.5f, P.seed + 251u);
    float dune_a = sinf(p_dune.x * P.dune_freq) * cosf(p_dune.y * P.dune_freq * 1.05f);
    float dune_v = 0.5f + 0.5f * dune_a;
    //Sharpen ridges into dune-crest shape.
    dune_v = dune_v * dune_v;
    float dune_contrib = dune_v * P.dune_amp_km * dune_mask;

    //========================================
    //5. Rolling hills everywhere. Gives the surface texture between belts.
    //========================================
    float hill_n = d_fbm(p.x, p.y, p.z,
                         P.hill_octaves, P.hill_freq,
                         2.0f, 0.5f, P.seed + 71u);
    float hill_contrib = (hill_n - 0.5f) * 2.0f * P.hill_amp_km * roughness;

    //========================================
    //6. Lowland chaos: extra ridged roughness in basins. Models the
    //   chaotic-terrain look of low Mars regions.
    //========================================
    float lowland_n = d_ridged_mfbm(p.x, p.y, p.z,
                                    P.lowland_rough_octaves, P.lowland_rough_freq,
                                    2.05f, 0.55f, P.seed + 79u);
    float lowland_mask    = 1.0f - regional_curve;
    float lowland_contrib = (lowland_n - 0.40f) * P.lowland_rough_amp_km * lowland_mask * roughness;

    //========================================
    //7. Fine universal roughness.
    //========================================
    float fine_n = d_fbm(p.x, p.y, p.z,
                         P.fine_octaves, P.fine_freq,
                         2.0f, 0.5f, P.seed + 97u);
    float fine_contrib = (fine_n - 0.5f) * 2.0f * P.fine_amp_km * roughness;

    //========================================
    //Compose.
    //========================================
    float elev = base_elev + plateau_contrib + mesoscale_contrib
                           + mountain_contrib + peak_contrib + dune_contrib
                           + hill_contrib + lowland_contrib + fine_contrib;

    //========================================
    //Crust fields. Single rocky-body composition: continental-rock
    //density throughout. Thickness via inverse Airy on elevation.
    //========================================
    constexpr float kRockDensity   = 2700.0f;
    constexpr float kMantleDensity = 3300.0f;
    constexpr float kRefOffset     = 4.0f;
    constexpr float kBuoyancy      = (kMantleDensity - kRockDensity) / kMantleDensity;

    float thickness = (elev + kRefOffset) / kBuoyancy;
    if (thickness < 4.0f)  thickness = 4.0f;
    if (thickness > 80.0f) thickness = 80.0f;

    float age_n = d_fbm(p.x, p.y, p.z, 4, P.age_freq, 2.0f, 0.5f, P.seed + 109u);
    float age   = P.age_mid_myr + (age_n - 0.5f) * 2.0f * P.age_var_myr;
    if (age < 0.0f) age = 0.0f;

    int idx = face * stride * stride + (j + halo) * stride + (i + halo);
    out_bedrock  [idx] = elev;
    out_thickness[idx] = thickness;
    out_density  [idx] = kRockDensity;
    out_age      [idx] = age;
    out_type     [idx] = 0;          //all rocky/continental crust
}

void run_synth(Field<float>&         bedrock,
               Field<float>&         thickness,
               Field<float>&         density,
               Field<float>&         age,
               Field<std::uint8_t>&  type,
               const BedrockNoiseParams& P) {
    const auto& g      = bedrock.grid();
    const int   n      = g.n;
    const int   stride = g.stride();
    const int   halo   = CubedSphereGrid::HALO;
    if (n <= 0) return;

    dim3 block(16, 16, 1);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y,
              6);
    synth_kernel<<<grid, block>>>(
        bedrock.data(),
        thickness.data(),
        density.data(),
        age.data(),
        type.data(),
        n, stride, halo,
        P);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}

//====================================
//Pass interface
//====================================

void BedrockNoisePass::declare_params(ParamRegistry& reg) const {
    const char* o = name();

    //Planet scale
    reg.declare_float("bedrock.planet_radius_km",  6371.0f, 100.0f, 100000.0f,
                      "planet radius",
                      "Target body radius. All bedrock freq knobs are tuned for Earth "
                      "(6371 km); at other radii every freq is scaled by radius/6371 so "
                      "the same freq value gives the same km-spacing of features. Useful "
                      "presets: Earth 6371, Mars 3389, Moon 1737, Mercury 2440, Ceres 470. "
                      "Amplitudes (*_km) and warp amounts are NOT auto-scaled - tune those "
                      "per body if needed.", "km", o);

    //Regional baselines
    reg.declare_float("bedrock.lowland_base_km",      -1.8f, -8.0f, 2.0f,
                      "lowland base",
                      "Mean elevation of regional lowland basins (Mars: ~-2 km)",
                      "km", o);
    reg.declare_float("bedrock.highland_base_km",      2.4f, -2.0f, 8.0f,
                      "highland base",
                      "Mean elevation of regional highlands before plateau/mountain detail "
                      "(Mars: ~+2.5 km)", "km", o);

    //Regional fBm
    reg.declare_float("bedrock.regional_warp_freq",    0.9f,  0.1f, 4.0f,
                      "regional warp base freq",
                      "Base octave freq of the multi-octave warp on the dichotomy",
                      "", o);
    reg.declare_float("bedrock.regional_warp_amount",  0.30f, 0.0f, 0.7f,
                      "regional warp amount",
                      "Total warp displacement magnitude. With multi-octave warp this is "
                      "the budget shared across all octaves, not per-octave.", "", o);
    reg.declare_int  ("bedrock.regional_warp_octaves", 3,     1,    6,
                      "regional warp octaves",
                      "Octaves of the displacement fBm. 1 = single-scale warp (the old v3 "
                      "behaviour). 3 spreads the warp across continent, sub-continent and "
                      "regional scales so coastlines and the dichotomy boundary crinkle.",
                      "", o);
    reg.declare_float("bedrock.regional_freq",         2.5f,  0.2f, 8.0f,
                      "regional freq",
                      "Base octave freq of the regional dichotomy fBm. ~2.5 gives ~15 "
                      "Perlin cells around the equator at the base octave - enough for "
                      "real planet-wide structure. <1 = single Mars-style hemisphere "
                      "dichotomy; >5 = many small regional zones.",
                      "", o);
    reg.declare_int  ("bedrock.regional_octaves",      7,     1,   12,
                      "regional octaves",
                      "fBm octaves for regional dichotomy. 7 with base 2.5 covers scales "
                      "from planet to ~250 km.", "", o);

    //Plateau
    reg.declare_float("bedrock.plateau_threshold",     0.68f, 0.40f, 0.95f,
                      "plateau threshold",
                      "Above this regional value, plateau lift kicks in", "", o);
    reg.declare_float("bedrock.plateau_softness",      0.05f, 0.005f, 0.20f,
                      "plateau softness", "Smoothstep width on plateau threshold", "", o);
    reg.declare_float("bedrock.plateau_lift_km",       2.0f,  0.0f,  5.0f,
                      "plateau lift", "Elevation bump on plateau patches", "km", o);

    //Mesoscale layer (v11)
    reg.declare_float("bedrock.mesoscale_freq",        6.0f,  0.5f, 30.0f,
                      "mesoscale freq",
                      "Mid-frequency fBm freq for sub-continent elevation "
                      "variation. 6 gives ~600-1000 km wavelength on Earth "
                      "(auto-scales with planet_radius_km). Fills the gap "
                      "between regional (~continent-scale) and hill (~mountain "
                      "spacing) so plateaus and lowlands have internal relief.",
                      "", o);
    reg.declare_int  ("bedrock.mesoscale_octaves",     5,     1,    8,
                      "mesoscale octaves", "fBm octaves for mesoscale layer", "", o);
    reg.declare_float("bedrock.mesoscale_amp_km",      1.5f,  0.0f,  5.0f,
                      "mesoscale amp",
                      "RMS amplitude of the mesoscale layer. The actual range "
                      "is roughly +/- this value across the planet. Also "
                      "controls how 'deep' dune basins get since the dune mask "
                      "reads the mesoscale field.", "km", o);

    //Tectonic plate boundary mask (v8 replaces belt fBm)
    reg.declare_float("bedrock.plate_freq",            0.8f,  0.1f,  4.0f,
                      "plate freq",
                      "Worley plant-point density. Each plant defines a 'plate'; "
                      "0.8 gives ~10 plates around an Earth-sized planet "
                      "(scaled by planet_radius_km). Lower = fewer larger plates.",
                      "", o);
    reg.declare_float("bedrock.plate_warp_freq",       0.4f,  0.05f, 3.0f,
                      "plate warp base freq",
                      "Base octave freq for the multi-octave warp that curves "
                      "plate boundaries away from polygonal Voronoi seams into "
                      "Andes/Himalaya-shaped arcs.", "", o);
    reg.declare_float("bedrock.plate_warp_amount",     0.18f, 0.0f,  0.50f,
                      "plate warp amount",
                      "Total warp displacement budget on plate positions. Larger = "
                      "more boundary curvature. >0.25 starts to make plates fold "
                      "onto themselves.", "", o);
    reg.declare_int  ("bedrock.plate_warp_octaves",    2,     1,    5,
                      "plate warp octaves",
                      "Octaves of the plate-warp fBm. 2-3 keeps boundaries "
                      "smoothly curving; higher adds wiggle but can produce "
                      "self-intersection.", "", o);
    reg.declare_float("bedrock.boundary_sharpness",    12.0f, 1.0f,  60.0f,
                      "boundary sharpness",
                      "exp(-sharpness * dist_to_boundary) falloff. 12 gives a ~250 "
                      "km wide mountain strip along each convergent boundary at "
                      "Earth radius; higher = narrower belts, lower = broader.",
                      "", o);

    //Convergence noise: which plate boundaries become mountain ranges.
    reg.declare_float("bedrock.convergence_freq",      1.5f,  0.2f,  8.0f,
                      "convergence freq",
                      "fBm freq for the noise that picks which boundaries are "
                      "convergent. Sub-plate scale (lower than plate_freq) so a "
                      "whole boundary arc tends to be coherently mountain-building "
                      "or coherently flat, not flickering segment by segment.",
                      "", o);
    reg.declare_int  ("bedrock.convergence_octaves",   3,     1,    6,
                      "convergence octaves", "fBm octaves for convergence noise",
                      "", o);
    reg.declare_float("bedrock.convergence_threshold", 0.55f, 0.20f, 0.90f,
                      "convergence threshold",
                      "Above this value the boundary is convergent (mountain-building). "
                      "0.55 ~ 45% of boundaries get mountains. Higher = sparser ranges.",
                      "", o);
    reg.declare_float("bedrock.convergence_softness",  0.08f, 0.005f, 0.30f,
                      "convergence softness",
                      "Smoothstep width on the convergence threshold.", "", o);

    //Mountain ridges
    reg.declare_float("bedrock.mountain_warp_freq",   30.0f,  0.5f, 200.0f,
                      "mountain warp base freq",
                      "Base octave freq of the multi-octave warp on the ridged mfbm input. "
                      "Higher octaves up to lacunarity^(N-1) * base reach the ridge scale "
                      "and break Perlin's closed contours into chain-like fragments.",
                      "", o);
    reg.declare_float("bedrock.mountain_warp_amount",  0.04f, 0.0f, 0.25f,
                      "mountain warp amount",
                      "Total ridge-warp displacement budget (shared across octaves). "
                      "0.03-0.06 gives natural ridge curvature. Larger values bend ridges "
                      "into long sinuous chains but if they exceed ~0.1 the chains start "
                      "to fold onto themselves.", "", o);
    reg.declare_int  ("bedrock.mountain_warp_octaves", 4,     1,    6,
                      "mountain warp octaves",
                      "Octaves of the displacement fBm. 4 with base 30 spans 30..240 "
                      "(crosses the default mountain_freq=120) - the high octave "
                      "dissolves the closed-contour 'bubble' pattern of ridged Perlin.",
                      "", o);
    reg.declare_float("bedrock.mountain_freq",       120.0f,  10.0f, 600.0f,
                      "mountain freq",
                      "Ridged multifractal base freq. CRITICAL knob - controls ridge "
                      "spacing. ~120 gives ridges ~170 km apart (~8 px at N=1024). "
                      "Lower values produce smooth blobs that look like bubbles.",
                      "", o);
    reg.declare_int  ("bedrock.mountain_octaves",      6,     1,   12,
                      "mountain octaves",
                      "Ridged multifractal octave count. 5-7 is the sweet spot; "
                      "more = aliasing at preview resolution.", "", o);
    reg.declare_float("bedrock.mountain_amp_km",       4.5f,  0.0f, 15.0f,
                      "mountain amp",
                      "Peak ridge-line height in belts. Leaves headroom for the peak "
                      "layer (default peak_amp_km=3.0) so total summit height tops out "
                      "around 7-8 km - Himalaya range.", "km", o);
    reg.declare_float("bedrock.mountain_exponent",     2.6f,  1.0f,  4.0f,
                      "mountain exponent",
                      "Ridge peak sharpness (1.0 = soft rounded, 2.5 = jagged Alps, "
                      "4.0 = razor crests)", "", o);
    reg.declare_float("bedrock.mountain_lacunarity",   2.0f,  1.5f,  3.0f,
                      "mountain lacunarity", "Freq multiplier per octave", "", o);
    reg.declare_float("bedrock.mountain_gain",         0.55f, 0.3f,  0.8f,
                      "mountain gain",
                      "Amplitude multiplier per octave. Higher = more high-freq "
                      "energy = sharper-looking mountains.", "", o);
    reg.declare_float("bedrock.mountain_regional_lo",  0.30f, 0.0f,  1.0f,
                      "mountain in lowlands",
                      "Mountain amplitude in deepest lowlands (0 = no mountains there)",
                      "", o);
    reg.declare_float("bedrock.mountain_regional_hi",  1.00f, 0.0f,  1.0f,
                      "mountain in highlands",
                      "Mountain amplitude in highest highlands", "", o);

    //Singular peaks
    reg.declare_float("bedrock.peak_freq",            80.0f,  5.0f, 400.0f,
                      "peak freq",
                      "Worley-distance peak density. Sets the typical spacing of "
                      "isolated summits: ~80 = one plant point per ~80 km on Earth "
                      "(scaled by planet_radius_km). Set to 0-equivalent low values "
                      "for continent-scale single peaks; high values for dense peak "
                      "fields.", "", o);
    reg.declare_float("bedrock.peak_radius",           0.55f, 0.05f, 1.50f,
                      "peak radius",
                      "Falloff radius of each peak in cell-size units. 0.5 = each "
                      "peak's influence dies within half a cell of its center "
                      "(compact, isolated peaks). 1.0+ makes neighbouring peaks "
                      "blend together.", "", o);
    reg.declare_float("bedrock.peak_sharpness",        4.0f,  1.0f, 16.0f,
                      "peak sharpness",
                      "Profile exponent on (1 - r/peak_radius). 1.0 = linear cone, "
                      "4.0 = sharp Gaussian-style summit, 8.0+ = needle-like spikes.",
                      "", o);
    reg.declare_float("bedrock.peak_amp_km",           3.0f,  0.0f, 10.0f,
                      "peak amp",
                      "Maximum height of a singular peak above the ridge it sits on. "
                      "Stacks with mountain_amp_km, so total summit height = ridge "
                      "amp + peak amp.", "km", o);

    //Hills
    reg.declare_float("bedrock.hill_freq",            45.0f,  5.0f, 150.0f,
                      "hill freq",
                      "Mid-freq fBm freq for rolling hills (~5 deg = 560 km)", "", o);
    reg.declare_int  ("bedrock.hill_octaves",          4,     1,    8,
                      "hill octaves", "fBm octaves for hills", "", o);
    reg.declare_float("bedrock.hill_amp_km",           0.40f, 0.0f,  3.0f,
                      "hill amp",
                      "Hill amplitude RMS. Kept moderate so plains read as plains "
                      "rather than busy textured noise that competes with the "
                      "mountain belt signal.", "km", o);

    //Lowland chaos
    reg.declare_float("bedrock.lowland_rough_freq",   70.0f, 10.0f, 300.0f,
                      "lowland rough freq",
                      "Ridged-noise freq for lowland chaos terrain (~3 deg)", "", o);
    reg.declare_int  ("bedrock.lowland_rough_octaves", 4,     1,    8,
                      "lowland rough octaves", "Octaves for lowland chaos", "", o);
    reg.declare_float("bedrock.lowland_rough_amp_km",  0.45f, 0.0f,  2.0f,
                      "lowland rough amp", "Lowland chaos amplitude", "km", o);

    //Fine
    reg.declare_float("bedrock.fine_freq",           280.0f, 60.0f, 800.0f,
                      "fine freq", "High-freq detail noise (~1.2 deg, ~130 km)", "", o);
    reg.declare_int  ("bedrock.fine_octaves",          3,     1,    6,
                      "fine octaves", "Detail noise octaves", "", o);
    reg.declare_float("bedrock.fine_amp_km",           0.15f, 0.0f,  0.6f,
                      "fine amp", "Detail roughness amplitude", "km", o);

    //Terrain variation (v9). One mask that decorrelates per-region roughness
    //from the regional dichotomy, so smooth plains and rough plains can sit
    //next to each other.
    reg.declare_float("bedrock.terrain_var_freq",      1.8f,  0.2f,  8.0f,
                      "terrain variation freq",
                      "fBm freq of the per-region roughness modulator. 1.8 gives "
                      "~11 regions around an Earth-sized equator. Lower = bigger "
                      "regions (one whole hemisphere is smooth), higher = patchier.",
                      "", o);
    reg.declare_int  ("bedrock.terrain_var_octaves",   4,     1,    8,
                      "terrain variation octaves",
                      "fBm octaves for the variation mask", "", o);
    reg.declare_float("bedrock.terrain_var_amp",       0.7f,  0.0f,  1.0f,
                      "terrain variation amount",
                      "Modulation strength. 0 = uniform (every region has the same "
                      "small-scale amplitudes, the v8 behaviour). 0.7 = amplitudes "
                      "vary from 30%% to 170%% across regions. 1.0 = full range "
                      "(some regions completely smooth, others 2x amplitude).",
                      "", o);

    //Mountain shape variation (v10) - reuses terrain_var_n
    reg.declare_float("bedrock.mountain_freq_var",     0.5f,  0.0f,  1.0f,
                      "mountain freq variation",
                      "How much mountain ridge SPACING varies across regions. 0 = "
                      "all belts have the same ridge spacing (v9 behaviour). 0.5 = "
                      "freq varies from 50%% to 150%% (Alps-style fine ridges in "
                      "some belts, older-Caledonides-style coarse ridges in others).",
                      "", o);
    reg.declare_float("bedrock.mountain_exp_var",      0.3f,  0.0f,  1.0f,
                      "mountain sharpness variation",
                      "How much ridge SHARPNESS varies across regions. 0 = uniform "
                      "sharpness everywhere. 0.3 = exp scaled 70%% to 130%% (some "
                      "belts soft/rounded, others jagged crests).",
                      "", o);

    //Dune regime (v10) - replaces peaks/mountains in smooth zones
    reg.declare_float("bedrock.dune_threshold",        0.25f, 0.0f,  0.95f,
                      "dune regime threshold",
                      "Cells with terrain_var_n below this value become dune zones "
                      "(mountains/peaks suppressed, replaced with directional "
                      "ripple). With Perlin-fBm std ~0.15, 0.25 puts ~15%% of the "
                      "surface in the dune regime. 0 disables dunes entirely.",
                      "", o);
    reg.declare_float("bedrock.dune_softness",         0.08f, 0.005f, 0.30f,
                      "dune transition softness",
                      "Smoothstep width on the dune threshold.", "", o);
    reg.declare_float("bedrock.dune_freq",             90.0f, 5.0f, 600.0f,
                      "dune freq",
                      "Sin-wave freq for dune crests. 90 gives ~4 deg wavelength "
                      "(~440 km on Earth) - large 'draa' scale, visible from "
                      "planet view. Bump for finer ripple if your bake resolves it.",
                      "", o);
    reg.declare_float("bedrock.dune_warp_freq",        3.0f,  0.3f, 30.0f,
                      "dune warp freq",
                      "Base freq of the position-warp that varies dune crest "
                      "direction across the planet. Low = dunes run in long "
                      "smoothly-curving lines; high = dunes wiggle a lot.",
                      "", o);
    reg.declare_float("bedrock.dune_warp_amount",      0.07f, 0.0f, 0.30f,
                      "dune warp amount",
                      "Magnitude of the dune position-warp. 0.07 gives natural "
                      "long-curving dune crests like Olympia Undae.", "", o);
    reg.declare_float("bedrock.dune_amp_km",           0.4f,  0.0f, 3.0f,
                      "dune amplitude", "Peak dune-crest height", "km", o);

    //Age
    reg.declare_float("bedrock.age_freq",              2.8f,  0.3f, 12.0f,
                      "age freq", "fBm freq for crust-age variation", "", o);
    reg.declare_float("bedrock.age_mid_myr",         2500.0f, 0.0f, 4500.0f,
                      "age mid", "Mean crust age (ancient rocky body)", "Myr", o);
    reg.declare_float("bedrock.age_var_myr",         1500.0f, 0.0f, 4000.0f,
                      "age var", "Age half-range. Final age = mid +- var.", "Myr", o);

    reg.declare_int  ("bedrock.seed",                  static_cast<int>(0xC0FFEE17),
                      std::numeric_limits<int>::min(),
                      std::numeric_limits<int>::max(),
                      "seed", "RNG seed (cast to uint32)", "", o);
}

void BedrockNoisePass::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    BedrockNoiseParams p;
    p.planet_radius_km      = reg.get_float("bedrock.planet_radius_km");
    p.lowland_base_km       = reg.get_float("bedrock.lowland_base_km");
    p.highland_base_km      = reg.get_float("bedrock.highland_base_km");
    p.regional_warp_freq    = reg.get_float("bedrock.regional_warp_freq");
    p.regional_warp_amount  = reg.get_float("bedrock.regional_warp_amount");
    p.regional_warp_octaves = reg.get_int  ("bedrock.regional_warp_octaves");
    p.regional_freq         = reg.get_float("bedrock.regional_freq");
    p.regional_octaves      = reg.get_int  ("bedrock.regional_octaves");
    p.plateau_threshold     = reg.get_float("bedrock.plateau_threshold");
    p.plateau_softness      = reg.get_float("bedrock.plateau_softness");
    p.plateau_lift_km       = reg.get_float("bedrock.plateau_lift_km");
    p.mesoscale_freq        = reg.get_float("bedrock.mesoscale_freq");
    p.mesoscale_octaves     = reg.get_int  ("bedrock.mesoscale_octaves");
    p.mesoscale_amp_km      = reg.get_float("bedrock.mesoscale_amp_km");
    p.plate_freq            = reg.get_float("bedrock.plate_freq");
    p.plate_warp_freq       = reg.get_float("bedrock.plate_warp_freq");
    p.plate_warp_amount     = reg.get_float("bedrock.plate_warp_amount");
    p.plate_warp_octaves    = reg.get_int  ("bedrock.plate_warp_octaves");
    p.boundary_sharpness    = reg.get_float("bedrock.boundary_sharpness");
    p.convergence_freq      = reg.get_float("bedrock.convergence_freq");
    p.convergence_octaves   = reg.get_int  ("bedrock.convergence_octaves");
    p.convergence_threshold = reg.get_float("bedrock.convergence_threshold");
    p.convergence_softness  = reg.get_float("bedrock.convergence_softness");
    p.mountain_warp_freq    = reg.get_float("bedrock.mountain_warp_freq");
    p.mountain_warp_amount  = reg.get_float("bedrock.mountain_warp_amount");
    p.mountain_warp_octaves = reg.get_int  ("bedrock.mountain_warp_octaves");
    p.mountain_freq         = reg.get_float("bedrock.mountain_freq");
    p.mountain_octaves      = reg.get_int  ("bedrock.mountain_octaves");
    p.mountain_amp_km       = reg.get_float("bedrock.mountain_amp_km");
    p.mountain_exponent     = reg.get_float("bedrock.mountain_exponent");
    p.mountain_lacunarity   = reg.get_float("bedrock.mountain_lacunarity");
    p.mountain_gain         = reg.get_float("bedrock.mountain_gain");
    p.mountain_regional_lo  = reg.get_float("bedrock.mountain_regional_lo");
    p.mountain_regional_hi  = reg.get_float("bedrock.mountain_regional_hi");
    p.peak_freq             = reg.get_float("bedrock.peak_freq");
    p.peak_radius           = reg.get_float("bedrock.peak_radius");
    p.peak_sharpness        = reg.get_float("bedrock.peak_sharpness");
    p.peak_amp_km           = reg.get_float("bedrock.peak_amp_km");
    p.hill_freq             = reg.get_float("bedrock.hill_freq");
    p.hill_octaves          = reg.get_int  ("bedrock.hill_octaves");
    p.hill_amp_km           = reg.get_float("bedrock.hill_amp_km");
    p.lowland_rough_freq    = reg.get_float("bedrock.lowland_rough_freq");
    p.lowland_rough_octaves = reg.get_int  ("bedrock.lowland_rough_octaves");
    p.lowland_rough_amp_km  = reg.get_float("bedrock.lowland_rough_amp_km");
    p.fine_freq             = reg.get_float("bedrock.fine_freq");
    p.fine_octaves          = reg.get_int  ("bedrock.fine_octaves");
    p.fine_amp_km           = reg.get_float("bedrock.fine_amp_km");
    p.terrain_var_freq      = reg.get_float("bedrock.terrain_var_freq");
    p.terrain_var_octaves   = reg.get_int  ("bedrock.terrain_var_octaves");
    p.terrain_var_amp       = reg.get_float("bedrock.terrain_var_amp");
    p.mountain_freq_var     = reg.get_float("bedrock.mountain_freq_var");
    p.mountain_exp_var      = reg.get_float("bedrock.mountain_exp_var");
    p.dune_threshold        = reg.get_float("bedrock.dune_threshold");
    p.dune_softness         = reg.get_float("bedrock.dune_softness");
    p.dune_freq             = reg.get_float("bedrock.dune_freq");
    p.dune_warp_freq        = reg.get_float("bedrock.dune_warp_freq");
    p.dune_warp_amount      = reg.get_float("bedrock.dune_warp_amount");
    p.dune_amp_km           = reg.get_float("bedrock.dune_amp_km");
    p.age_freq              = reg.get_float("bedrock.age_freq");
    p.age_mid_myr           = reg.get_float("bedrock.age_mid_myr");
    p.age_var_myr           = reg.get_float("bedrock.age_var_myr");
    p.seed                  = static_cast<std::uint32_t>(reg.get_int("bedrock.seed"));

    //Planet-radius scaling. Multiply every frequency knob by (radius / 6371)
    //so a given freq value produces the same km-spacing of features on any
    //rocky body. Earth radius (6371 km) is identity, so existing tunings on
    //Earth-scale planets are unchanged. Warp amounts and amplitudes are not
    //scaled; the user tunes those per body if they want different relief.
    constexpr float kReferenceRadiusKm = 6371.0f;
    const float rs = p.planet_radius_km / kReferenceRadiusKm;
    p.regional_warp_freq *= rs;
    p.regional_freq      *= rs;
    p.mesoscale_freq     *= rs;
    p.plate_freq         *= rs;
    p.plate_warp_freq    *= rs;
    p.convergence_freq   *= rs;
    p.mountain_warp_freq *= rs;
    p.mountain_freq      *= rs;
    p.peak_freq          *= rs;
    p.hill_freq          *= rs;
    p.lowland_rough_freq *= rs;
    p.fine_freq          *= rs;
    p.terrain_var_freq   *= rs;
    p.dune_freq          *= rs;
    p.dune_warp_freq     *= rs;
    p.age_freq           *= rs;

    progress.stage("noise");
    progress.fraction(0.0f);
    run_synth(state.bedrock_elevation,
              state.crust_thickness,
              state.crust_density,
              state.crust_age,
              state.crust_type,
              p);
    PB_LOG_INFO("bedrock_noise",
                "rocky body synthesised: radius %.0f km (scale %.3f), regional "
                "%d octaves @ freq %.2f, mountain %d-oct mfbm @ freq %.2f, "
                "tectonic plate_freq %.2f, convergence threshold %.2f",
                static_cast<double>(p.planet_radius_km),
                static_cast<double>(rs),
                p.regional_octaves, static_cast<double>(p.regional_freq),
                p.mountain_octaves, static_cast<double>(p.mountain_freq),
                static_cast<double>(p.plate_freq),
                static_cast<double>(p.convergence_threshold));
    progress.fraction(1.0f);
}

}
