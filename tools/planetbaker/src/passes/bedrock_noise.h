#pragma once

#include <cstdint>

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

class ParamRegistry;

//====================================
//BedrockNoiseParams: all knobs the per-pixel noise kernel consumes. Moved
//out of the anonymous .cu namespace so BakePass can synthesize bedrock
//directly at the bake resolution instead of bicubic-upsampling the working
//PlanetState (the upsample blew the file size up without adding any
//frequency content beyond the source grid's Nyquist).
//
//Defaults match the values in declare_params(); load_bedrock_params() reads
//the live registry into this struct and applies the planet-radius freq
//scaling, so the result is ready to feed straight to a synthesis kernel.
//====================================
struct BedrockNoiseParams {
    float planet_radius_km      = 6371.0f;

    //v16 "alien dramatic" preset. Continental dichotomy reduced to a whisper
    //(+/-0.2 km total span) so the planet's silhouette is dominated by the
    //mountain belts + near-camera detail, not by the regional fBm hemispheres
    //which were drowning everything in Earth-scale v13. plateau and mesoscale
    //also compressed proportionally so they don't reintroduce big continental
    //bumps. See changelog above for the rationale.
    float lowland_base_km       = -0.1f;
    float highland_base_km      = 0.2f;

    float regional_warp_freq    = 0.9f;
    float regional_warp_amount  = 0.30f;
    int   regional_warp_octaves = 3;
    float regional_freq         = 2.5f;
    int   regional_octaves      = 7;

    float plateau_threshold     = 0.68f;
    float plateau_softness      = 0.05f;
    float plateau_lift_km       = 0.2f;   // v16: tiny plateau bumps

    float mesoscale_freq        = 6.0f;
    int   mesoscale_octaves     = 5;
    float mesoscale_amp_km      = 0.15f;  // v16: tiny mesoscale

    float plate_freq            = 0.8f;
    float plate_warp_freq       = 0.4f;
    float plate_warp_amount     = 0.18f;
    int   plate_warp_octaves    = 2;
    float boundary_sharpness    = 12.0f;
    float convergence_freq      = 1.5f;
    int   convergence_octaves   = 3;
    float convergence_threshold = 0.55f;
    float convergence_softness  = 0.08f;

    //v15: revert v14's mountain shape changes - they pushed energy past
    //the 16k bake's Nyquist (5 km wavelength) and the 4k GPU heightmap's
    //Nyquist (20 km), so the "sharper peaks" became aliased noise at every
    //view distance. Back to v13's smoother-but-resolvable values.
    float mountain_warp_freq    = 30.0f;
    float mountain_warp_amount  = 0.04f;
    int   mountain_warp_octaves = 4;
    float mountain_freq         = 120.0f;
    int   mountain_octaves      = 6;
    float mountain_amp_km       = 12.0f;  // v16: Olympus-Mons class belt peaks
    float mountain_exponent     = 2.6f;
    float mountain_lacunarity   = 2.0f;
    float mountain_gain         = 0.55f;
    float mountain_regional_lo  = 0.30f;
    float mountain_regional_hi  = 1.00f;

    float peak_freq             = 80.0f;
    float peak_radius           = 0.55f;
    float peak_sharpness        = 4.0f;
    float peak_amp_km           = 4.0f;   // v16: stacks with mountain_amp for ~+16 km total summits

    float hill_freq             = 45.0f;
    int   hill_octaves          = 4;
    float hill_amp_km           = 1.0f;   // v16: visible rolling-hills texture in plains

    float lowland_rough_freq    = 70.0f;
    int   lowland_rough_octaves = 4;
    float lowland_rough_amp_km  = 1.0f;   // v16: chaotic-basin texture in lowlands

    //fine layer 1 (v16 amp: ~1 km of high-freq detail in every region,
    //so plains carry visible noise instead of reading as flat between
    //the geological-scale layers). Frequency varies across the surface
    //via fine_freq_var driven by terrain_var_n.
    float fine_freq             = 280.0f;
    int   fine_octaves          = 3;
    float fine_amp_km           = 1.0f;
    float fine_freq_var         = 0.5f;   // local freq = base * mix(1-var, 1+var, terrain_var_n)

    //fine layer 2 (v16 amp: half of layer 1, ~500 m). Higher base freq
    //for a different scale; freq variation driven by mesoscale_n so its
    //spatial pattern decorrelates from layer 1.
    float fine2_freq            = 700.0f;
    int   fine2_octaves         = 3;
    float fine2_amp_km          = 0.5f;
    float fine2_freq_var        = 0.5f;   // local freq = base * mix(1-var, 1+var, mesoscale_n)

    float terrain_var_freq      = 1.8f;
    int   terrain_var_octaves   = 4;
    float terrain_var_amp       = 0.7f;

    float mountain_freq_var     = 0.5f;
    float mountain_exp_var      = 0.3f;

    float dune_threshold        = 0.25f;
    float dune_softness         = 0.08f;
    float dune_freq             = 90.0f;
    float dune_warp_freq        = 3.0f;
    float dune_warp_amount      = 0.07f;
    float dune_amp_km           = 0.5f;   // v16: visible dune crests in low-terrain_var regions

    float age_freq              = 2.8f;
    float age_mid_myr           = 2500.0f;
    float age_var_myr           = 1500.0f;

    std::uint32_t seed          = 0xC0FFEE17u;
};

//====================================
//Reads `bedrock.*` params from the registry and applies the planet-radius
//frequency scaling (so the returned struct is ready for a synthesis launch).
//The pass's own run() uses the same logic - this function lets BakePass
//share the loaded params without rerunning the pass.
//====================================
BedrockNoiseParams load_bedrock_params(const ParamRegistry& reg);

//====================================
//Per-pixel bedrock synthesis at an arbitrary resolution, single face. Writes
//`n * n` floats (no halo, no padding) into `dst_device` in row-major order;
//the value is the bedrock elevation in KILOMETRES at the equiangular
//cubed-sphere cell (face, i, j). face must be in [0, 6).
//
//Used by BakePass to write the on-disk cubemap at its OWN resolution
//instead of upsampling a low-resolution working grid. CUDA host code -
//synchronises on the default stream before return.
//====================================
void bake_bedrock_face(int face, int n,
                       const BedrockNoiseParams& P,
                       float* dst_device);


//====================================
//BedrockNoisePass: per-pixel layered noise synthesis of a rocky planet's
//bedrock elevation and crust fields. Replaces the prior plate-tectonics
//simulation entirely. Output is a deterministic function of (position,
//seed, params) evaluated independently per cubed-sphere cell, so there is
//no mesh granularity in the result; coastlines and ridge chains are smooth
//continuous curves at whatever resolution you bake at.
//
//Layers (each evaluated at every pixel and added):
//  1. Continental mask          - low-freq domain-warped fBm thresholded
//                                 into land vs ocean basins. Soft via smoothstep.
//  2. Mountain-belt mask        - mid-freq fBm thresholded so ridges
//                                 cluster into chains, not cover all land.
//  3. Mountain ridges            - ridged multifractal at high freq,
//                                 domain-warped so ridges curve.
//  4. Continental hills          - mid-freq fBm at modest amplitude.
//  5. Plateau lift               - smoothstep at upper continental tail
//                                 produces flat-topped highlands.
//  6. Abyssal hills              - ridged fBm at high freq on the ocean side.
//  7. Fine roughness             - high-freq fBm at small amplitude,
//                                 everywhere.
//
//ImpactsPass (M5) is expected to stamp craters on top of this layer; it
//reads bedrock_elevation, adds craters, writes bedrock_elevation back.
//Erosion passes downstream further sculpt the result.
//====================================

class BedrockNoisePass : public Pass {
public:
    const char*   name()    const override { return "bedrock_noise"; }
    //declare_params registers under "bedrock.*" for terseness (the keys
    //pre-date the pass rename); override the default `<name>.` so dirty
    //tracking, cache hashing and the UI prefix_iter walk all find them.
    std::string   param_prefix() const override { return "bedrock."; }
    //v4: multi-octave fBm domain warp + macro contrast bump.
    //v3's single-octave warp couldn't simultaneously bend continents AND
    //distort individual ridges. Result: ridged-Perlin's zero-crossings stayed
    //as closed contour loops, producing a 'bubble network' look no matter
    //how the single-octave knobs were tuned. v4 replaces d_warp with a
    //multi-octave fBm warp so low octaves bend continent-scale features and
    //high octaves break individual ridges - in one primitive. Also raises
    //regional base freq (1.0 -> 2.5) and octaves (5 -> 7) and steepens the
    //dichotomy curve, so the planet-wide shape has real macro structure
    //rather than a ~6-cell base-octave smear.
    //
    //v5: adds bedrock.planet_radius_km. All frequency knobs are tuned for
    //Earth scale (6371 km); at other radii every freq is scaled by
    //(radius / 6371) before kernel launch so the same freq value gives the
    //same km-spacing of features on any body. Identity at Earth radius, so
    //existing Earth-scale tunings stay bit-identical.
    //
    //v6: adds layer 4b - Worley-distance 'singular peaks' that stack on
    //top of the continuous ridge network. Models the isolated summits
    //(Matterhorn, K2, Olympus Mons) that pure ridged mfbm cannot make
    //because ridged Perlin only produces continuous ridge LINES, not
    //isolated POINTS. Also tunes defaults for clearer belts: belt_threshold
    //0.55 -> 0.62 (sparser ranges), mountain_exponent 2.3 -> 2.6 (sharper
    //ridges), mountain_amp_km 5.5 -> 4.5 (headroom for the new peak layer
    //so total summit height stays ~7-8 km), hill_amp_km 0.55 -> 0.40
    //(plains look like plains). Dendritic / branching valleys still come
    //from the downstream HydraulicErosionPass when its iterations > 0.
    //
    //v7: tightens the regional dichotomy smoothstep from (0.25, 0.75) to
    //(0.40, 0.60). v4-v6's wide band left most cells in the soft middle
    //of the curve, so the planet looked mid-tone everywhere with bright
    //highland blobs floating around rather than a clear Mars-style
    //highland-hemisphere vs lowland-hemisphere regime. The narrower band
    //saturates anything slightly above 0.5 to highland and anything
    //slightly below to lowland, producing a real bimodal macro shape.
    //
    //v8: replaces the random-fBm belt mask with a procedural tectonic
    //plate-boundary mask. Mountain belts now form along the edges of
    //jittered Voronoi 'plates' (Worley F2-F1 distance field), gated by
    //a separate convergence noise so only ~half the boundaries are
    //mountain-building - the rest are flat (transform / divergent).
    //Result is coherent Andes/Himalaya-shaped mountain arcs along plate
    //edges instead of random belt blobs at fBm noise peaks. Per-pixel
    //and mesh-free, so the prior Goldberg-mesh resampling / hex-Voronoi
    //artifacts can't recur. Param set changes: belt_freq, belt_octaves,
    //belt_threshold, belt_softness REMOVED; plate_freq, plate_warp_*,
    //boundary_sharpness, convergence_* ADDED.
    //
    //v9: adds a per-region terrain_var modulator. v8 applied the same
    //hill_amp / fine_amp / mountain_amp / peak_amp across the entire
    //planet, so every region had an identical roughness signature - a
    //smooth plain looked the same as a chaotic plain looked the same
    //as a heavily-cratered plain. v9 samples a low-freq fBm decorrelated
    //from the regional dichotomy and modulates each detail layer's
    //amplitude by mix(1-amp, 1+amp, var_n). Smooth regions and rough
    //regions can sit next to each other without changing any other knob.
    //
    //v10: extends terrain_var_n to drive landscape SHAPE as well as
    //amplitude. (a) Mountain ridge frequency and sharpness now vary per
    //region: rough zones get fine sharp ridges, smooth zones get coarser
    //rounded ridges. (b) Adds a 'dune regime' that activates where
    //terrain_var_n falls below dune_threshold - mountains and peaks are
    //hard-suppressed and replaced with a sin-based directional ripple
    //(crests bent into long curves by a multi-octave position warp,
    //Olympia-Undae style). Default dune_threshold=0.25 puts ~15%% of the
    //surface in the dune regime.
    //
    //v11: adds a MESOSCALE elevation layer (mid-freq fBm at freq ~6) that
    //bridges the gap between regional (~continent-scale) and hill scale,
    //so plateaus and lowlands have internal relief. Same field is also
    //repurposed as the dune-placement signal: dunes form where the
    //mesoscale field dips negative (local depressions) instead of v10's
    //tie to the coarse terrain_var. Dunes are now decorrelated from the
    //regional dichotomy and naturally biased to lower local elevations.
    //
    //v12: rebalances the layer stack for a "near-camera dominated" look.
    //Continental-scale amplitudes (lowland/highland/plateau/mesoscale/
    //mountain/peak/hill/lowland_rough/dune) all reduced ~20x; v11's
    //fine_amp boosted 10x. NEW fine2 layer stacks a second high-freq fBm
    //(default 2.5x finer base freq) on top of fine layer 1, so near-camera
    //pixels see two layered scales of detail instead of one. BOTH fine
    //layers also get FREQUENCY variation across the surface, driven by
    //decorrelated existing noise fields: fine layer 1 freq follows
    //terrain_var_n (correlates with mountain shape - rough regions get
    //finer near-camera detail); fine layer 2 freq follows mesoscale_n
    //(decorrelates from layer 1, gives spatial heterogeneity even where
    //terrain_var_n is uniform). Amplitudes still vary via the existing
    //roughness mask. Param adds: fine_freq_var, fine2_freq, fine2_octaves,
    //fine2_amp_km, fine2_freq_var. Default lowland/highland/plateau/
    //mesoscale/mountain/peak/hill/lowland_rough/dune amp values reduced 20x.
    //
    //v13: Earth-scale amplitudes. v12's 20x-reduced continental + 10x-
    //boosted fine ended up reading as "flat" because the geological-scale
    //layers were too quiet to register at any view distance and the fine
    //layer dominated everything within a few hundred metres of the camera.
    //v13 restores v11's amplitudes for continental, plateau, mesoscale,
    //mountain, peak, hill, lowland_rough, dune (so peaks reach ~9 km and
    //the planet is geologically structured like Earth) AND drops fine_amp
    //back to v11's 0.15 km. fine layer 2 sits at half of layer 1
    //(0.075 km / 75 m) for a subtle second-scale texture. The layered
    //frequency-variation architecture introduced in v12 stays. Param set
    //unchanged from v12; only default values change.
    //
    //v14: mountain ridge SHAPE knobs re-tuned to fix "no sharper peaks /
    //no mountains in the distance" - the v13 mountain layer was tall but
    //smooth (broad rounded ridges), so amp bumps just scaled the smooth
    //lumps up rather than adding visible detail. v14 increases base freq
    //(120->200), octave count (6->8), gain (0.55->0.70), and exponent
    //(2.6->3.5). Continental / fine layers and their amplitudes are
    //unchanged - the fix is purely in the mid-scale ridge frequency
    //content and peak sharpness. After v14, bumping mountain_amp_km
    //scales VISIBLE detail up instead of just smoother lumps. Param set
    //unchanged from v13; only mountain_freq, mountain_octaves,
    //mountain_gain, mountain_exponent defaults change.
    //
    //v15: REVERT v14's mountain shape changes. v14's highest mountain octave
    //was at ~1.6 km wavelength, well below the 16k bake's ~5 km Nyquist and
    //massively below the 4k GPU heightmap's ~20 km Nyquist - the "sharper
    //ridges" became aliasing artifacts that looked WORSE than v13's smoother
    //resolvable ridges, especially when the camera approached the surface.
    //v15 restores v13 mountain shape values. The actual fix for "no
    //close-to-surface detail" is renderer-side: either bump
    //TERRAIN_HEIGHTMAP_GPU_RESOLUTION (4k -> 8k or 16k) so the shading
    //normal sees finer features, or add a runtime procedural displacement
    //layer that doesn't depend on the baked heightmap resolution. Param set
    //and defaults match v13 for the mountain layer; everything else stays
    //at v13.
    //
    //v16: "alien dramatic" preset. v15's Earth-scale dichotomy made the
    //continental layer visually dominant (peaks of the +/-4 km dichotomy
    //showed as the main feature of the planet), drowning the mountain belts
    //and near-camera detail. v16 shrinks the continental layer to a whisper
    //(+/-0.2 km total span) and boosts mountain belts to Olympus-Mons class
    //(+12 km mountain_amp + 4 km peak_amp + plateau + highland = ~+16 km
    //summits in convergent belts), with proportionally larger hill /
    //lowland_rough / fine / fine2 amplitudes so plains carry ~+/-2 km of
    //visible texture - no more flat-looking expanses between belts.
    //
    //Implies kHeightExaggeration = 1.0 in the renderer; these absolute
    //amplitudes ARE the displayed relief, no multiplier needed.
    //
    //Param set unchanged from v15; only the amp defaults change.
    std::uint64_t version() const override { return 16; }

    FieldSet reads()  const override { return {}; }
    FieldSet writes() const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::CrustThickness)
            .set(FieldId::CrustDensity)
            .set(FieldId::CrustAge)
            .set(FieldId::CrustType);
    }

    void declare_params(ParamRegistry& reg) const override;
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) override;
};

}
