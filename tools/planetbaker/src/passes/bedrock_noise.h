#pragma once

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

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
    std::uint64_t version() const override { return 11; }

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
