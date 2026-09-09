#include "Includes_v8.hlsli"
#include "Temporal_Merge_v8.hlsli"
#include "Temporal_Sampling_v8.hlsli"

//====================================
//TEMPORAL GI  (pass 1/3: candidate pick + resolve + route)
//====================================
//Owns the reprojection / permutation candidate choice, then runs
//TemporalMergeBody: writes the forward + reverse job descriptors Pass_shift_v8
//reads (dispatched right after as a single Depth=2 DispatchRays — the SAME
//one binary spatial's Depth=Ntn+1 dispatch also uses). Pass_temp_merge
//(stage 3/3) does the actual MIS combine + reservoir write once both this pass
//and the shift dispatch have finished. Keeps this full-screen pass free of
//TraceRay/RayQuery and any replay-walk live state entirely — it's pure
//candidate resolution + routing, no reconnection math at all.
//
//PESSIMISTIC STATUS: SPM_w0 is written to TEMP_STATUS_DEAD as the very FIRST
//thing, before any early-return check (emitter, tempGI-off, out-of-bounds
//permutation). TemporalMergeBody only overwrites it to TEMP_STATUS_OK once a
//candidate actually resolves; every other exit — including ones added here in
//the future — leaves a coherent status for Pass_temp_merge without needing a
//matching store at each return site.

[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    g_pathStateBuffer.Store(SPM_w0(pixelIdx), TEMP_STATUS_DEAD);
    //Pessimistic job clear, same reasoning as w0 above:
    //Pass_shift_v8's temporal_shift Depth-slices are dispatched
    //unconditionally by the host (the dispatch has no per-mode gate)
    //and check ONLY slotZ (not w0) to decide whether a job exists, so every
    //early-return path here — including TemporalMergeBody's own internal
    //"candidate dead" return — must leave "no job" behind, not whatever a
    //past frame (tempGI on, or a since-resolved candidate) last wrote.
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), SP_UNDEF);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 1u), SP_UNDEF);

    //emitter / disabled early out
    const uint myFlags = load_flagsWord(g_sample_current, pixelIdx);
    if (myFlags & SD_FLAG_EMITTER)
        return;
    if (!(rs_flags & 2u))
        return;

    const int2 permCoord = TemporalCandidatePixel(pixelIdx, launchIndex);
    if (any(permCoord < 0)) return;

    //shared merge body (resolves + writes the forward/reverse job descriptors
    //Pass_shift_v8 reads next — no queue, no inline shifting here)
    TemporalMergeBody(pixelIdx, launchIndex, permCoord, false);
}
