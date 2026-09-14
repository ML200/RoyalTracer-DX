#include "Includes_v8.hlsli"
#include "Temporal_Merge_v8.hlsli"
#include "Temporal_Sampling_v8.hlsli"

[shader("raygeneration")]
// Reproject and accumulate the temporal indirect-light estimate.
void Pass_temp_gi_v8()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    g_pathStateBuffer.Store(SPM_w0(pixelIdx), TEMP_STATUS_DEAD);

    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), SP_UNDEF);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 1u), SP_UNDEF);

    const uint myFlags = load_flagsWord(g_sample_current, pixelIdx);
    if (myFlags & SD_FLAG_EMITTER)
        return;
    if (!(rs_flags & 2u))
        return;

    const int2 permCoord = TemporalCandidatePixel(pixelIdx, launchIndex);
    if (any(permCoord < 0)) return;

    TemporalMergeBody(pixelIdx, launchIndex, permCoord, false);
}
