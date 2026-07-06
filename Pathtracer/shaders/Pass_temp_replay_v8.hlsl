#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"   // HitContext (replayed bounce-loop ctx)
#include "Hybrid_Replay_v8.hlsli"
#define TEMPORAL_CAN_REPLAY 1
#include "Temporal_Merge_v8.hlsli"

//====================================
//TEMPORAL GI  (pass 2/2: k>2 replay merges) — COMPACTED INDIRECT
//====================================
//Consumes the temporal replay queue that Pass_temp_gi filled (g_raygenQueue
//counter at byte 4, packed pixel entries from byte 16; the parked reprojected
//candidate coordinate sits in SPM_w1). Runs the SAME TemporalMergeBody, now
//with TEMPORAL_CAN_REPLAY=1 so k>2 directions evaluate through the hybrid
//prefix replay (the only reuse-side TraceRay site). Dispatched 1D over exactly
//the queued count via ExecuteIndirect — pixels with pure k==2 shifts never pay
//for this pass's live state.

[shader("raygeneration")]
void Pass_temp_replay_v8()
{
    const uint  packedPx = g_raygenQueue.Load(16u + DispatchRaysIndex().x * 4u);
    const uint2 pixel    = uint2(packedPx & 0xFFFFu, packedPx >> 16);
    const uint2 imgSize  = uint2(IMG_W, IMG_H);
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    //parked candidate: resolved coordinate + the dual-MV band policy (bit31)
    const uint permPk = g_pathStateBuffer.Load(SPM_w1(pixelIdx));
    const int2 permCoord = int2((int)(permPk & 0xFFFFu), (int)((permPk >> 16) & 0x7FFFu));
    const bool dualCand  = (permPk & 0x80000000u) != 0u;

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    TemporalMergeBody(pixelIdx, pixel, permCoord, dualCand);
}
