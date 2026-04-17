#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI - Neighbor Selection Pre-pass  (single neighbor)
//─────────────────────────────────────────────────────────────────────────────

// Per-pixel layout in g_pathStateBuffer (8 bytes, linear y*W+x indexing):
//   offset 0: uint validCount  (0 or 1)
//   offset 4: uint nID         (0xFFFFFFFF when invalid)
static const uint GI_SEL_STRIDE = 8u;

uint gi_sel_addr(uint linearIdx) { return linearIdx * GI_SEL_STRIDE; }

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const uint2  launchIndex = tid.xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   linearIdx   = launchIndex.y * IMG_W + launchIndex.x;
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    const uint baseAddr = gi_sel_addr(linearIdx);

    // Emitter or spatial GI disabled -> no neighbor
    if (load_isEmitter(g_sample_current, pixelIdx) || !(rs_flags & 8u))
    {
        g_pathStateBuffer.Store2(baseAddr, uint2(0u, 0xFFFFFFFFu));
        return;
    }

    // Lightweight loads for rejection
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);

    // Pick at most 1 valid neighbor across rs_spatTriesGI attempts.
    // Radius shrinks linearly from rs_spatRadMaxGI to rs_spatRadMinGI over all tries.
    uint selectedID = 0xFFFFFFFFu;
    uint validCount = 0u;

    const uint totalTries = max(2u, rs_spatTriesGI);

    [loop]
    for (uint i = 0; i < totalTries && validCount == 0u; ++i)
    {
        float t      = float(i) / float(totalTries - 1u);
        uint  radius = (uint)lerp(float(rs_spatRadMaxGI), float(rs_spatRadMinGI), t);

        const uint iID = GetRandomPixelCircleWeighted(
            radius, dims.x, dims.y,
            launchIndex.x, launchIndex.y,
            seed);

        bool ok = false;
        if (!load_isEmitter(g_sample_current, iID))
        {
            uint nInstID_t = load_instID(g_sample_current, iID);
            uint nPrimID_t = load_primID(g_sample_current, iID);
            if (GetMatIDFast(nInstID_t, nPrimID_t) == myMatID)
            {
                const float3 n1s_r = load_n1_s_with_instID(g_sample_current, iID, nInstID_t);
                if (!RejectNormal_GI(myN1s, n1s_r, 0.36f))
                {
                    float2 nBary_t = load_bary(g_sample_current, iID);
                    const float3 x1_r = ReconstructPosition(nInstID_t, nPrimID_t, nBary_t);
                    if (!RejectDistance_GI(myPos, x1_r, myN1s, 0.1f))
                        ok = true;
                }
            }
        }

        if (!ok) continue;

        // Lightweight validity: M > 0 is sufficient for pre-selection
        uint rM = load_M_gi(g_Reservoirs_current_gi, iID);
        if (rM > 0u)
        {
            selectedID = iID;
            validCount = 1u;
        }
    }

    // Write (validCount, nID)
    g_pathStateBuffer.Store2(baseAddr, uint2(validCount, selectedID));
}