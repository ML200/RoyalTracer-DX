#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI — Neighbor Selection Pre-pass
//  Writes compact neighbor list to g_pathStateBuffer for the raygen merge pass.
//─────────────────────────────────────────────────────────────────────────────

// Per-pixel layout in g_pathStateBuffer (40 bytes, linear y*W+x indexing):
//   offset 0:  uint  validCount
//   offset 4:  float M_sum
//   offset 8:  uint  nIds[SPAT_COUNT_MAX_GI]  (8 × 4 = 32 bytes)
static const uint GI_SEL_STRIDE = 40u;

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

    // Default: 0 valid neighbors
    const uint baseAddr = gi_sel_addr(linearIdx);

    // Emitter or spatial GI disabled → no neighbors
    if (load_isEmitter(g_sample_current, pixelIdx) || !(rs_flags & 8u))
    {
        g_pathStateBuffer.Store2(baseAddr, uint2(0u, asuint(0.0f)));
        return;
    }

    // Lightweight loads for rejection
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myN1g    = load_n1_g_with_instID(g_sample_current, pixelIdx, myInstID);
    const float2 myUV     = load_uv(g_sample_current, pixelIdx);
    float3 myKd; float myPr, myPm;
    RefetchMaterial(myMatID, myUV, myKd, myPr, myPm);

    // Specularity: reduce spatial samples for reflective surfaces
    const float3 camPos      = InitOrigin();
    const float  NoV         = saturate(dot(normalize(camPos - myPos), myN1s));
    const float  specularity = Luma(EnvBRDFApprox2(myKd, myPr, myPm, NoV));

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);

    // Budgeting
    const float M_c_raw = load_M_gi(g_Reservoirs_current_gi, pixelIdx);
    const float conf = min(60.0f, M_c_raw) / max(1u, rs_tempMcapGI);

    const uint nbrBudget = uint(float(min(rs_spatCountMaxGI, SPAT_COUNT_MAX_GI)) * (1.0f - specularity) + 0.5f);

    const uint radiusBudget =
        rs_spatRadMinGI +
        uint((1.0f - conf) * float(rs_spatRadMaxGI - rs_spatRadMinGI) + 0.5f);

    // Neighbor selection
    uint  nIds[SPAT_COUNT_MAX_GI];
    uint  validCount = 0;
    float M_sum      = 0.0f;

    [loop]
    for (uint i = 0; i < nbrBudget; ++i)
    {
        const uint iID = GetRandomPixelCircleWeighted(
            radiusBudget, dims.x, dims.y,
            launchIndex.x, launchIndex.y,
            seed);

        bool ok = false;
        if (!load_isEmitter(g_sample_current, iID))
        {
            uint nInstID_t = load_instID(g_sample_current, iID);
            uint nPrimID_t = load_primID(g_sample_current, iID);
            if (GetMatIDFast(nInstID_t, nPrimID_t) == myMatID)
            {
                const float3 n1g_r = load_n1_g_with_instID(g_sample_current, iID, nInstID_t);
                if (!RejectNormal_GI(myN1g, n1g_r, 0.36f))
                {
                    float2 nBary_t = load_bary(g_sample_current, iID);
                    const float3 x1_r = ReconstructPosition(nInstID_t, nPrimID_t, nBary_t);
                    if (!RejectDistance_GI(myPos, x1_r, myN1g, 0.1f))
                        ok = true;
                }
            }
        }

        if (!ok)
            continue;

        // Load reservoir only for plausible candidates
        uint  rM   = load_M_gi(g_Reservoirs_current_gi, iID);
        uint  rObj = load_objID_gi(g_Reservoirs_current_gi, iID);
        float3 rN  = load_n2_s_gi(g_Reservoirs_current_gi, iID, rObj);

        if (IsValidReservoir_GI_opt(rN, rM))
        {
            nIds[validCount++] = iID;
            M_sum += min(SPAT_MCAP_GI, rM);
        }
    }

    // Write results: validCount, M_sum, and neighbor IDs
    g_pathStateBuffer.Store2(baseAddr, uint2(validCount, asuint(M_sum)));

    [unroll]
    for (uint k = 0; k < SPAT_COUNT_MAX_GI; ++k)
    {
        uint id = (k < validCount) ? nIds[k] : 0xFFFFFFFFu;
        g_pathStateBuffer.Store(baseAddr + 8u + k * 4u, id);
    }
}
