#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI - Neighbor Selection Pre-pass (paired reuse)
//
//  Partner selection uses precomputed self-inverting reuse textures
//  (Lin, Kettunen, Wyman 2026, §3). Each pixel samples delta (dx, dy)
//  from slot s of the reuse-texture stack; the partner at screen coord
//  (pixel + delta) samples the same slot and gets back the inverse
//  delta — guaranteeing both pixels see each other as partners.
//
//  Texture sizes (254, 230, 210) mirror Renderer::InitReuseTextures.
//─────────────────────────────────────────────────────────────────────────────

Texture2D<int2> g_reuseTexture0 : register(t19);
Texture2D<int2> g_reuseTexture1 : register(t20);
Texture2D<int2> g_reuseTexture2 : register(t21);

// Per-pixel scratch layout in g_pathStateBuffer (56 bytes):
//   offset 0:   uint  validCount              (this pass)
//   offset 4:   float my_Jc                   (filled by shift pass)
//   offset 8 + s*16 + 0:  uint  nID           (this pass; 0xFFFFFFFFu if slot s rejected)
//   offset 8 + s*16 + 4:  uint  F_pack        (shift pass)
//   offset 8 + s*16 + 8:  float F_mag         (shift pass, visibility baked)
//   offset 8 + s*16 + 12: float Jn            (shift pass)
// M_sum is recomputed in the merge pass from per-slot partner.M loads.
static const uint GI_SEL_STRIDE      = 56u;
static const uint GI_SEL_SLOT_BASE   = 8u;
static const uint GI_SEL_SLOT_STRIDE = 16u;

uint gi_sel_addr(uint linearIdx) { return linearIdx * GI_SEL_STRIDE; }
uint gi_sel_slot_addr(uint linearIdx, uint slot)
{
    return gi_sel_addr(linearIdx) + GI_SEL_SLOT_BASE + slot * GI_SEL_SLOT_STRIDE;
}

// Sample slot s, applying the per-frame offset + dihedral
int2 SampleReuseDelta(uint2 launchIndex, uint slot)
{
    uint2 offset;
    uint  flags;
    int2  texSize;

    if (slot == 0u)
    {
        offset  = uint2(rs_reuseOffset0_x, rs_reuseOffset0_y);
        flags   = rs_reuseFlags0;
        texSize = int2(254, 254);
    }
    else if (slot == 1u)
    {
        offset  = uint2(rs_reuseOffset1_x, rs_reuseOffset1_y);
        flags   = rs_reuseFlags1;
        texSize = int2(230, 230);
    }
    else
    {
        offset  = uint2(rs_reuseOffset2_x, rs_reuseOffset2_y);
        flags   = rs_reuseFlags2;
        texSize = int2(210, 210);
    }

    // Shift sampling origin and wrap into the textures tileable domain
    int2 cLookup = int2((launchIndex + offset) % uint2(texSize));

    // Apply lookup transforms
    if (flags & 4u) cLookup = cLookup.yx;
    if (flags & 1u) cLookup.x = texSize.x - 1 - cLookup.x;
    if (flags & 2u) cLookup.y = texSize.y - 1 - cLookup.y;

    int2 d;
    if (slot == 0u)      d = g_reuseTexture0.Load(int3(cLookup, 0));
    else if (slot == 1u) d = g_reuseTexture1.Load(int3(cLookup, 0));
    else                 d = g_reuseTexture2.Load(int3(cLookup, 0));

    // Inverse transforms on the returned delta
    if (flags & 1u) d.x = -d.x;
    if (flags & 2u) d.y = -d.y;
    if (flags & 4u) d = d.yx;

    return d;
}

// Symmetric pair rejection
bool PairRejected(uint aMat, float3 aPos, float3 aN,
                  uint bMat, float3 bPos, float3 bN)
{
    if (aMat != bMat) return true;
    if (RejectNormal_GI(aN, bN, 0.36f)) return true;
    if (RejectDistance_GI(aPos, bPos, aN, 0.1f)) return true;
    if (RejectDistance_GI(bPos, aPos, bN, 0.1f)) return true;
    return false;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const uint2  launchIndex = tid.xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    //Scratch is indexed by pixelIdx
    const uint baseAddr = gi_sel_addr(pixelIdx);

    //Emitter or spatial GI disabled -> no neighbors
    if (load_isEmitter(g_sample_current, pixelIdx) || !(rs_flags & 8u))
    {
        g_pathStateBuffer.Store(baseAddr, 0u);  // validCount=0
        return;
    }

    //Reservoir empty
    const uint myM = load_M_gi(g_Reservoirs_current_gi, pixelIdx);
    if (myM == 0u)
    {
        g_pathStateBuffer.Store(baseAddr, 0u);
        return;
    }

    //Lightweight loads for rejection
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);

    //Decompacted: nIds[s] is slot s partner (or 0xFFFFFFFFu if rejected)
    uint  nIds[SPAT_COUNT_MAX_GI];
    [unroll]
    for (uint i = 0u; i < SPAT_COUNT_MAX_GI; ++i) nIds[i] = 0xFFFFFFFFu;

    uint validCount = 0;

    [loop]
    for (uint s = 0u; s < SPAT_COUNT_MAX_GI; ++s)
    {
        const int2 delta   = SampleReuseDelta(launchIndex, s);
        const int2 partner = int2(launchIndex) + delta;

        //Out-of-screen: no wrap in screen space, drop this slot
        if (any(partner < int2(0, 0)) || any(partner >= int2(IMG_W, IMG_H)))
            continue;

        const uint bID = MapPixelID(dims, (uint2)partner);

        if (load_isEmitter(g_sample_current, bID)) continue;

        const uint   bInstID = load_instID(g_sample_current, bID);
        const uint   bPrimID = load_primID(g_sample_current, bID);
        const uint   bMatID  = GetMatIDFast(bInstID, bPrimID);
        const float2 bBary   = load_bary(g_sample_current, bID);
        const float3 bPos    = ReconstructPosition(bInstID, bPrimID, bBary);
        const float3 bN1s    = load_n1_s_with_instID(g_sample_current, bID, bInstID);

        if (PairRejected(myMatID, myPos, myN1s, bMatID, bPos, bN1s)) continue;

        const uint bM = load_M_gi(g_Reservoirs_current_gi, bID);
        if (bM == 0u) continue;

        nIds[s] = bID;
        ++validCount;
    }

    //Write header: validCount (my_Jc filled by shift pass)
    g_pathStateBuffer.Store(baseAddr, validCount);

    //Write nIDs at slot positions
    [unroll]
    for (uint k = 0u; k < SPAT_COUNT_MAX_GI; ++k)
    {
        g_pathStateBuffer.Store(gi_sel_slot_addr(pixelIdx, k), nIds[k]);
    }
}
