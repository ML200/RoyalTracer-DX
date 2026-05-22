#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPATIAL GI NEIGHBOR SELECTION
//====================================
//paired reuse via self inverting textures, partner at pixel+delta sees inverse delta

Texture2D<int2> g_reuseTexture0 : register(t19);
Texture2D<int2> g_reuseTexture1 : register(t20);
Texture2D<int2> g_reuseTexture2 : register(t21);

//====================================
//SCRATCH LAYOUT
//====================================
//4B header (validCount) plus SPAT_COUNT_MAX*24B per slot, M_sum recomputed in
//merge pass. Each slot is [nID(4) | F(12) | Jn(4) | Jc(4)]. Jc lives in every
//slot so the merge pass picks up partner Jc inside the same 32B sector as the
//partner shift load, killing the second scattered fetch we used to do at
//sel_addr(nID)+4.
static const uint SEL_STRIDE      = 4u + SPAT_COUNT_MAX * 24u;
static const uint SEL_SLOT_BASE   = 4u;
static const uint SEL_SLOT_STRIDE = 24u;

uint sel_addr(uint linearIdx) { return linearIdx * SEL_STRIDE; }
uint sel_slot_addr(uint linearIdx, uint slot)
{
    return sel_addr(linearIdx) + SEL_SLOT_BASE + slot * SEL_SLOT_STRIDE;
}

//====================================
//REUSE DELTA SAMPLING
//====================================
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

    //shift origin, wrap into tileable domain
    int2 cLookup = int2((launchIndex + offset) % uint2(texSize));

    //lookup transforms
    if (flags & 4u) cLookup = cLookup.yx;
    if (flags & 1u) cLookup.x = texSize.x - 1 - cLookup.x;
    if (flags & 2u) cLookup.y = texSize.y - 1 - cLookup.y;

    int2 d;
    if (slot == 0u)      d = g_reuseTexture0.Load(int3(cLookup, 0));
    else if (slot == 1u) d = g_reuseTexture1.Load(int3(cLookup, 0));
    else                 d = g_reuseTexture2.Load(int3(cLookup, 0));

    //inverse transforms on returned delta
    if (flags & 1u) d.x = -d.x;
    if (flags & 2u) d.y = -d.y;
    if (flags & 4u) d = d.yx;

    return d;
}

//====================================
//PAIR REJECTION
//====================================
//symmetric material, normal, distance. distThresh is rs_rejDistance scaled by
//camera-to-surface length so the slab grows with world-space pixel footprint;
//fixed threshold rejects all far neighbors on large scenes
bool PairRejected(uint aMat, float3 aPos, float3 aN,
                  uint bMat, float3 bPos, float3 bN,
                  float distThresh)
{
    if (aMat != bMat) return true;
    if (RejectNormal(aN, bN, rs_rejNormalDot)) return true;
    if (RejectDistance(aPos, bPos, aN, distThresh)) return true;
    if (RejectDistance(bPos, aPos, bN, distThresh)) return true;
    return false;
}

//====================================
//SELECT PASS ENTRY
//====================================
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const uint2  launchIndex = tid.xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    const uint baseAddr = sel_addr(pixelIdx);

    //emitter or spatial off, no neighbors
    if (load_isEmitter(g_sample_current, pixelIdx) || !(rs_flags & 8u))
    {
        g_pathStateBuffer.Store(baseAddr, 0u);
        return;
    }

    //empty reservoir, skip
    const uint myM = load_M(g_Reservoirs_current, pixelIdx);
    if (myM == 0u)
    {
        g_pathStateBuffer.Store(baseAddr, 0u);
        return;
    }

    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myMatID  = load_matID(g_sample_current, pixelIdx);
    const float3 myPos    = load_x1(g_sample_current, pixelIdx);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);

    //slab thickness scales with camera distance so pixel footprint at depth still passes
    const float distThresh = rs_rejDistance * length(myPos - InitOrigin());

    //nIds[s] is partner for slot s, 0xFFFFFFFFu means rejected
    uint  nIds[SPAT_COUNT_MAX];
    [unroll]
    for (uint i = 0u; i < SPAT_COUNT_MAX; ++i) nIds[i] = 0xFFFFFFFFu;

    uint validCount = 0;

    [loop]
    for (uint s = 0u; s < SPAT_COUNT_MAX; ++s)
    {
        const int2 delta   = SampleReuseDelta(launchIndex, s);
        const int2 partner = int2(launchIndex) + delta;

        //screen-space bounds, no wrap
        if (any(partner < int2(0, 0)) || any(partner >= int2(IMG_W, IMG_H)))
            continue;

        const uint bID = MapPixelID(dims, (uint2)partner);

        if (load_isEmitter(g_sample_current, bID)) continue;

        const uint   bInstID = load_instID(g_sample_current, bID);
        const uint   bMatID  = load_matID(g_sample_current, bID);
        const float3 bPos    = load_x1(g_sample_current, bID);
        const float3 bN1s    = load_n1_s_with_instID(g_sample_current, bID, bInstID);

        if (PairRejected(myMatID, myPos, myN1s, bMatID, bPos, bN1s, distThresh)) continue;

        const uint bM = load_M(g_Reservoirs_current, bID);
        if (bM == 0u) continue;

        nIds[s] = bID;
        ++validCount;
    }

    //header validCount, my_Jc is filled by the shift pass
    g_pathStateBuffer.Store(baseAddr, validCount);

    [unroll]
    for (uint k = 0u; k < SPAT_COUNT_MAX; ++k)
    {
        g_pathStateBuffer.Store(sel_slot_addr(pixelIdx, k), nIds[k]);
    }
}
