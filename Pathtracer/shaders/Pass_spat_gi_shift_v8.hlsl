#include "Includes_v8.hlsli"

//====================================
//SPATIAL GI SHIFT PASS
//====================================
//caches each pixel's shift to partner x2, merge reads both sides without further rays

//scratch layout mirrors Pass_spat_gi_select_v8.hlsl
//slot is [nID(4) | F(12) | Jn(4) | Jc(4)], Jc duplicated per slot so the
//merge pass picks up partner Jc in the same sector as partner shift
static const uint SEL_STRIDE      = 4u + SPAT_COUNT_MAX * 24u;
static const uint SEL_SLOT_BASE   = 4u;
static const uint SEL_SLOT_STRIDE = 24u;

uint sel_addr(uint linearIdx) { return linearIdx * SEL_STRIDE; }
uint sel_slot_addr(uint linearIdx, uint slot)
{
    return sel_addr(linearIdx) + SEL_SLOT_BASE + slot * SEL_SLOT_STRIDE;
}

//====================================
//SHIFT PASS ENTRY
//====================================
[shader("raygeneration")]
void Pass_spat_gi_shift_v8()
{
    //sort by validCount so similar work pixels pair up
    uint sortKey;
    {
        const uint2 li = DispatchRaysIndex().xy;
        const uint  px = MapPixelID(uint2(IMG_W, IMG_H), int2(li));
        sortKey = g_pathStateBuffer.Load(sel_addr(px));
    }
    dx::MaybeReorderThread(sortKey, 2);

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);
    const uint   baseAddr    = sel_addr(pixelIdx);

    const uint validCount = g_pathStateBuffer.Load(baseAddr);
    if (validCount == 0u) return;

    //my vertex, persists across slots
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(myInstID, myPrimID, myBary, InitOrigin());

    //my Jc, env/miss uses Jc=1, shift preserves direction
    float my_Jc = 1.0f;
    {
        const uint myMatID = load_matID(g_Reservoirs_current, pixelIdx);
        if (myMatID != MATID_ENV_MISS)
        {
            const uint   myObjID = load_objID(g_Reservoirs_current, pixelIdx);
            const float3 my_x2   = load_x2   (g_Reservoirs_current, pixelIdx, myObjID);
            const float3 my_n2s  = load_n2_s (g_Reservoirs_current, pixelIdx, myObjID);
            my_Jc = ComputeJc(sv.x, my_x2, my_n2s);
        }
    }

    //my_Jc lives in every slot so partner-side reads in the merge gather get
    //it in the same sector as the partner shift load
    const uint my_Jc_pk = asuint(my_Jc);
    [unroll]
    for (uint jcS = 0u; jcS < SPAT_COUNT_MAX; ++jcS)
    {
        g_pathStateBuffer.Store(sel_slot_addr(pixelIdx, jcS) + 20u, my_Jc_pk);
    }

    //one shift per valid slot, partner loaded per field, pack1 fetched once
    [loop]
    for (uint s = 0u; s < SPAT_COUNT_MAX; ++s)
    {
        const uint slotAddr = sel_slot_addr(pixelIdx, s);
        const uint nID      = g_pathStateBuffer.Load(slotAddr);
        if (nID == 0xFFFFFFFFu) continue;

        const uint   p_objID = load_objID(g_Reservoirs_current, nID);
        const uint   p_matID = load_matID(g_Reservoirs_current, nID);
        const uint4  pack1   = g_Reservoirs_current.Load4(addr_pack1(nID));
        const float3 p_x2    = ObjectToWorldPos(p_objID, asfloat(pack1.xyz));
        const float3 p_n2s   = ObjectToWorldNrm(p_objID, UnpackNormal(pack1.w));
        const float3 p_L2    = load_L2(g_Reservoirs_current, nID);
        const float3 p_V2    = load_V2(g_Reservoirs_current, nID);
        const float2 p_uv    = load_uv_res(g_Reservoirs_current, nID);
        const float  p_eta   = load_eta(g_Reservoirs_current, nID);

        //sentinel matIDs have no BSDF at x2, skip material load
        float3 rKd = 0.0f; float rPr = 0.0f, rPm = 0.0f;
        if (!IsSentinelMatID(p_matID))
            RefetchMaterial(p_matID, p_uv, rKd, rPr, rPm);

        //shift my x1 to partner x2
        float  Jn = 0.0f;
        float3 c  = Reconnect(
            sv.x, sv.n_s, sv.o, sv.matID,
            sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat,
            p_matID, p_x2, p_n2s, p_L2, p_V2,
            rKd, rPr, rPm, p_eta,
            Jn);

        //bake visibility into the stored contribution
        {
            float vis;
            if (p_matID == MATID_ENV_MISS)
            {
                vis = IsVisible(sv.x, sv.n_s, normalize(p_x2), 10000.0f) ? 1.0f : 0.0f;
            }
            else
            {
                const float3 conn = p_x2 - sv.x;
                const float  cd   = length(conn);
                vis = (cd > EPSILON &&
                       IsVisible(sv.x, sv.n_s, conn / cd, cd * 0.999f))
                      ? 1.0f : 0.0f;
            }
            c *= vis;
        }

        //slot layout, 12B F plus 4B Jn, target mag is GetPHat(F)
        g_pathStateBuffer.Store4(slotAddr + 4u,
            uint4(asuint(c), asuint(Jn)));
    }
}
