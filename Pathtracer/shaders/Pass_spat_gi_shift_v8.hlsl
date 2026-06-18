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
    /*uint sortKey;
    {
        const uint2 li = DispatchRaysIndex().xy;
        const uint  px = MapPixelID(uint2(IMG_W, IMG_H), int2(li));
        sortKey = g_pathStateBuffer.Load(sel_addr(px));
    }
    dx::MaybeReorderThread(sortKey, 2);*/

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);
    const uint   baseAddr    = sel_addr(pixelIdx);

    const uint validCount = g_pathStateBuffer.Load(baseAddr);
    if (validCount == 0u) return;

    //my vertex, persists across slots - rebuilt from the baked G-buffer
    const float3 myPos    = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, myPos, InitOrigin());

    //my Jc, env/miss uses Jc=1, shift preserves direction
    float my_Jc = 1.0f;
    {
        const uint myMatID = load_matID_res(g_Reservoirs_current, pixelIdx);
        if (myMatID != MATID_ENV_MISS)
        {
            //one pack1 fetch for both x2 and n2 (load_x2/load_n2_s each did
            //their own Load4 of the same record)
            const uint   myObjID = load_objID(g_Reservoirs_current, pixelIdx);
            const uint4  p1      = g_Reservoirs_current.Load4(addr_pack1(pixelIdx));
            const float3 my_x2   = ObjectToWorldPos(myObjID, asfloat(p1.xyz));
            const float3 my_n2s  = ObjectToWorldNrm(myObjID, UnpackNormal(p1.w));
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

        //partner reconnection payload in 4 grouped fetches (pack1 | pay |
        //v2 | objID) - was ~8 scattered single-field loads
        Reservoir pr = (Reservoir)0;
        loadReservoirPayload(g_Reservoirs_current, nID, pr);

        //shift my x1 to partner x2
        float  Jn = 0.0f;
        float3 c  = Reconnect(
            sv.x, sv.n_s, sv.o, sv.matID,
            sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat,
            pr.matID, pr.x2, pr.n2_s, pr.L2, pr.V2,
            pr.Kd, pr.Pr, pr.Pm, pr.eta,
            Jn);

        //bake visibility into the stored contribution. The ray is skipped
        //when the reconnection already evaluated to zero
        //(c == 0 <=> GetPHat(c) == 0: luminance weights positive, c >= 0)
        if (GetPHat(c) > 0.0f)
        {
            float vis;
            if (pr.matID == MATID_ENV_MISS)
            {
                //miss: synthesize a far endpoint along the stored sky direction
                const float3 md = normalize(pr.x2);
                vis = IsVisible(sv.x, sv.n_s, sv.x + md * RAY_TMAX_PLANET, -md) ? 1.0f : 0.0f;
            }
            else
            {
                vis = IsVisible(sv.x, sv.n_s, pr.x2, pr.n2_s) ? 1.0f : 0.0f;
            }
            c *= vis;
        }

        //slot layout, 12B F plus 4B Jn, target mag is GetPHat(F)
        g_pathStateBuffer.Store4(slotAddr + 4u,
            uint4(asuint(c), asuint(Jn)));
    }
}
