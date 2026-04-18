#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI - Shift Pass
//
//  For each pair produced by the select pass, compute this pixels shift to
//  the paired partners x2 once and cache the result in scratch. The merge
//  pass then reads both sides' cached shifts without performing any
//  reconnection or visibility rays.
//
//  By the reuse-texture self-inversion property, if this pixel writes its
//  shift for slot s, the partner writes ITS shift for slot s in its own
//  scratch. Merge cross-references at the same slot index on both sides.
//─────────────────────────────────────────────────────────────────────────────

// Scratch layout mirrors Pass_spat_gi_select_v8.hlsl
static const uint SEL_STRIDE      = 56u;
static const uint SEL_SLOT_BASE   = 8u;
static const uint SEL_SLOT_STRIDE = 16u;

uint sel_addr(uint linearIdx) { return linearIdx * SEL_STRIDE; }
uint sel_slot_addr(uint linearIdx, uint slot)
{
    return sel_addr(linearIdx) + SEL_SLOT_BASE + slot * SEL_SLOT_STRIDE;
}

[shader("raygeneration")]
void Pass_spat_gi_shift_v8()
{
    //Sort by validCount (0..3) so pixels with similar work pair up.
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

    //MY vertex, persists across slots
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(myInstID, myPrimID, myBary, InitOrigin());

    //MY Jc, partners read this from my scratch header for their canonical MIS.
    //Env/miss samples (MATID_ENV_MISS) use Jc = 1 — the reconnection shift
    //preserves direction under reparameterization. Triangle-light samples
    //use the same geometric formula as vertex samples.
    {
        const uint myMatID = load_matID(g_Reservoirs_current, pixelIdx);
        float my_Jc = 1.0f;
        if (myMatID != MATID_ENV_MISS)
        {
            const uint   myObjID = load_objID(g_Reservoirs_current, pixelIdx);
            const float3 my_x2   = load_x2   (g_Reservoirs_current, pixelIdx, myObjID);
            const float3 my_n2s  = load_n2_s (g_Reservoirs_current, pixelIdx, myObjID);
            my_Jc = ComputeJc(sv.x, my_x2, my_n2s);
        }
        g_pathStateBuffer.Store(baseAddr + 4u, asuint(my_Jc));
    }

    //One shift per valid slot. Partner is loaded per-field (skip M / W / F /
    //F_mag / w_sum fields shift doesn't need) and pack1 is fetched once for
    //both x2 and n2_s.
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

        // Sentinel matIDs (env, triangle-light) have no BSDF at x2 — the
        // unified Reconnect branches skip these values. Don't index the
        // materials array with a sentinel.
        float3 rKd = 0.0f; float rPr = 0.0f, rPm = 0.0f;
        if (!IsSentinelMatID(p_matID))
            RefetchMaterial(p_matID, p_uv, rKd, rPr, rPm);

        // MY x1 -> partner x2
        float  Jn = 0.0f;
        float3 c  = Reconnect(
            sv.x, sv.n_s, sv.o, sv.matID,
            sv.Kd, sv.Pr, sv.Pm,
            p_matID, p_x2, p_n2s, p_L2, p_V2,
            rKd, rPr, rPm, p_eta,
            Jn);

        // Visibility — baked into the stored contribution. Env/miss samples
        // store x2 as a DIRECTION, so cast a fixed-far shadow ray in that
        // direction; triangle/vertex samples use the position-based form.
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

        const float  F_mag  = GetPHat(c);
        const float3 F_norm = (F_mag > 1e-20f) ? c / F_mag : float3(0, 0, 0);
        const uint   F_pack = PackRGB9E5(F_norm);

        g_pathStateBuffer.Store3(slotAddr + 4u,
            uint3(F_pack, asuint(F_mag), asuint(Jn)));
    }
}
