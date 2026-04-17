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
static const uint GI_SEL_STRIDE      = 64u;
static const uint GI_SEL_SLOT_BASE   = 16u;
static const uint GI_SEL_SLOT_STRIDE = 16u;

uint gi_sel_addr(uint linearIdx) { return linearIdx * GI_SEL_STRIDE; }
uint gi_sel_slot_addr(uint linearIdx, uint slot)
{
    return gi_sel_addr(linearIdx) + GI_SEL_SLOT_BASE + slot * GI_SEL_SLOT_STRIDE;
}

[shader("raygeneration")]
void Pass_spat_gi_shift_v8()
{
    //Sort by validCount (0..3) so pixels with similar work pair up.
    uint sortKey;
    {
        const uint2 li = DispatchRaysIndex().xy;
        const uint  px = MapPixelID(uint2(IMG_W, IMG_H), int2(li));
        sortKey = g_pathStateBuffer.Load(gi_sel_addr(px));
    }
    dx::MaybeReorderThread(sortKey, 2);

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);
    const uint   baseAddr    = gi_sel_addr(pixelIdx);

    const uint validCount = g_pathStateBuffer.Load(baseAddr);
    if (validCount == 0u) return;

    //MY vertex, persists across slots
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(myInstID, myPrimID, myBary, InitOrigin());

    //MY Jc, partners read this from my scratch header for their canonical MIS
    {
        const Reservoir_GI myRdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
        const float my_Jc = ComputeJc(sv.x, myRdi.x2_gi, myRdi.n2_s_gi);
        g_pathStateBuffer.Store(baseAddr + 8u, asuint(my_Jc));
    }

    //One shift per valid slot
    [loop]
    for (uint s = 0u; s < SPAT_COUNT_MAX_GI; ++s)
    {
        const uint slotAddr = gi_sel_slot_addr(pixelIdx, s);
        const uint nID      = g_pathStateBuffer.Load(slotAddr);

        if (nID == 0xFFFFFFFFu) continue;

        //Partner reservoir
        const Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, nID);

        float3 rKd; float rPr, rPm;
        RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rKd, rPr, rPm);

        // MY x1 -> partner x2
        float  Jn = 0.0f;
        float3 c  = ReconnectGI(
            sv.x, sv.n_s, sv.o, sv.matID,
            sv.Kd, sv.Pr, sv.Pm,
            rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
            rKd, rPr, rPm, rdi_r.eta_gi,
            Jn);

        // Visibility — baked into the stored contribution
        {
            const float3 conn = rdi_r.x2_gi - sv.x;
            const float  cd   = length(conn);
            const float  vis  = (cd > EPSILON &&
                                 IsVisible(sv.x, sv.n_s, conn / cd, cd * 0.999f))
                                ? 1.0f : 0.0f;
            c *= vis;
        }

        // Pack (color dir + magnitude + Jn) into 12 bytes
        const float  F_mag  = GetPHat(c);
        const float3 F_norm = (F_mag > 1e-20f) ? c / F_mag : float3(0, 0, 0);
        const uint   F_pack = PackRGB9E5(F_norm);

        g_pathStateBuffer.Store3(slotAddr + 4u,
            uint3(F_pack, asuint(F_mag), asuint(Jn)));
    }
}
