#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - SELECT  (pass 1/3: cell search + selection + shift routing, NO rays)
//====================================
//First stage of the split SPMIS reuse (select -> shift -> merge), the structural
//analogue of the texture path's Pass_spat_gi_select. Owns the memory-heavy selection -
//hash-cell WRS search, the canonical partner pick, and the Ntilde inner-RIS draws - and
//writes the chosen candidate indices into the per-pixel scratch planes
//(g_pathStateBuffer, free in SPMIS mode). Casts NO rays and stores NO reservoir; the
//shift MAPPING (reconnection or replay) happens in shift, the WRS combine + output in
//merge. Layout + status codes: see HashGridHash_v8.hlsli (SPM_*).
//
//JOB ROUTING lives here too, for the unified Pass_shift_v8 (round-8): select
//decides per direction whether a shift job exists at all and writes a full
//job descriptor (see the "UNIFIED SHIFT JOB" note in HashGridHash_v8.hlsli),
//so every shift thread knows its one job from two loads. Slot d=0 is the
//canonical (reverse) job, d=1..Ntn are the draws (draw index = d-1):
//    slotS(d) = startPx                  receiver: partnerPx (canonical) /
//                                        pixelIdx (draws — always the centre)
//    slotZ(d) = SP_UNDEF                 dead (W<=0, unreusable, legacy gates,
//                                        or canonical ineligible)
//    slotZ(d) = resPx                    a job exists; Pass_shift_v8 produces
//                                        the raw {c,Jn,cachedNew} unconditionally
//                                        and the geometric reject band is
//                                        re-derived post-fence in the merges
//                                        for walked AND direct jobs alike (no
//                                        pre-visibility-ray hint any more —
//                                        see Pass_shift_v8.hlsl)
//    slotZ(0)|= SPM_KILLPH_BIT           legacy near-specular canonical
//                                        zeroing (hybrid OFF only)
//needWalk itself is NOT stored: Pass_shift_v8 re-derives it from the source
//reservoir's own rcInfo, which select already loaded here to decide whether a
//job exists in the first place — recomputing it is free, and it keeps the
//shift shader from needing ANY routing decision passed in besides "where."
//The gates are byte-identical to the ones shift used to run (same post-temporal
//reservoir data, barrier-fenced), and classification consumes NO RNG, so the
//realized selection is unchanged.
//
//2026-07 L2 rework (this pass was L2-throughput/latency bound on the scattered search):
//  * every probe is ONE 32B sector-aligned SP_SRCH record {cell|pos, conf|n} written by
//    sort — replaces the SP_HASH[npx] + 16B-record pair in two regions (2 dependent
//    sectors -> 1). Own record supplies cellCenter/myPos/myN (no instanceProps fan-out).
//  * TWO RNG STREAMS: probe offsets + partner/RIS index draws come from an ADDRESS
//    stream (seedO, hashed from GetSeed().y); WRS/RIS acceptance draws keep their own
//    DECISION stream (seedA, hashed from GetSeed().x, dim 6 as before). The old single
//    stream made probe i+1's ADDRESS depend on probe i's loaded data (the conditional
//    accept draw), serializing the search into 12 dependent L2 round-trips. With split
//    streams all addresses are precomputable, so the gathers are issued in BATCHES of 4
//    and their latency overlaps. NOTE: this changes the realized noise pattern (same
//    distributions, same convergence — like merge's dim-9 stream note), NOT the math.
//The selection math itself is identical to the monolithic Pass_spmis_reuse.

//cell-search shape now rides cbuffer slots 26-28 (editor sliders): initial
//radius spmis_searchR0, per-probe growth spmis_searchGrow, probe count
//spmis_searchIters (host-clamped 4..32; the 4-wide probe batches skip the
//tail when the count is not a multiple of 4).
#define SP_SEARCH_R0    spmis_searchR0
#define SP_SEARCH_GROW  spmis_searchGrow
#define SP_SEARCH_ITERS spmis_searchIters

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    //Pessimistic job clear, FIRST thing, unconditionally: the unified
    //Pass_shift_v8's spatial_shift Depth-slices are dispatched full-screen
    //by the host (the dispatch has no per-pixel/per-mode gate) and
    //check ONLY slotZ to decide whether a job exists, so every early-return
    //path below — spatial mode off, emitter, no resolved cell — must leave
    //"no job" behind rather than whatever a past frame (or, within THIS same
    //frame, temporal's own d=0/1 writes) last left there; g_pathStateBuffer
    //persists across frames and nothing else clears it. Harmless either way
    //for RENDER correctness (merge's own early-returns on these same statuses
    //never read the job slots at all) — this is purely about not wasting the
    //shift loop's work on stale descriptors.
    [unroll]
    for (uint dOff = 0u; dOff < SPMIS_TOTAL_ROLES; ++dOff)
        g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, dOff), SP_UNDEF);

    if (!SPMIS_SPATIAL_MODE) return;

    //entry gates read ONE bundled header word group (flags|matID|Kd|PrPm) instead of
    //three field-wise loads of the same 36B record.
    uint hdrFlags, hdrMatID; float cpr, cpm;
    load_SD_header(g_sample_current, pixelIdx, hdrFlags, hdrMatID, cpr, cpm);

    //emitter -> mark skip; merge writes the V2 sentinel, shift does nothing.
    if ((hdrFlags & SD_FLAG_EMITTER) != 0u)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_SKIP));
        return;
    }

    //own search record: cell + world pos/normal + center-cell confidence, exactly the
    //values the old path re-derived from SP_HASH + the G-buffer + instanceProps.
    const uint4 recA = g_spmisBuffer.Load4(SP_SRCH(pixelIdx));         // cell | pos
    const uint4 recB = g_spmisBuffer.Load4(SP_SRCH(pixelIdx) + 16u);   // conf | n
    const uint  cellCenter = recA.x;
    const uint  centerM    = load_M(g_Reservoirs_current, pixelIdx);

    //no center sample or no resolved cell -> passthrough (Pass_spmis_passthrough
    //casts the one deferred visibility ray, merge writes the shadowed canonical).
    if (centerM == 0u || cellCenter == SP_UNDEF)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_PASS));
        return;
    }

    //Low-roughness glass, LEGACY ONLY: the near-specular transmit/refract lobe is strongly
    //view-dependent, so with the pin frozen at x2 spatial resampling across neighbouring
    //pixels spikes p_hat -> fireflies. Skip SPMIS for it (-> passthrough: shadowed canonical).
    //Under the HYBRID shift the pin sits BEYOND the specular chain (random replay walks
    //through it), so glass pixels take full spatial reuse like everything else.
    if (!HYBRID_SHIFT_ON &&
        cpr < SPMIS_GLASS_ROUGHNESS_MIN &&
        LoadKd_w(hdrMatID) < 1.0f - EPSILON)
    {
        g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(0u, SPM_STATUS_PASS));
        return;
    }

    const float3 myPos  = asfloat(recA.yzw);
    const float3 myN    = asfloat(recB.yzw);
    const float3 camPos = InitOrigin();

    //two streams: seedA = decisions (accepts), seedO = addresses (offsets + index draws)
    uint2 seed  = GetSeed(pixelIdx, time, 6);
    uint  seedA = Hash32(seed.x);
    uint  seedO = Hash32(seed.y);

    const uint Ntn = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);

    //==== CELL SEARCH - WRS, center pre-seeded; batched fused-record probes ====
    float weight_sum   = asfloat(recB.x);   // own record conf == SP_CONF[cellCenter]
    uint  selectedCell = cellCenter;
    float radius       = SP_SEARCH_R0;
    const float planeThresh = spmis_planeDist * length(myPos - camPos);

    [loop]
    for (uint b = 0u; b < SP_SEARCH_ITERS; b += 4u)
    {
        //phase 1: draw offsets from the address stream, issue all 4 record loads —
        //independent addresses, so the gathers pipeline instead of serializing.
        uint4 rA[4];
        uint4 rB[4];
        [unroll]
        for (uint j = 0u; j < 4u; ++j)
        {
            rA[j] = uint4(SP_UNDEF, 0u, 0u, 0u);
            rB[j] = uint4(0u, 0u, 0u, 0u);
            if (b + j >= SP_SEARCH_ITERS) continue;

            int2 off = int2(round(radius * (RandomFloatPCG(seedO) * 2.0f - 1.0f)),
                            round(radius * (RandomFloatPCG(seedO) * 2.0f - 1.0f)));
            radius *= SP_SEARCH_GROW;
            int2 nc  = int2(tid.xy) + off;
            if (nc.x < 0) nc.x = -nc.x; else if (nc.x >= (int)IMG_W) nc.x = 2 * (int)IMG_W - nc.x - 1;
            if (nc.y < 0) nc.y = -nc.y; else if (nc.y >= (int)IMG_H) nc.y = 2 * (int)IMG_H - nc.y - 1;

            const uint npx = MapPixelID(dims, nc);   // bounds-checked, 0xFFFFFFFF if out
            if (npx != 0xFFFFFFFFu)
            {
                rA[j] = g_spmisBuffer.Load4(SP_SRCH(npx));         // cell | pos
                rB[j] = g_spmisBuffer.Load4(SP_SRCH(npx) + 16u);   // conf | n
            }
        }

        //phase 2: serial WRS on the batch, decision stream only (no addresses depend
        //on these draws). Reject order matches the old path: cell key -> normal cone
        //-> plane distance; rejected probes consume NO accept draw (as before).
        [unroll]
        for (uint j2 = 0u; j2 < 4u; ++j2)
        {
            const uint ncell = rA[j2].x;
            if (ncell == SP_UNDEF || ncell == cellCenter) continue;   // cheap reject key

            //normal cone (uniform branch; default spmis_normalSimCos == -1 skips it —
            //the neighbour normal now rides the record, no extra fetch either way).
            if (spmis_normalSimCos > -1.0f)
            {
                const float3 nN = asfloat(rB[j2].yzw);
                if (dot(myN, nN) <= spmis_normalSimCos) continue;
            }
            //plane-distance reject (bit-identical per-pixel position the G-buffer path used).
            const float3 nPos = asfloat(rA[j2].yzw);
            if (abs(dot(nPos - myPos, myN)) > planeThresh) continue;

            const float w = asfloat(rB[j2].x);   // cell confidence sum (== SP_CONF[ncell])
            weight_sum += w;
            if (RandomFloatPCG(seedA) < w / weight_sum) selectedCell = ncell;
        }
    }
    const uint reuseCell = selectedCell;

    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF (one fetch)
    const uint  cellBase  = agg.w;          // OFF
    const float pixCount  = (float)agg.x;   // PIXCNT
    const uint  nzCount   = agg.y;          // NZ
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;   // CONF

    //==== canonical partner pick (Nc=1 uniform over the cell); shift reconnects + rays ====
    uint partnerPx = SP_UNDEF;
    if (neighbors_conf_sum > EPSILON && pixCount > 0.0f)
    {
        uint k = (uint)(pixCount * RandomFloatPCG(seedO));   // index draw -> address stream
        if (k >= (uint)pixCount) k = (uint)pixCount - 1u;
        partnerPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + k));
    }

    //canonical job (slot d=0): full ELIGIBILITY check up front (mirrors the
    //old shift's direct-path re-derivation exactly) — an ineligible canonical
    //gets NO job at all, so merge's mis_c degenerates to 1 without Pass_shift_v8
    //ever running for this slot. Eligible always gets a job; Pass_shift_v8 bands
    //post-fence in Pass_spmis_merge regardless of walk/direct, so there's no
    //routing split here any more — only the legacy killPh gate remains.
    uint canonRes = SP_UNDEF;
    const bool partnerOk = (partnerPx != SP_UNDEF) && neighbors_conf_sum > EPSILON && pixCount > 0.0f;
    if (partnerOk)
    {
        const Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);
        const bool  canonReusable = !HYBRID_SHIFT_ON || RcReusable(rdi.rcInfo);
        const float p_c = GetPHat(rdi.F) * ((rdi.W > 0.0f) ? 1.0f : 0.0f);
        if (canonReusable && p_c > EPSILON)
        {
            canonRes = pixelIdx;
            //legacy near-specular canonical zeroing (hybrid OFF only —
            //only REJECTION is unbiased there), pre-ray like the old inline path
            if (!HYBRID_SHIFT_ON &&
                rdi.matID != MATID_LIGHT_TRI && rdi.matID != MATID_ENV_MISS &&
                !IsVolumeVertex(rdi.matID) && rdi.Pr < rs_reconnectRoughnessMin)
                canonRes |= SPM_KILLPH_BIT;
        }
    }

    g_pathStateBuffer.Store(SPM_w0(pixelIdx), SPM_packHdr(reuseCell, SPM_STATUS_NORM));
    //startBufLast is always 0 (g_sample_current) for spatial, so the raw
    //partnerPx (top bit clear) is already the correctly-encoded startPx word.
    g_pathStateBuffer.Store(SPM_slotS(pixelIdx, 0u), partnerPx & 0x7FFFFFFFu);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), canonRes);

    //==== Ntilde inner-RIS selections (each prop. UCW*target*M, P=1/UCW) ====
    //Every slot in [0,Ntn) is written (valid zPx or SP_UNDEF) so merge can skip
    //unmaterialized draws without a separate clear; merge recomputes Ntn identically.
    //Index draws (k) come from the address stream so each 4-wide chunk of SORTEDW taps
    //is issued together; acceptance stays on the decision stream.
    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        uint  outZ = SP_UNDEF;
        float outP = 0.0f;
        if (nzCount > 0u)
        {
            float ris_wsum = 0.0f, ris_selTF = 0.0f;
            uint  ris_selK = SP_UNDEF;
            [loop]
            for (uint r0 = 0u; r0 < spmis_risN; r0 += 4u)
            {
                uint  kArr[4];
                float tfArr[4];
                [unroll]
                for (uint j = 0u; j < 4u; ++j)
                {
                    kArr[j] = 0u; tfArr[j] = 0.0f;
                    if (r0 + j >= spmis_risN) continue;
                    uint k = (uint)((float)nzCount * RandomFloatPCG(seedO));
                    if (k >= nzCount) k = nzCount - 1u;
                    kArr[j]  = k;
                    tfArr[j] = asfloat(g_spmisBuffer.Load(SP_A(SP_SORTEDW, cellBase + k)));
                }
                [unroll]
                for (uint j2 = 0u; j2 < 4u; ++j2)
                {
                    if (r0 + j2 >= spmis_risN) continue;
                    const float w = tfArr[j2] * (float)nzCount / (float)spmis_risN;
                    ris_wsum += w;
                    if (ris_wsum > 0.0f && RandomFloatPCG(seedA) < w / ris_wsum)
                    { ris_selK = kArr[j2]; ris_selTF = tfArr[j2]; }
                }
            }
            if (ris_selK != SP_UNDEF && ris_selTF > 0.0f)
            {
                const uint zPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + ris_selK));
                if (zPx != SP_UNDEF) { outZ = zPx; outP = ris_selTF / ris_wsum; }
            }
        }

        //==== job routing (was the top of shift's draw loop, gate-for-gate) ====
        //dead-reservoir + mode-specific reusability gates. NO roughness or
        //distance gating under the hybrid shift — full reuse across all
        //roughnesses; the shifted BSDF magnitude self-gates and the geometric
        //band in Pass_spmis_merge bounds the singularities post-fence, for
        //walked AND direct draws alike (draws have no killPh — only the
        //legacy branch below can kill a draw's job outright).
        uint outRes = SP_UNDEF;
        if (outZ != SP_UNDEF)
        {
            const uint  zRcInfo = load_rcInfo(g_Reservoirs_current, outZ);
            const float zW      = load_W(g_Reservoirs_current, outZ);
            bool dead = (zW <= 0.0f);
            if (!dead)
            {
                if (HYBRID_SHIFT_ON)
                {
                    if (!RcReusable(zRcInfo))
                        dead = true;
                }
                else if (RcReplayLen(zRcInfo) > 0u)
                {
                    dead = true;   //stale replay candidate after a hybrid-off toggle
                }
                else
                {
                    //legacy near-specular reconnection-vertex reject (hybrid OFF)
                    const uint zMat = load_matID_res(g_Reservoirs_current, outZ);
                    if (zMat != MATID_LIGHT_TRI && zMat != MATID_ENV_MISS &&
                        !IsVolumeVertex(zMat))
                    {
                        float zEta, zPr, zPm;
                        UnpackEtaPrPm(g_Reservoirs_current.Load(addr_pay(outZ) + 8u),
                                      zEta, zPr, zPm);
                        if (zPr < rs_reconnectRoughnessMin)
                            dead = true;
                    }
                }
            }
            if (!dead)
                outRes = outZ;
        }

        //draws' receiver is ALWAYS the centre pixel itself (startBufLast=0).
        g_pathStateBuffer.Store(SPM_slotS(pixelIdx, d + 1u), pixelIdx);
        g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d + 1u), outRes);
        g_pathStateBuffer.Store(SPM_slotP(pixelIdx, d + 1u), asuint(outP));
    }
}
