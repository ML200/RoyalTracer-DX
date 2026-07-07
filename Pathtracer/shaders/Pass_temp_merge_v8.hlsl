#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//TEMPORAL GI  (pass 3/3: MIS combine + reservoir update, NO rays)
//====================================
//Final stage of the split temporal reuse (mirrors spatial's select/shift/merge
//split the same way Pass_spmis_merge does). Reads the raw {c, Jn, cachedNew}
//the unified Pass_shift_v8 produced for job slots d=0 (forward) / d=1
//(reverse) — see the "UNIFIED SHIFT JOB" note in HashGridHash_v8.hlsli — does
//the PSS MIS combine + reservoir accept, and writes the result. Pure data: no
//TraceRay, no RayQuery, so it runs lean / high-occupancy like Pass_spmis_merge.
//
//SPM_w0 == TEMP_STATUS_DEAD (temp_gi's pessimistic first write, never
//overwritten to OK) means no candidate resolved this frame for this pixel —
//the reservoir is left exactly as raygen produced it, matching the old
//TemporalMergeBody's early-return behavior.
//
//FORWARD (d=0) always has a job when this pixel is OK (temp_gi guarantees a
//valid neighbour reservoir before ever writing OK), so its slotJ/slotC reads
//are unconditional. REVERSE (d=1) may have no job at all (my own sample
//invalid/unreusable) — Pass_shift_v8 never runs for a SP_UNDEF slot, so
//slotJ/slotC there can hold STALE data from a past frame; gate on the
//descriptor first, exactly like Pass_spmis_merge gates its draws.

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    if (g_pathStateBuffer.Load(SPM_w0(pixelIdx)) == TEMP_STATUS_DEAD)
        return;

    const uint permPk       = g_pathStateBuffer.Load(SPM_w1(pixelIdx));
    const int2 cand         = int2((int)(permPk & 0xFFFFu), (int)((permPk >> 16) & 0x7FFFu));
    const bool dual         = (permPk & 0x80000000u) != 0u;
    const uint tempPixelIdx = MapPixelID(dims, cand);

    const Reservoir rdi_r = loadReservoir(g_Reservoirs_last, tempPixelIdx);
    Reservoir       rdi   = loadReservoir(g_Reservoirs_current, pixelIdx);

    //Per-pixel sky observer for any DEFERRED env-replay finish below: Pass_shift_v8
    //kept the atmosphere stack out of its RT binary and persisted {partial, dir,
    //misPdf}; EnvTailFinish reads this observer (per-pixel approximation of the
    //offset bounce vertex — sky observer is very low-frequency over a neighbourhood).
    SetSkyObserver(load_x1(g_sample_current, pixelIdx) + sceneOriginWorld);

    //forward: guaranteed to exist (see header) — Jn's sign (preVisDead) is
    //irrelevant here, temporal never had that direct/replay dead-vs-
    //materialize distinction spatial's merge needs (c_f already carries the
    //right zero on any failure, walk or direct).
    const uint2  j0    = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, 0u));
    const float3 c0raw = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, 0u)));
    //DEFERRED env-replay (forward sample = rdi_r): slotJ = {PackNormal(dir), misPdf},
    //c0raw = partial → finish sky+sun here; cachedNew_f/Jn_f are the env const 1.
    float  cachedNew_f, Jn_f;
    float3 c_f;
    if (RcEnvReplay(rdi_r.rcInfo))
    {
        cachedNew_f = 1.0f; Jn_f = 1.0f;
        c_f = (GetPHat(c0raw) > 0.0f)
            ? EnvTailFinish(c0raw, UnpackNormal(j0.x), asfloat(j0.y)) : (float3)0.0f;
    }
    else
    {
        cachedNew_f = asfloat(j0.x);
        Jn_f        = abs(asfloat(j0.y));
        c_f         = c0raw;
    }

    //reverse: may not exist at all.
    float  Jn_r = 0.0f;
    float3 c_r  = (float3)0.0f;
    if (g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, 1u)) != SP_UNDEF)
    {
        const uint2  j1    = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, 1u));
        const float3 c1raw = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, 1u)));
        //DEFERRED env-replay (reverse sample = my own rdi): finish sky+sun here,
        //Jn_r is the env const 1. (cachedNew_r isn't used by the reverse term.)
        if (RcEnvReplay(rdi.rcInfo))
        {
            Jn_r = 1.0f;
            c_r  = (GetPHat(c1raw) > 0.0f)
                ? EnvTailFinish(c1raw, UnpackNormal(j1.x), asfloat(j1.y)) : (float3)0.0f;
        }
        else
        {
            Jn_r = abs(asfloat(j1.y));
            c_r  = c1raw;
        }
    }

    uint2 seed = GetSeed(pixelIdx, time, 12);
    seed.x = Hash32(seed.x);

    //====================================
    //MIS + RESERVOIR MERGE  (PSS weights)
    //====================================
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c = GetPHat(rdi.F) * visReuse_c;

    //geometric band (Jn/gBase only, pdf factors stay exact): clamp-scale for
    //base candidates, REJECT band for dual candidates (see Temporal_Merge header).
    float geomScale_f, geomScale_r;
    if (dual)
    {
        geomScale_f = RcGeomReject(Jn_f, rdi_r.gBase, spmis_jacThreshold) ? 0.0f : 1.0f;
        geomScale_r = RcGeomReject(Jn_r, rdi.gBase,   spmis_jacThreshold) ? 0.0f : 1.0f;
    }
    else
    {
        geomScale_f = RcGeomClampScale(Jn_f, rdi_r.gBase, temp_jacClamp);
        geomScale_r = RcGeomClampScale(Jn_r, rdi.gBase,   temp_jacClamp);
    }

    //shifted target densities (lum(c*vis) * Jn / cachedJac_src), band-scaled
    const float fwdT = (rdi_r.cachedJac > 0.0f)
        ? GetPHat(c_f) * Jn_f * geomScale_f / rdi_r.cachedJac : 0.0f;   //neighbour sample at me
    const float revT = (rdi.cachedJac > 0.0f)
        ? GetPHat(c_r) * Jn_r * geomScale_r / rdi.cachedJac   : 0.0f;   //my sample at neighbour
    const float n_n  = GetPHat(rdi_r.F) * ((rdi_r.W > 0.0f) ? 1.0f : 0.0f);

    //correlation reduction cCap, dup count D refreshes the chain (2026 paper).
    const float D        = saturate(gScratchPing[uint3(uint2(cand), 6)].x);
    const float effMcapF = (CORR_REDUCTION_OFF || rdi_r.matID == MATID_ENV_MISS)
                           ? (float)rs_tempMcap
                           : lerp((float)rs_tempMcap, 1.0f, pow(D, rs_corrReductionPow));

    const uint mcapU   = max(1u, rs_tempMcap);
    const uint effMcap = (uint)clamp(round(effMcapF), 1.0f, (float)mcapU);

    //Roughness-dependent neighbour mcap: LEGACY ONLY — it papered over the old
    //shift's specular temporal lag; the hybrid shift replays exactly that
    //transport, so the full mcap applies to every roughness.
    uint dynTempMcap = effMcap;
    if (!HYBRID_SHIFT_ON)
    {
        float myPr, myPm;
        load_prpm(g_sample_current, pixelIdx, myPr, myPm);
        const float roughScale    = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, myPr);
        const float tempMcapScale = lerp(1.0f, roughScale, myPm);   // dielectrics exempt
        dynTempMcap = (uint)clamp(round(effMcap * tempMcapScale), 1.0f, (float)mcapU);
    }

    const uint M_c = clamp(min(effMcap,     rdi.M),   1u, mcapU);
    const uint M_n = clamp(min(dynTempMcap, rdi_r.M), 1u, mcapU);

    //  canonical: m_c = M_c p_c / (M_c p_c + M_n revT)   [canonical at center vs shifted to neighbour]
    //  neighbour: m_n = M_n n_n / (M_n n_n + M_c fwdT)   [neighbour in its domain vs shifted to center]
    const float denom_c = M_c * p_c + M_n * revT;
    const float denom_n = M_n * n_n + M_c * fwdT;
    const float mis_c = (denom_c > EPSILON) ? (M_c * p_c / denom_c) : 1.0f;
    const float mis_n = (denom_n > EPSILON) ? (M_n * n_n / denom_n) : 0.0f;

    //resampling weights: canonical keeps its own-domain form; the neighbour's is
    //its shifted density at this pixel times its unbiased contribution weight.
    const float w_c = mis_c * p_c * rdi.W;
    const float w_n = mis_n * fwdT * rdi_r.W;

    //accept payload: re-anchored PSS contribution + new-side jacobian cache
    const float3 F_shifted = (cachedNew_f > 0.0f) ? (c_f * Jn_f / cachedNew_f) : (float3)0.0f;

    rdi.w_sum = w_c;

    float p_hat_final = p_c;
    bool  accepted    = false;
    if (UpdateReservoir(rdi, w_n, M_n, rdi_r, F_shifted, cachedNew_f, Jn_f, seed))
    {
        p_hat_final = GetPHat(F_shifted);
        accepted    = true;
    }

    //bounded UCW: an unclamped w_sum/p_hat spikes + feeds back every frame near
    //grazing/occluded surfaces (see FinalizeUCW). ucw_clampMax breaks it.
    rdi.W = FinalizeUCW(rdi.w_sum, p_hat_final, ucw_clampMax);

    if (accepted)
    {
        storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
    }
    else
    {
        //payload bits in memory are still raygen's - persist only W and M
        store_W(g_Reservoirs_current, pixelIdx, rdi.W);
        store_M(g_Reservoirs_current, pixelIdx, rdi.M);
    }
}
