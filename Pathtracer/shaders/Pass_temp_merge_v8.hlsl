#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"
#include "Temporal_ReuseMath_v8.hlsli"

[numthreads(16, 16, 1)]
// Merge temporal radiance while carrying validation metadata.
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

    SetSkyObserver(load_x1(g_sample_current, pixelIdx) + sceneOriginWorld);

    const uint2  j0    = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, 0u));
    const float3 c0raw = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, 0u)));

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

    float  Jn_r = 0.0f;
    float3 c_r  = (float3)0.0f;
    if (g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, 1u)) != SP_UNDEF)
    {
        const uint2  j1    = g_pathStateBuffer.Load2(SPM_slotJ(pixelIdx, 1u));
        const float3 c1raw = asfloat(g_pathStateBuffer.Load3(SPM_slotC(pixelIdx, 1u)));

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

    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c = GetPHat(rdi.F) * visReuse_c;

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

    const float fwdT = (rdi_r.cachedJac > 0.0f)
        ? GetPHat(c_f) * Jn_f * geomScale_f / rdi_r.cachedJac : 0.0f;
    const float revT = (rdi.cachedJac > 0.0f)
        ? GetPHat(c_r) * Jn_r * geomScale_r / rdi.cachedJac   : 0.0f;
    const float n_n  = GetPHat(rdi_r.F) * ((rdi_r.W > 0.0f) ? 1.0f : 0.0f);

    const float D        = saturate(gScratchPing[uint3(uint2(cand), 6)].x);
    const uint mcapU   = max(1u, rs_tempMcap);
    const uint effMcap = TemporalConfidenceCap(rs_tempMcap, D, rs_corrReductionPow,
        CORR_REDUCTION_OFF || rdi_r.matID == MATID_ENV_MISS);

    uint dynTempMcap = effMcap;
    if (!HYBRID_SHIFT_ON)
    {
        float myPr, myPm;
        load_prpm(g_sample_current, pixelIdx, myPr, myPm);
        const float roughScale    = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, myPr);
        const float tempMcapScale = lerp(1.0f, roughScale, myPm);
        dynTempMcap = (uint)clamp(round(effMcap * tempMcapScale), 1.0f, (float)mcapU);
    }

    const uint M_c = clamp(min(effMcap,     rdi.M),   1u, mcapU);
    const uint M_n = clamp(min(dynTempMcap, rdi_r.M), 1u, mcapU);

    const float denom_c = M_c * p_c + M_n * revT;
    const float denom_n = M_n * n_n + M_c * fwdT;
    const float mis_c = (denom_c > EPSILON) ? (M_c * p_c / denom_c) : 1.0f;
    const float mis_n = (denom_n > EPSILON) ? (M_n * n_n / denom_n) : 0.0f;

    const float w_c = mis_c * p_c * rdi.W;
    const float w_n = mis_n * fwdT * rdi_r.W;

    const float3 F_shifted = (cachedNew_f > 0.0f) ? (c_f * Jn_f / cachedNew_f) : (float3)0.0f;

    rdi.w_sum = w_c;

    float p_hat_final = p_c;
    bool  accepted    = false;
    if (UpdateReservoir(rdi, w_n, M_n, rdi_r, F_shifted, cachedNew_f, Jn_f, seed))
    {
        p_hat_final = GetPHat(F_shifted);
        accepted    = true;
    }

    rdi.W = FinalizeUCW(rdi.w_sum, p_hat_final, ucw_clampMax);

    if (accepted)
    {
        storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
    }
    else
    {

        store_W(g_Reservoirs_current, pixelIdx, rdi.W);
        store_M(g_Reservoirs_current, pixelIdx, rdi.M);
    }
}
