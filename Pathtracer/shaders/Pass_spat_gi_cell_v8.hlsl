#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//STOCHASTIC PAIRWISE-MIS CELL SPATIAL REUSE  (Hedstrom et al. 2026 - full §5)
//====================================
//A/B alternative to the texture-paired select/shift/_v8_1 path, gated by
//RS_FLAG_CELL_SPATIAL (rs_flags 0x10). When the bit is set the texture passes
//no-op and this pair of passes (Pass_spat_gi_cellbuild_v8 then this) owns 100%
//of spatial reuse; when clear, both cell passes return immediately.
//
//This is the FULL paper, including §5.1's cell search:
//  1. Pass_spat_gi_cellbuild_v8 builds tile-local cells -> per-pixel records
//     (cellKey, confSum, c_i, w_i) in the path-state scratch buffer.
//  2. (here) §5.1 CELL SEARCH: a single growing-radius WRS pass selects a
//     neighbour CELL weighted by its confidence sum (skips cells whose key
//     differs from ours), reaching outside disocclusions as the radius grows.
//  3. Gather the chosen cell (the 8x8 screen block around the picked pixel,
//     filtered to our cellKey) -> the pool of M<=64 similar candidates.
//  4. Stochastic pairwise MIS over the pool: §4.3 confidence scaling, canonical
//     weight Eq.18 (Nc=1), Ntilde non-canonical draws prop. to Eq.17 weight with
//     the (1/Ntilde)(1/P_s) correction, reusing PairwiseMIS_Neighbor_Spat.
//Output contract mirrors Pass_spat_gi_v8_1: gScratchPing[.,2]=F*W +
//storeReservoir(g_Reservoirs_last,...), neighbours read from g_Reservoirs_current.

static const uint  CELL_REC      = 20u;            // per-pixel record stride (cellbuild)
static const float SEARCH_GROWTH = 1.25f;          // §5.1 radius growth per iteration
static const uint  MAXCELL       = 64u;            // 8x8 tile == max cell size
//SOFT normal cone for cell membership (query-relative, no hard quantization
//boundary -> no edge grid where normals vary fast). ~25 deg.
static const float CELL_COHERENCE_COS = 0.9f;

uint cell_addr(uint px) { return px * CELL_REC; }

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!CELL_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    //emitter: mirror Pass_spat_gi_v8_1's early-out exactly (V2 sentinel only)
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        g_Reservoirs_last.Store(addr_v2(pixelIdx), PROBE_DI_NORMAL_ZERO_CODE);
        return;
    }

    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);
    if (rdi.M == 0u)
    {
        const float W = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        gScratchPing[uint3(tid.xy, 2)] = float4(rdi.F * W, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    //cell key = surface (instID) + a query-relative normal cone (tested below),
    //read straight from the G-buffer so it matches cellbuild's record exactly.
    const uint   myInst = load_instID(g_sample_current, pixelIdx);
    const float3 myN    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInst);

    const float cap    = (float)rs_cellMcap;
    const uint  Ntilde = max(rs_cellN, 1u);

    const float M_c        = min(cap, (float)rdi.M);
    rdi.M = (uint)M_c;
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;
    float3      contrib_final = rdi.F * visReuse_c;

    uint2 seed = GetSeed(pixelIdx, time, 5);

    //====================================
    //§5.1 CELL SEARCH (single growing-radius WRS pass)
    //====================================
    //Sample one pixel per iteration in the current radius, fetch its cell, skip
    //cells whose key differs from ours, and WRS-select weighted by the cell's
    //confidence sum. Grow the radius each iteration to reach past disocclusions.
    int2 chosen = int2(tid.xy);
    bool found  = false;            // did the search actually select a cell?
    if (myInst != 0xFFFFFFFFu)
    {
        float wsum  = 0.0f;
        float r     = max((float)rs_cellRadius, 1.0f);
        const uint iters = clamp(rs_cellSearchIters, 1u, 32u);
        for (uint i = 0u; i < iters; ++i)
        {
            const float u1 = RandomFloatSingle(seed.x);
            const float u2 = RandomFloatSingle(seed.x);
            const float rr = r * sqrt(u1);
            const float a  = 6.2831853f * u2;
            const int2  q  = int2(tid.xy) + int2(round(float2(rr * cos(a), rr * sin(a))));
            r *= SEARCH_GROWTH;

            if (q.x < 0 || q.y < 0 || q.x >= (int)IMG_W || q.y >= (int)IMG_H) continue;
            const uint  qpx  = MapPixelID(dims, (uint2)q);
            const uint4 qrec = g_pathStateBuffer.Load4(cell_addr(qpx));   // inst, normalPk, confSum, c_i
            if (qrec.x != myInst) continue;                              // same surface
            if (dot(myN, UnpackNormal(qrec.y)) < CELL_COHERENCE_COS) continue;  // soft cone
            const float cw = asfloat(qrec.z);
            if (cw <= 0.0f) continue;
            wsum += cw;
            if (RandomFloatSingle(seed.x) < cw / wsum) { chosen = q; found = true; }
        }
    }

    //§5.1: no cell found (common on THIN objects - the random disk samples miss the
    //few-px sliver). Take LESS reuse, exactly as the paper does: canonical-only
    //passthrough. NEVER reuse from a fixed home tile - that revived the 8x8 grid.
    if (!found)
    {
        const float W = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        gScratchPing[uint3(tid.xy, 2)] = float4(rdi.F * W, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    //====================================
    //GATHER THE CHOSEN CELL  (8x8 window CENTERED on 'chosen', filtered to surface + cone)
    //====================================
    //Center the window on 'chosen' (translation invariant) instead of snapping to a
    //fixed 8-aligned screen block: a fixed block quantizes the pool to a screen grid
    //(the 8x8 grid); a centered window slides continuously so adjacent pixels' pools
    //overlap and there is no boundary for a grid to align to. All three consumers
    //(gather / canonical pick / non-canonical draw) read this one tileOrigin, so the
    //candW[] slot->pixel mapping stays consistent (no bias).
    const int2 tileOrigin = chosen - int2(4, 4);   // 8x8 window centered on chosen
    float candW[MAXCELL];                     // selection weight per tile-local slot; <0 = not in cell
    float cSigmaRaw = 0.0f;
    float Wsel_nz   = 0.0f;
    uint  M_nc      = 0u;

    [loop]
    for (uint li = 0u; li < MAXCELL; ++li)
    {
        candW[li] = -1.0f;
        const int2 sc = tileOrigin + int2(li & 7u, li >> 3u);
        if (sc.x < 0 || sc.y < 0 || sc.x >= (int)IMG_W || sc.y >= (int)IMG_H) continue;
        const uint spx = MapPixelID(dims, (uint2)sc);
        if (spx == pixelIdx) continue;        // self is the canonical, handled separately
        const uint4 rec = g_pathStateBuffer.Load4(cell_addr(spx));   // inst, normalPk, confSum, c_i
        if (rec.x != myInst) continue;                               // same surface
        if (dot(myN, UnpackNormal(rec.y)) < CELL_COHERENCE_COS) continue;   // soft cone
        const float w_i = asfloat(g_pathStateBuffer.Load(cell_addr(spx) + 16u));
        candW[li] = w_i;                      // 0 allowed (in cell, no sample)
        cSigmaRaw += asfloat(rec.w);          // c_i
        if (w_i > 0.0f) Wsel_nz += w_i;
        ++M_nc;
    }

    const uint  M_cell = M_nc + 1u;           // incl. self == paper's M
    const float s      = (M_cell > 0u) ? min(1.0f, (float)Ntilde / (float)M_cell) : 1.0f;  // §4.3
    const float cSigma = s * cSigmaRaw;
    const float M_sum  = M_c + cSigma;

    //my canonical surface vertex + geometric Jacobian, shared by both shifts
    const float3        camPos = InitOrigin();
    const float3        myPos  = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv_me  = BuildVertex(g_sample_current, pixelIdx, myPos, camPos);
    const float my_Jc = (rdi.matID == MATID_ENV_MISS) ? 1.0f
                                                       : ComputeJc(myPos, rdi.x2, rdi.n2_s);

    //====================================
    //CANONICAL MIS WEIGHT  (Eq.18, Nc=1 uniform over the cell)
    //====================================
    float mis_c = M_c / max(M_sum, 1.0f);
    if (p_c > 0.0f && M_nc > 0u)
    {
        uint pick = (uint)(RandomFloatSingle(seed.x) * (float)M_nc);
        pick = min(pick, M_nc - 1u);

        uint seen = 0u, zli = 0xFFFFFFFFu;
        [loop]
        for (uint li = 0u; li < MAXCELL; ++li)
        {
            if (candW[li] < 0.0f) continue;
            if (seen == pick) { zli = li; break; }
            ++seen;
        }
        if (zli != 0xFFFFFFFFu)
        {
            const int2  sc   = tileOrigin + int2(zli & 7u, zli >> 3u);
            const uint  zpPx = MapPixelID(dims, (uint2)sc);
            const float c_zp = s * asfloat(g_pathStateBuffer.Load(cell_addr(zpPx) + 12u));  // §4.3 scaled c_i

            const uint   zpInst = load_instID(g_sample_current, zpPx);
            const float3 zpPos  = load_x1(g_sample_current, zpPx);
            const SurfaceVertex sv_zp = BuildVertex(g_sample_current, zpPx, zpPos, camPos);

            float Jn_rev = 0.0f;
            float3 cr = Reconnect(
                sv_zp.x, sv_zp.n_s, sv_zp.o, sv_zp.matID,
                sv_zp.Kd, sv_zp.Pr, sv_zp.Pm, sv_zp.etai, sv_zp.etat,
                rdi.matID, rdi.x2, rdi.n2_s, rdi.L2, rdi.V2,
                rdi.Kd, rdi.Pr, rdi.Pm, rdi.eta, Jn_rev);

            float ph = GetPHat(cr), vis = 0.0f;
            if (ph > 0.0f)
            {
                if (rdi.matID == MATID_ENV_MISS)
                    vis = IsVisibleEnvMiss(sv_zp.x, sv_zp.n_s, normalize(rdi.x2), RAY_TMAX_PLANET, zpInst) ? 1.0f : 0.0f;
                else { const float3 cn = rdi.x2 - sv_zp.x; const float cd = length(cn);
                       vis = (cd > EPSILON && IsVisible(sv_zp.x, sv_zp.n_s, cn / cd, cd * 0.999f)) ? 1.0f : 0.0f; }
            }
            const float pHatBackZ = ph * JacobianRatio(Jn_rev, my_Jc) * vis;   // p_hat<-z'(Yc)

            const float beta_den = cSigma * pHatBackZ + M_c * p_c;
            if (beta_den > EPSILON)
            {
                const float beta = (c_zp / max(M_sum, 1.0f)) * (M_c * p_c) / beta_den;
                mis_c += (float)M_nc * beta;       // (M_nc / Nc) with Nc=1
            }
        }
    }

    rdi.w_sum = mis_c * p_c * rdi.W;

    //====================================
    //NON-CANONICAL STOCHASTIC REUSE  (Eq.16, Ntilde draws WITH replacement)
    //====================================
    if (Wsel_nz > 0.0f)
    {
        for (uint d = 0u; d < Ntilde; ++d)
        {
            //pick z ~ P_s(z) = w_z / Wsel_nz by CDF walk over the cell's nonzero slots
            const float targetW = RandomFloatSingle(seed.x) * Wsel_nz;
            uint  zli = 0xFFFFFFFFu;
            float acc = 0.0f;
            [loop]
            for (uint li = 0u; li < MAXCELL; ++li)
            {
                if (candW[li] <= 0.0f) continue;
                acc += candW[li];
                if (targetW <= acc) { zli = li; break; }
            }
            if (zli == 0xFFFFFFFFu) continue;
            const float w_z_sel = candW[zli];
            if (w_z_sel <= 0.0f) continue;

            const int2   sc   = tileOrigin + int2(zli & 7u, zli >> 3u);
            const uint   zPx  = MapPixelID(dims, (uint2)sc);
            const float3 zPos = load_x1(g_sample_current, zPx);
            const float  ci_z = asfloat(g_pathStateBuffer.Load(cell_addr(zPx) + 12u));  // unscaled confidence
            const float  c_z  = s * ci_z;                                               // §4.3 scaled

            Reservoir pr = loadReservoir(g_Reservoirs_current, zPx);

            //forward shift: neighbor z's sample reconnected at MY primary hit
            float Jn = 0.0f;
            float3 c = Reconnect(
                sv_me.x, sv_me.n_s, sv_me.o, sv_me.matID,
                sv_me.Kd, sv_me.Pr, sv_me.Pm, sv_me.etai, sv_me.etat,
                pr.matID, pr.x2, pr.n2_s, pr.L2, pr.V2,
                pr.Kd, pr.Pr, pr.Pm, pr.eta, Jn);

            float ph = GetPHat(c), vis = 0.0f;
            if (ph > 0.0f)
            {
                if (pr.matID == MATID_ENV_MISS)
                    vis = IsVisibleEnvMiss(sv_me.x, sv_me.n_s, normalize(pr.x2), RAY_TMAX_PLANET, myInst) ? 1.0f : 0.0f;
                else { const float3 cn = pr.x2 - sv_me.x; const float cd = length(cn);
                       vis = (cd > EPSILON && IsVisible(sv_me.x, sv_me.n_s, cn / cd, cd * 0.999f)) ? 1.0f : 0.0f; }
            }
            c *= vis;

            const float Jc_z         = (pr.matID == MATID_ENV_MISS) ? 1.0f
                                                                    : ComputeJc(zPos, pr.x2, pr.n2_s);
            const float pHat_me_to_z = GetPHat(c) * JacobianRatio(Jn, Jc_z);   // p_hat(Y)*|dT|

            const float m_pw = PairwiseMIS_Neighbor_Spat(
                                    M_sum, M_c, c_z, pHat_me_to_z, pr.W, GetPHat(pr.F));

            //stochastic correction: per with-replacement draw, (1/Ntilde)(1/P_s(z))
            const float m_tilde = (Wsel_nz / w_z_sel) / (float)Ntilde * m_pw;
            const float w_draw  = m_tilde * pHat_me_to_z * pr.W;

            rdi.w_sum += w_draw;
            rdi.M     += (uint)ci_z;   // unscaled capped confidence (next-frame cap bounds it)

            if (rdi.w_sum > 0.0f && RandomFloatSingle(seed.x) < (w_draw / rdi.w_sum))
            {
                rdi.x2 = pr.x2; rdi.n2_s = pr.n2_s; rdi.objID = pr.objID;
                rdi.matID = pr.matID; rdi.eta = pr.eta;
                rdi.Kd = pr.Kd; rdi.Pr = pr.Pr; rdi.Pm = pr.Pm;
                rdi.L2 = pr.L2; rdi.V2 = pr.V2;
                contrib_final = c;
            }
        }
    }

    //====================================
    //FINALIZE  (mirrors Pass_spat_gi_v8_1)
    //====================================
    rdi.F = contrib_final;
    const float F_mag = GetPHat(rdi.F);

    if (F_mag > EPSILON && rdi.w_sum > 0.0f && rdi.w_sum < 1e10f)
    {
        float W = rdi.w_sum / F_mag;
        if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
        rdi.W = W;
    }
    else
    {
        rdi.W = 0.0f;
    }

    gScratchPing[uint3(tid.xy, 2)] = float4(rdi.F * rdi.W, 0);
    storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
}
