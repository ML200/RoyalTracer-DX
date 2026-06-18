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
//  1. Pass_spat_gi_cellbuild_v8 builds tile-local cells -> a per-pixel record in
//     the path-state scratch buffer (Load4 = [inst, normalPk, c_i, w_i]; confSum@+16).
//  2. (here) §5.1 CELL SEARCH: a single growing-radius WRS pass selects a
//     neighbour CELL weighted by its confidence sum (skips cells whose key
//     differs from ours), reaching outside disocclusions as the radius grows.
//  3. Gather the chosen cell (the 8x8 screen block around the picked pixel,
//     filtered to our cellKey) AND select our reused samples in the SAME pass by
//     streaming reservoir sampling (no per-slot candidate array).
//  4. Stochastic pairwise MIS over the pool: §4.3 confidence scaling, canonical
//     weight Eq.18 (Nc=1), Ntilde non-canonical draws prop. to Eq.17 weight with
//     the (1/Ntilde)(1/P_s) correction, reusing PairwiseMIS_Neighbor_Spat.
//Output contract mirrors Pass_spat_gi_v8_1: gScratchPing[.,2]=F*W +
//storeReservoir(g_Reservoirs_last,...), neighbours read from g_Reservoirs_current.
//
//PERF: the gather no longer fills a float candW[64] local array (a dynamically
//indexed array spills to scratch memory and was re-walked once per draw). Instead
//the single gather scan runs the canonical uniform reservoir + Ntilde non-canonical
//WRS reservoirs inline, keeping all selection state in registers. The per-pixel
//record is laid out so the gather needs ONE Load4 (c_i + w_i live in the first
//16 B); confSum, which only the §5.1 search reads, moved to +16.

static const uint  CELL_REC      = 20u;            // per-pixel record stride (cellbuild)
static const float SEARCH_GROWTH = 1.25f;          // §5.1 radius growth per iteration
//Gather window = CELL_GATHER_DIM x CELL_GATHER_DIM screen pixels centered on the
//cell the §5.1 search picked (paper: 8 -> M<=64). This is the per-pixel GATHER
//LOAD count. LOWER it (6 -> 36, or 4 -> 16) to cut gather record loads + cone taps
//~quadratically, at the cost of a smaller candidate pool (slightly higher variance,
//still unbiased - §4.3 uses M=poolsize). Keep it a power of two. Default 8 = paper.
static const uint  CELL_GATHER_DIM = 8u;
static const uint  MAXCELL         = CELL_GATHER_DIM * CELL_GATHER_DIM;
//§5.1 search: samples per growing-radius iteration. The search budget is
//rs_cellSearchIters x CELL_SEARCH_K samples; spreading K angularly-stratified
//samples per radius raises the found-rate (esp. thin/edge/disocclusion geometry)
//~Kx. It costs record loads, NOT rays, and the pass is ray-bound, so it is nearly
//free. K=1 == the original one-sample-per-iter behaviour. Raise for harder scenes.
static const uint  CELL_SEARCH_K   = 4u;
//Compile-time cap on Ntilde so the per-draw reservoir state (ncPx/ncWi/ncCi/ncWsum)
//is fixed-size and stays in registers instead of spilling to scratch. Ntilde is
//clamped to this in-shader; the editor slider / Renderer clamp match it. The paper
//uses Ntilde=3 and Fig.7 shows >3 gives diminishing returns, so 8 is generous.
static const uint  NT_MAX        = 8u;
//SOFT normal cone for cell membership (query-relative, no hard quantization
//boundary -> no edge grid where normals vary fast). ~25 deg. Used as the
//membership floor when COMPAT_MODE is OFF; COMPAT_MODE uses rs_cellCompatFloor.
static const float CELL_COHERENCE_COS = 0.9f;
//Strict-positive floor for the Junkins compatibility proposal weight (COMPAT_MODE)
//so every membership-passing member keeps P(i) > 0 (Eq.15 positivity) - the
//stochastic-MIS correction is only unbiased if no contributing member is
//unreachable. The real lower bound is pow(rs_cellCompatFloor, rs_cellBeta); this
//only bites if the floor is set near 0.
static const float COMPAT_EPS = 1e-4f;

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
    const uint  Ntilde = clamp(rs_cellN, 1u, NT_MAX);

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
        const float dA   = 6.2831853f / (float)CELL_SEARCH_K;   // angular stratification step
        for (uint i = 0u; i < iters; ++i)
        {
            //K angularly-stratified samples at this radius: a random base angle plus
            //even K-way splits covers all orientations, so a THIN sliver is hit no
            //matter how it lies, and the per-pixel found-probability rises ~Kx. The
            //random base angle keeps 'chosen' varying per pixel (no grid).
            const float baseA = 6.2831853f * RandomFloatSingle(seed.x);
            [unroll]
            for (uint k = 0u; k < CELL_SEARCH_K; ++k)
            {
                const float rr = r * sqrt(RandomFloatSingle(seed.x));
                const float a  = baseA + dA * (float)k;
                const int2  q  = int2(tid.xy) + int2(round(float2(rr * cos(a), rr * sin(a))));
                if (q.x < 0 || q.y < 0 || q.x >= (int)IMG_W || q.y >= (int)IMG_H) continue;
                const uint  qpx  = MapPixelID(dims, (uint2)q);
                const uint4 qrec = g_pathStateBuffer.Load4(cell_addr(qpx));   // inst, normalPk, c_i, w_i
                if (qrec.x != myInst) continue;                              // same surface
                float hN = 1.0f;                                             // Junkins compatibility (COMPAT_MODE)
                if (!CELL_IGNORE_NORMALS)
                {
                    const float nd       = dot(myN, UnpackNormal(qrec.y));
                    //COMPAT_MODE = Junkins compatibility-guided selection: loosen the
                    //hard 0.9 cone to rs_cellCompatFloor (a WIDER, reachable pool) and
                    //rank cells by the continuous h_n = nd^beta instead of a 1/0 step.
                    //Pure G-buffer -> the search distribution may use it freely.
                    const float floorCos = COMPAT_MODE ? rs_cellCompatFloor : CELL_COHERENCE_COS;
                    if (nd < floorCos) continue;
                    if (COMPAT_MODE) hN = max(pow(saturate(nd), rs_cellBeta), COMPAT_EPS);
                }
                const float cw0 = asfloat(g_pathStateBuffer.Load(cell_addr(qpx) + 16u));  // confSum (at +16)
                if (cw0 <= 0.0f) continue;
                const float cw = cw0 * hN;   // cell confidence sum x compatibility
                wsum += cw;
                if (RandomFloatSingle(seed.x) < cw / wsum) { chosen = q; found = true; }
            }
            r *= SEARCH_GROWTH;
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
    //GATHER THE CHOSEN CELL + STREAMING SELECTION  (one pass over the 8x8 window)
    //====================================
    //8x8 window CENTERED on 'chosen' (translation invariant) instead of snapping
    //to a fixed 8-aligned screen block: a fixed block quantizes the pool to a
    //screen grid (the 8x8 grid); a centered window slides continuously so adjacent
    //pixels' pools overlap and there is no boundary for a grid to align to.
    //
    //We scan the window ONCE and, in the same pass, (a) accumulate cSigma / Wsel_nz
    /// M_nc and (b) draw our reused samples by streaming reservoir sampling. This
    //replaces the old float candW[64] (a dynamically indexed local array that spills
    //to scratch memory and was re-walked for the canonical pick + each non-canonical
    //draw). All selection state below is scalar/fixed-size -> stays in registers.
    const int  hw         = (int)(CELL_GATHER_DIM >> 1u);
    const int2 tileOrigin = chosen - int2(hw, hw);

    float cSigmaRaw = 0.0f;          // Σ c_i over gathered non-canonical (pre §4.3 scale)
    float Wsel_nz   = 0.0f;          // Σ w_i over gathered w_i>0 (the cellReservoirs set)
    uint  M_nc      = 0u;            // count of gathered in-cell non-canonical pixels

    //canonical reverse-shift target: ONE pixel drawn uniformly over the in-cell
    //non-canonical pixels (Eq.18, P_c = 1/M_nc; includes w_i==0 members, i.e. the
    //full cellPixelIndices set minus self). Streaming count-based reservoir.
    int   canPx   = -1;
    float canCi   = 0.0f;
    uint  canSeen = 0u;

    //Ntilde independent 1-sample WRS reservoirs == Ntilde with-replacement draws
    //∝ w_i over the cellReservoirs set (Eq.17). If draw d picks slot z, ncWi[d]/
    //ncCi[d] cache its weight/confidence so no record re-read is needed afterwards.
    int   ncPx  [NT_MAX];
    float ncWi  [NT_MAX];
    float ncCi  [NT_MAX];
    float ncWsum[NT_MAX];
    [unroll] for (uint dInit = 0u; dInit < NT_MAX; ++dInit)
    { ncPx[dInit] = -1; ncWi[dInit] = 0.0f; ncCi[dInit] = 0.0f; ncWsum[dInit] = 0.0f; }

    [loop]
    for (uint li = 0u; li < MAXCELL; ++li)
    {
        const int2 sc = tileOrigin + int2(li % CELL_GATHER_DIM, li / CELL_GATHER_DIM);
        if (sc.x < 0 || sc.y < 0 || sc.x >= (int)IMG_W || sc.y >= (int)IMG_H) continue;
        const uint spx = MapPixelID(dims, (uint2)sc);
        if (spx == pixelIdx) continue;        // self is the canonical, handled separately
        const uint4 rec = g_pathStateBuffer.Load4(cell_addr(spx));   // inst, normalPk, c_i, w_i
        if (rec.x != myInst) continue;                               // same surface
        float hN = 1.0f;                                             // Junkins compatibility (COMPAT_MODE)
        if (!CELL_IGNORE_NORMALS)
        {
            const float nd       = dot(myN, UnpackNormal(rec.y));
            const float floorCos = COMPAT_MODE ? rs_cellCompatFloor : CELL_COHERENCE_COS;
            if (nd < floorCos) continue;
            if (COMPAT_MODE) hN = max(pow(saturate(nd), rs_cellBeta), COMPAT_EPS);
        }

        const float ci = asfloat(rec.z);
        //Eq.17 weight x Junkins compatibility h_n. h enters ONLY the non-canonical
        //PROPOSAL (Wsel_nz + ncWsum accumulator + cached ncWi below) so the
        //(Wsel_nz/w_z_sel) stochastic-MIS correction in the reuse loop divides it
        //right back out -> unbiased (h is pure G-buffer). It must stay OUT of the
        //contribution term + m_pw (it does: those load the reservoir fresh). The
        //confidence c_i is left UNSCALED (the deterministic MIS heuristic + §4.3
        //must see the true confidence, not the reshaped proposal).
        const float wi = asfloat(rec.w) * hN;   // 0 allowed (in cell, no contributing sample)

        cSigmaRaw += ci;
        ++M_nc;

        //canonical uniform reservoir (count-based) over every in-cell member.
        //NOT h-weighted: Eq.18's P_c stays 1/M_nc (the audited 1/P_c = M_nc factor).
        ++canSeen;
        if (RandomFloatSingle(seed.x) * (float)canSeen < 1.0f) { canPx = (int)spx; canCi = ci; }

        //non-canonical WRS reservoirs over contributing members only (w_i > 0)
        if (wi > 0.0f)
        {
            Wsel_nz += wi;
            [unroll]
            for (uint d = 0u; d < NT_MAX; ++d)
            {
                if (d >= Ntilde) break;
                ncWsum[d] += wi;
                if (RandomFloatSingle(seed.x) * ncWsum[d] < wi)
                { ncPx[d] = (int)spx; ncWi[d] = wi; ncCi[d] = ci; }
            }
        }
    }

    const uint  M_cell = M_nc + 1u;           // incl. self == paper's M
    const float s      = (M_cell > 0u) ? min(1.0f, (float)Ntilde / (float)M_cell) : 1.0f;  // §4.3
    const float cSigma = s * cSigmaRaw;
    const float M_sum  = M_c + cSigma;

    //my canonical surface vertex + geometric Jacobian, shared by both shifts
    const float3        camPos = InitOrigin();
    const float3        myPos  = load_x1_with_instID(g_sample_current, pixelIdx, myInst);
    const SurfaceVertex sv_me  = BuildVertex(g_sample_current, pixelIdx, myPos, camPos);
    const float my_Jc = (rdi.matID == MATID_ENV_MISS) ? 1.0f
                                                       : ComputeJc(myPos, rdi.x2, rdi.n2_s);

    //====================================
    //CANONICAL MIS WEIGHT  (Eq.18, Nc=1 uniform over the cell)
    //====================================
    float mis_c = M_c / max(M_sum, 1.0f);
    if (p_c > 0.0f && M_nc > 0u && canPx >= 0)
    {
        const uint  zpPx = (uint)canPx;
        const float c_zp = s * canCi;                       // §4.3 scaled c_i

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
            { const float3 md = normalize(rdi.x2);                                 // miss: far endpoint along sky dir
              vis = IsVisible(sv_zp.x, sv_zp.n_s, sv_zp.x + md * RAY_TMAX_PLANET, -md) ? 1.0f : 0.0f; }
            else
              vis = IsVisible(sv_zp.x, sv_zp.n_s, rdi.x2, rdi.n2_s) ? 1.0f : 0.0f;
        }
        const float pHatBackZ = ph * JacobianRatio(Jn_rev, my_Jc) * vis;   // p_hat<-z'(Yc)

        const float beta_den = cSigma * pHatBackZ + M_c * p_c;
        if (beta_den > EPSILON)
        {
            const float beta = (c_zp / max(M_sum, 1.0f)) * (M_c * p_c) / beta_den;
            mis_c += (float)M_nc * beta;       // (M_nc / Nc) with Nc=1
        }
    }

    rdi.w_sum = mis_c * p_c * rdi.W;

    //====================================
    //NON-CANONICAL STOCHASTIC REUSE  (Eq.16, Ntilde draws WITH replacement)
    //====================================
    if (Wsel_nz > 0.0f)
    {
        [unroll]
        for (uint d = 0u; d < NT_MAX; ++d)
        {
            if (d >= Ntilde) break;
            if (ncPx[d] < 0) continue;              // this reservoir picked nothing
            const float w_z_sel = ncWi[d];
            if (w_z_sel <= 0.0f) continue;
            const float ci_z = ncCi[d];             // unscaled capped confidence
            const float c_z  = s * ci_z;            // §4.3 scaled
            const uint  zPx  = (uint)ncPx[d];

            const float3 zPos = load_x1(g_sample_current, zPx);
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
                { const float3 md = normalize(pr.x2);                                 // miss: far endpoint along sky dir
                  vis = IsVisible(sv_me.x, sv_me.n_s, sv_me.x + md * RAY_TMAX_PLANET, -md) ? 1.0f : 0.0f; }
                else
                  vis = IsVisible(sv_me.x, sv_me.n_s, pr.x2, pr.n2_s) ? 1.0f : 0.0f;
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
