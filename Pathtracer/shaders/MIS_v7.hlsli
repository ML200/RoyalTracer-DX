// Initial sampling
float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    return pdf_nee / (M1 * pdf_nee + M2 * pdf_bsdf);
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    return pdf_bsdf / (M1 * pdf_bsdf + M2 * pdf_nee);
}

// Initial sampling
/*float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    return 1.0f / (M1 * 1.0f + M2 * 1.0f);
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    return 1.0f / (M1 * 1.0f + M2 * 1.0f);
}*/

// defensive pair-wise MIS, canonical sample
inline float PairwiseMIS_Canonical_Temp(
    float M_c,      // multiplicity / confidence of canonical
    float M_n,      // multiplicity / confidence of neighbour
    float p_c,      // p̂_c(y_c)
    float p_n,      // p̂←n(y_c)  **was called n_c before**
    float M_sum)    // = M_c + M_n
{
    float num   = M_c * p_c;                  // M_c p_c
    float denom = num + M_n * p_n;            // M_c p_c + M_n p_n

    float m_c = M_c / M_sum;                  // 1st term of Eq. (7.6)
    if (denom > 0.0f)
        m_c += (M_n / M_sum) * (num / denom); // 2nd term

    return m_c;                               // m_c ∈ [0,1]
}

// defensive pair-wise MIS, neighbour sample
inline float PairwiseMIS_Neighbour_Temp(
    float M_c,
    float M_n,
    float p_c,      // p̂_c(y_c)
    float n_n,      // p̂_n(y_n)  **new variable**
    float M_sum)
{
    float num   = M_n * n_n;                  // M_n n_n
    float denom = num + M_c * p_c;            // M_n n_n + M_c p_c

    return (denom > 0.0f)
           ? (M_n / M_sum) * (num / denom)
           : 0.0f;
}

// ──────────────────────────────────────────────────────────────────────
// Non-defensive pair-wise MIS   –   canonical sample  (y_c)
// m_c = (M_c · p_c) / (M_c · p_c + M_n · p_n)
// ──────────────────────────────────────────────────────────────────────
float PairwiseMIS_Canonical_Temp_NonDef(
    float M_c,      // multiplicity / confidence of canonical
    float M_n,      // multiplicity / confidence of neighbour
    float p_c,      // p̂_c (y_c)
    float p_n,
    float M_sum)      // p̂←n (y_c)
{
    float num   = M_c * p_c;
    float denom = num + M_n * p_n;

    return (denom > 0.0f) ? (num / denom) : 0.0f;   // 0 ≤ m_c ≤ 1
}


// ──────────────────────────────────────────────────────────────────────
// Non-defensive pair-wise MIS   –   neighbour sample  (y_n)
// m_n = (M_n · n_n) / (M_n · n_n + M_c · n_c)
// ──────────────────────────────────────────────────────────────────────
float PairwiseMIS_Neighbour_Temp_NonDef(
    float M_c,
    float M_n,
    float n_c,      // p̂_c (y_n)   ← canonical PDF at neighbour point
    float n_n,
    float M_sum)      // p̂_n (y_n)   ← neighbour’s own PDF
{
    float num   = M_n * n_n;
    float denom = num + M_c * n_c;

    return (denom > 0.0f) ? (num / denom) : 0.0f;   // 0 ≤ m_n ≤ 1
}

// Symmetric pair-wise MIS, canonical pixel
float PairwiseMIS_Sym_Canonical(
    float Mc,   // multiplicity of canonical reservoir
    float Mn,   // multiplicity of neighbour reservoir
    float pc,   // pdf of canonical sample at canonical point
    float pn,   // pdf of neighbour sample re-evaluated at canonical point
    float beta) // soft-clamp exponent in [0,1]
{
    // Prevent div/0 *once*; everything else folds out naturally
    if (Mc == 0.0f) return 0.0f;

    // Symmetric pdf ratio, raised to beta
    float D = (pc == 0.0f || pn == 0.0f) ? 0.0f
              : pow( min(pc / pn, pn / pc), beta );

    //    m_c = 1 / (1 + (M_n/M_c)*D)
    return 1.0f / (1.0f + (Mn / Mc) * D);
}

// Symmetric pair-wise MIS, neighbour pixel
float PairwiseMIS_Sym_Neighbour(
    float Mc, float Mn,
    float pc, float pn,
    float beta)
{
    if (Mn == 0.0f) return 0.0f;

    float D = (pc == 0.0f || pn == 0.0f) ? 0.0f
              : pow( min(pc / pn, pn / pc), beta );

    //    m_n = 1 / (1 + (M_c/M_n)*D)
    return 1.0f / (1.0f + (Mc / Mn) * D);
}



#ifdef ENABLE_RAY_QUERY_INLINE
// Algorithm 7 from the gentle intro
float PairwiseMIS_Canonical_Spat_DI(
    in float M_sum,
    in float p_c,
    in float M_c,
    in uint nIds[SPAT_COUNT_MAX_DI],// IDs of the candidates; early out if id is invalid
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x2_c,
    in float3 n2_c,
    in float3 L2_c
    )
{
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX_DI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            float3 x1 = load_x1(g_sample_current, nIds[i]);
            float3 n1 = load_n1(g_sample_current, nIds[i]);
            float p_hat_from = GetPHat(ReconnectDI(x1, n1, load_o(g_sample_current, nIds[i]), load_matID(g_sample_current, nIds[i]), x2_c, n2_c, L2_c)); // p_hat if the canonical sample as seen from the neighbor position
            p_hat_from *= VisibilityCheckCP(x1, x2_c, n1); // visibility check
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > 0.0f)
                m_c += (min(SPAT_MCAP_DI,load_M_di(g_Reservoirs_current_di, nIds[i]))/M_sum) * (m_num / m_den); // Load M explicitly from vram/cache
        }
    }
    return m_c;

}
#endif // ENABLE_RAY_QUERY_INLINE



#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat_DI(
    in float M_sum,
    in float M_c,
    in float M_n,
    in float p_c,
    in float p_hat_from,
    in uint nID,// ID of the current candidate
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x2_n,
    in float3 n2_n,
    in float3 L2_n
    )
{
    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_di(g_Reservoirs_current_di, nID) > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * GetPHat(ReconnectDI(load_x1(g_sample_current, nID), load_n1(g_sample_current, nID), load_o(g_sample_current, nID), load_matID(g_sample_current, nID), x2_n, n2_n, L2_n));
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den>0.0f)
        return (M_n/M_sum) * (m_num/m_den);
    return 0.0f;
}
#endif // ENABLE_RAY_QUERY_INLINE


#ifdef ENABLE_RAY_QUERY_INLINE
// Algorithm 7 from the gentle intro
float PairwiseMIS_Canonical_Spat_GI(
    in float M_sum,
    in float p_c,
    in float M_c,
    in uint nIds[SPAT_COUNT_MAX_GI],// IDs of the candidates; early out if id is invalid
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x1_c,
    in float3 x2_c,
    in float3 n2_c,
    in float3 L2_c,
    in float3 V2_c,
    in uint matID_c,
    in float pdfx2_c
    )
{
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX_GI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            float3 x1 = load_x1(g_sample_current, nIds[i]);
            float3 n1 = load_n1(g_sample_current, nIds[i]);
            float p_hat_from = GetPHat(ReconnectGI(x1, n1, load_o(g_sample_current, nIds[i]), load_matID(g_sample_current, nIds[i]), matID_c, x2_c, n2_c, L2_c, V2_c, pdfx2_c)/JacobianDeterminant(x1_c, x2_c, x1, n2_c)); // p_hat if the canonical sample as seen from the neighbor position
            p_hat_from *= VisibilityCheckCP(x1, x2_c, n1); // visibility check
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > 0.0f)
                m_c += (min(SPAT_MCAP_GI,load_M_gi(g_Reservoirs_current_gi, nIds[i]))/M_sum) * (m_num / m_den); // Load M explicitly from vram/cache
        }
    }
    return m_c;
}
#endif // ENABLE_RAY_QUERY_INLINE



#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat_GI(
    in float M_sum,
    in float M_c,
    in float M_n,
    in float p_c,
    in float p_hat_from,
    in uint nID,// ID of the current candidate
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x2_n,
    in float3 n2_n,
    in float3 L2_n,
    in float3 V2_n,
    in uint matID_n,
    in float pdfx2_n
    )
{
    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_gi(g_Reservoirs_current_gi, nID) > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * GetPHat(ReconnectGI(load_x1(g_sample_current, nID), load_n1(g_sample_current, nID), load_o(g_sample_current, nID), load_matID(g_sample_current, nID), matID_n, x2_n, n2_n, L2_n, V2_n, pdfx2_n));
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den>0.0f)
        return (M_n/M_sum) * (m_num/m_den);
    return 0.0f;
}
#endif // ENABLE_RAY_QUERY_INLINE


inline float SymRatio(float pA, float pB, float beta)          // D^β  (Eq. 13, raised to β)
{
    if(pA == 0.0f || pB == 0.0f)
        return 0.0f;
    float r = pA / pB;
    float D = min(r, 1.0f / r);
    return pow(D, beta);
}

#ifdef ENABLE_RAY_QUERY_INLINE
// Algorithm 7 from the gentle intro
float PairwiseMIS_Canonical_Spat_GI_Sym(
    in float M_sum,
    in float p_c,
    in float M_c,
    in uint nIds[SPAT_COUNT_MAX_GI],// IDs of the candidates; early out if id is invalid
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x1_c,
    in float3 x2_c,
    in float3 n2_c,
    in float3 L2_c,
    in float3 V2_c,
    in uint matID_c,
    in float pdfx2_c,
    in float beta
    )
{
    float m_no_r = M_sum - 1.0f;
    // Fast path: no neighbours ⇒ weight must be 1
    if (m_no_r == 0.0f)
        return 1.0f;

    float sum = 0.0f;

    for(int i = 0; i < SPAT_COUNT_MAX_GI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            float3 x1 = load_x1(g_sample_current, nIds[i]);
            float3 n1 = load_n1(g_sample_current, nIds[i]);
            float p_hat_from = GetPHat(ReconnectGI(x1, n1, load_o(g_sample_current, nIds[i]), load_matID(g_sample_current, nIds[i]), matID_c, x2_c, n2_c, L2_c, V2_c, pdfx2_c)/JacobianDeterminant(x1_c, x2_c, x1, n2_c)); // p_hat if the canonical sample as seen from the neighbor position
            p_hat_from *= VisibilityCheckCP(x1, x2_c, n1); // visibility check

            float D = SymRatio(p_c, p_hat_from, beta);
            sum += 1.0f/(1.0f + m_no_r * D);
        }
    }
    return (1.0f / m_no_r) * sum;
}
#endif // ENABLE_RAY_QUERY_INLINE



#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat_GI_Sym(
    in float M_sum,
    in float M_c,
    in float M_n,
    in float p_c,
    in float p_hat_from,
    in uint nID,// ID of the current candidate
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x2_n,
    in float3 n2_n,
    in float3 L2_n,
    in float3 V2_n,
    in uint matID_n,
    in uint pdfx2_n,
    in float beta
    )
{
    float m_no_r = M_sum - 1.0f;

    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_gi(g_Reservoirs_current_gi, nID) > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * GetPHat(ReconnectGI(load_x1(g_sample_current, nID), load_n1(g_sample_current, nID), load_o(g_sample_current, nID), load_matID(g_sample_current, nID), matID_n, x2_n, n2_n, L2_n, V2_n, pdfx2_n));
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float D = SymRatio(p_n, p_hat_from, beta);
    return D / (1.0f + m_no_r * D);

}
#endif // ENABLE_RAY_QUERY_INLINE

