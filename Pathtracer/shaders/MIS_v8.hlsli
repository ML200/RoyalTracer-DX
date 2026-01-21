// Initial sampling
float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    return pdf_nee / (M1 * pdf_nee + M2 * pdf_bsdf);
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    return pdf_bsdf / (M1 * pdf_bsdf + M2 * pdf_nee);
}

// defensive pair-wise MIS, canonical sample
inline float PairwiseMIS_Canonical_Temp(
    float M_c,
    float M_n,
    float p_c,
    float p_n,
    float M_sum)
{
    float num   = M_c * p_c;
    float denom = num + M_n * p_n;

    float m_c = M_c / M_sum;
    if (denom > 0.0f)
        m_c += (M_n / M_sum) * (num / denom);

    return m_c;
}

// defensive pair-wise MIS, neighbour sample
inline float PairwiseMIS_Neighbour_Temp(
    float M_c,
    float M_n,
    float p_c,
    float n_n,
    float M_sum)
{
    float num   = M_n * n_n;
    float denom = num + M_c * p_c;

    return (denom > 0.0f)
           ? (M_n / M_sum) * (num / denom)
           : 0.0f;
}

// Non-defensive pair-wise MIS
float PairwiseMIS_Canonical_Temp_NonDef(
    float M_c,
    float M_n,
    float p_c,
    float p_n,
    float M_sum)
{
    float num   = M_c * p_c;
    float denom = num + M_n * p_n;

    return (denom > 0.0f) ? (num / denom) : 0.0f;
}


// Non-defensive pair-wise MIS
float PairwiseMIS_Neighbour_Temp_NonDef(
    float M_c,
    float M_n,
    float n_c,
    float n_n,
    float M_sum)
{
    float num   = M_n * n_n;
    float denom = num + M_c * n_c;

    return (denom > 0.0f) ? (num / denom) : 0.0f;
}


#ifdef ENABLE_RAY_QUERY_INLINE
// Algorithm 7 from the gentle intro
float PairwiseMIS_Canonical_Spat_DI(
    in float M_sum,
    in float p_c,
    in float M_c,
    in uint nIds[SPAT_COUNT_MAX_DI],// IDs of the candidates; early out if id is invalid
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x1_c,
    in float3 x2_c,
    in float3 n2_c,
    in float3 L2_c,
    in uint objID_c
    )
{
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX_DI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            float3 x1 = load_x1(g_sample_current, nIds[i]);
            float3 n1 = load_n1_s(g_sample_current, nIds[i]);
            float p_hat_from = GetPHat(ReconnectDI(x1, n1, load_n1_g(g_sample_current, nIds[i]), load_o(g_sample_current, nIds[i]), load_matID(g_sample_current, nIds[i]), x2_c, n2_c, L2_c, load_localKd(g_sample_current, nIds[i]), load_localPr(g_sample_current, nIds[i]), load_localPm(g_sample_current, nIds[i]), load_etai(g_sample_current, nIds[i]), load_etat(g_sample_current, nIds[i]), objID_c)); // p_hat if the canonical sample as seen from the neighbor position
            p_hat_from *= JacobianDeterminantDI(x1_c, x2_c, x1, n2_c, objID_c);
            p_hat_from *= VisibilityCheckCP(x1, x2_c, n1, objID_c); // visibility check
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > 1e-4)
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
    in float3 L2_n,
    in uint objID_n
    )
{
    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_di(g_Reservoirs_current_di, nID) > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * GetPHat(ReconnectDI(load_x1(g_sample_current, nID), load_n1_s(g_sample_current, nID), load_n1_g(g_sample_current, nID), load_o(g_sample_current, nID), load_matID(g_sample_current, nID), x2_n, n2_n, L2_n, load_localKd(g_sample_current, nID), load_localPr(g_sample_current, nID), load_localPm(g_sample_current, nID), load_etai(g_sample_current, nID), load_etat(g_sample_current, nID), objID_n));
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den > 1e-4)
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
    in float3 x2_c,
    in float3 n2s_c,
    in float3 n2g_c,
    in float3 L2_c,
    in float3 V2_c,
    in uint   matID_c,
    in float3 localKd2_c,
    in float  localPr2_c,
    in float  localPm2_c,
    in float  etai2_c,
    in float  etat2_c,
    in float  pdfx2_c,
    in float  J_c
    )
{
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX_GI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            uint id = nIds[i];
            float3 x1 = load_x1(g_sample_current, id);
            float3 n1s = load_n1_s(g_sample_current, id);
            float3 n1g = load_n1_g(g_sample_current, id);
            float3 o   = load_o(g_sample_current, id);
            uint   mID1 = load_matID(g_sample_current, id);
            float3 kd1 = load_localKd(g_sample_current, id);
            float  pr1 = load_localPr(g_sample_current, id);
            float  pm1 = load_localPm(g_sample_current, id);
            float  ei1 = load_etai(g_sample_current, id);
            float  et1 = load_etat(g_sample_current, id);

            float Jn = 0.0f;
            float J = 0.0f;
            float p_hat_from = GetPHat(ReconnectGI(x1, n1s, n1g, o, mID1, kd1, pr1, pm1, ei1, et1, matID_c, x2_c, n2s_c, n2g_c, L2_c, V2_c, localKd2_c, localPr2_c, localPm2_c, etai2_c, etat2_c, pdfx2_c, J_c, true, Jn, J)); // p_hat if the canonical sample as seen from the neighbor position
            p_hat_from *= VisibilityCheckCP(x1, x2_c, n1s, 0u) * J; // visibility check
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > EPSILON)
                m_c += (min(SPAT_MCAP_GI,load_M_gi(g_Reservoirs_current_gi, id))/M_sum) * (m_num / m_den); // Load M explicitly from vram/cache
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
    in float3 n2s_n,
    in float3 n2g_n,
    in float3 L2_n,
    in float3 V2_n,
    in uint   matID_n,
    in float3 localKd2_n,
    in float  localPr2_n,
    in float  localPm2_n,
    in float  etai2_n,
    in float  etat2_n,
    in float  pdfx2_n
    )
{
    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_gi(g_Reservoirs_current_gi, nID) > 0.0f ? 1.0f : 0.0f;
    float3 x1  = load_x1(g_sample_current, nID);
    float3 n1s = load_n1_s(g_sample_current, nID);
    float3 n1g = load_n1_g(g_sample_current, nID);
    float3 o   = load_o(g_sample_current, nID);
    uint   mID1 = load_matID(g_sample_current, nID);
    float3 kd1 = load_localKd(g_sample_current, nID);
    float  pr1 = load_localPr(g_sample_current, nID);
    float  pm1 = load_localPm(g_sample_current, nID);
    float  ei1 = load_etai(g_sample_current, nID);
    float  et1 = load_etat(g_sample_current, nID);

    float Jn = 0.0f;
    float J = 0.0f;
    float p_n = visReuse * GetPHat(ReconnectGI(x1, n1s, n1g, o, mID1, kd1, pr1, pm1, ei1, et1, matID_n, x2_n, n2s_n, n2g_n, L2_n, V2_n, localKd2_n, localPr2_n, localPm2_n, etai2_n, etat2_n, pdfx2_n, 1.0f, false, Jn, J));
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den>EPSILON)
        return (M_n/M_sum) * (m_num/m_den);
    return 0.0f;
}
#endif // ENABLE_RAY_QUERY_INLINE


inline float SymRatio(float pA, float pB, float beta)
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
    in float3 x2_c,
    in float3 n2s_c,
    in float3 n2g_c,
    in float3 L2_c,
    in float3 V2_c,
    in uint   matID_c,
    in float3 localKd2_c,
    in float  localPr2_c,
    in float  localPm2_c,
    in float  etai2_c,
    in float  etat2_c,
    in float  pdfx2_c,
    in float  J_c,
    in float  beta
    )
{
    float m_no_r = M_sum - 1.0f;
    // Fast path: no neighbours ⇒ weight must be 1
    if (m_no_r == 0.0f)
        return 1.0f;

    float sum = 0.0f;

    for(int i = 0; i < SPAT_COUNT_MAX_GI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            uint id = nIds[i];
            float3 x1  = load_x1(g_sample_current, id);
            float3 n1s = load_n1_s(g_sample_current, id);
            float3 n1g = load_n1_g(g_sample_current, id);
            float3 o   = load_o(g_sample_current, id);
            uint   mID1 = load_matID(g_sample_current, id);
            float3 kd1 = load_localKd(g_sample_current, id);
            float  pr1 = load_localPr(g_sample_current, id);
            float  pm1 = load_localPm(g_sample_current, id);
            float  ei1 = load_etai(g_sample_current, id);
            float  et1 = load_etat(g_sample_current, id);

            float Jn = 0.0f;
            float J = 0.0f;
            float p_hat_from = GetPHat(ReconnectGI(x1, n1s, n1g, o, mID1, kd1, pr1, pm1, ei1, et1, matID_c, x2_c, n2s_c, n2g_c, L2_c, V2_c, localKd2_c, localPr2_c, localPm2_c, etai2_c, etat2_c, pdfx2_c, J_c, true, Jn, J)); // p_hat if the canonical sample as seen from the neighbor position
            p_hat_from *= VisibilityCheckCP(x1, x2_c, n1s, 0u) * J; // visibility check

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
    in float3 n2s_n,
    in float3 n2g_n,
    in float3 L2_n,
    in float3 V2_n,
    in uint   matID_n,
    in float3 localKd2_n,
    in float  localPr2_n,
    in float  localPm2_n,
    in float  etai2_n,
    in float  etat2_n,
    in float  pdfx2_n,
    in float  beta
    )
{
    float m_no_r = M_sum - 1.0f;

    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_gi(g_Reservoirs_current_gi, nID) > 0.0f ? 1.0f : 0.0f;

    float3 x1  = load_x1(g_sample_current, nID);
    float3 n1s = load_n1_s(g_sample_current, nID);
    float3 n1g = load_n1_g(g_sample_current, nID);
    float3 o   = load_o(g_sample_current, nID);
    uint   mID1 = load_matID(g_sample_current, nID);
    float3 kd1 = load_localKd(g_sample_current, nID);
    float  pr1 = load_localPr(g_sample_current, nID);
    float  pm1 = load_localPm(g_sample_current, nID);
    float  ei1 = load_etai(g_sample_current, nID);
    float  et1 = load_etat(g_sample_current, nID);

    float Jn = 0.0f;
    float J = 0.0f;
    float p_n = visReuse * GetPHat(ReconnectGI(x1, n1s, n1g, o, mID1, kd1, pr1, pm1, ei1, et1, matID_n, x2_n, n2s_n, n2g_n, L2_n, V2_n, localKd2_n, localPr2_n, localPm2_n, etai2_n, etat2_n, pdfx2_n, 1.0f, false, Jn, J));
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float D = SymRatio(p_n, p_hat_from, beta);
    return D / (1.0f + m_no_r * D);

}
#endif // ENABLE_RAY_QUERY_INLINE


