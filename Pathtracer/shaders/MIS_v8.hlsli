// Initial sampling
float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    float denom = M1 * pdf_nee + M2 * pdf_bsdf;
    return denom > 0.0f ? pdf_nee / denom : 0.0f;
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    float denom = M1 * pdf_bsdf + M2 * pdf_nee;
    return denom > 0.0f ? pdf_bsdf / denom : 0.0f;
}

// defensive pair-wise MIS, canonical sample
inline float PairwiseMIS_Canonical_Temp(
    float M_c,
    float M_n,
    float p_c,
    float p_n,
    float M_sum)
{
    M_sum = max(M_sum, 1.0f);
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
    M_sum = max(M_sum, 1.0f);
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
    in float M_sum_in,
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
    float M_sum = max(M_sum_in, 1.0f);
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX_DI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            uint nInstID = load_instID(g_sample_current, nIds[i]);
            uint nPrimID = load_primID(g_sample_current, nIds[i]);
            float2 nBary = load_bary(g_sample_current, nIds[i]);
            SurfaceVertex sv_n = BuildVertexLight(nInstID, nPrimID, nBary,
                load_n1_s_with_instID(g_sample_current, nIds[i], nInstID),
                load_n1_g_with_instID(g_sample_current, nIds[i], nInstID),
                load_uv(g_sample_current, nIds[i]),
                load_etai(g_sample_current, nIds[i]),
                load_etat(g_sample_current, nIds[i]),
                InitOrigin());
            float p_hat_from = GetPHat(ReconnectDI(sv_n.x, sv_n.n_s, sv_n.n_g, sv_n.o, sv_n.matID, x2_c, n2_c, L2_c, sv_n.Kd, sv_n.Pr, sv_n.Pm, sv_n.etai, sv_n.etat, objID_c));
            p_hat_from *= JacobianDeterminantDI(x1_c, x2_c, sv_n.x, n2_c, objID_c);
            // Visibility check
            {
                float3 _vd = (objID_c >= 0xFFFFFFFEu) ? normalize(x2_c) : ((x2_c - sv_n.x) / max(length(x2_c - sv_n.x), EPSILON));
                float  _vt = (objID_c >= 0xFFFFFFFEu) ? 10000.0f : (length(x2_c - sv_n.x) * 0.999f);
                p_hat_from *= IsVisible(sv_n.x, sv_n.n_g, _vd, _vt) ? 1.0f : 0.0f;
            }
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > 1e-4)
                m_c += (min(SPAT_MCAP_DI,load_M_di(g_Reservoirs_current_di, nIds[i]))/M_sum) * (m_num / m_den);
        }
    }
    return m_c;

}
#endif // ENABLE_RAY_QUERY_INLINE



#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat_DI(
    in float M_sum_in,
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
    float M_sum = max(M_sum_in, 1.0f);
    // Reconstruct p_n from the neigbour reservoir
    float visReuse = load_W_di(g_Reservoirs_current_di, nID) > 0.0f ? 1.0f : 0.0f;
    uint nInstID = load_instID(g_sample_current, nID);
    uint nPrimID = load_primID(g_sample_current, nID);
    float2 nBary = load_bary(g_sample_current, nID);
    SurfaceVertex sv_n = BuildVertexLight(nInstID, nPrimID, nBary,
        load_n1_s_with_instID(g_sample_current, nID, nInstID),
        load_n1_g_with_instID(g_sample_current, nID, nInstID),
        load_uv(g_sample_current, nID),
        load_etai(g_sample_current, nID),
        load_etat(g_sample_current, nID),
        InitOrigin());
    float p_n = visReuse * GetPHat(ReconnectDI(sv_n.x, sv_n.n_s, sv_n.n_g, sv_n.o, sv_n.matID, x2_n, n2_n, L2_n, sv_n.Kd, sv_n.Pr, sv_n.Pm, sv_n.etai, sv_n.etat, objID_n));
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
    in float M_sum_in,
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
    float M_sum = max(M_sum_in, 1.0f);
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX_GI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            uint id = nIds[i];
            uint nInstID = load_instID(g_sample_current, id);
            uint nPrimID = load_primID(g_sample_current, id);
            float2 nBary = load_bary(g_sample_current, id);
            SurfaceVertex sv_n1 = BuildVertexLight(nInstID, nPrimID, nBary,
                load_n1_s_with_instID(g_sample_current, id, nInstID),
                load_n1_g_with_instID(g_sample_current, id, nInstID),
                load_uv(g_sample_current, id),
                load_etai(g_sample_current, id),
                load_etat(g_sample_current, id),
                InitOrigin());

            SurfaceVertex sv_c2 = { x2_c, n2s_c, n2g_c, V2_c, localKd2_c, localPr2_c, localPm2_c, etai2_c, etat2_c, matID_c, float2(0,0) };

            float Jn = 0.0f;
            float J = 0.0f;
            float p_hat_from = GetPHat(ReconnectGI(
                sv_n1.x, sv_n1.n_s, sv_n1.n_g, sv_n1.o, sv_n1.matID,
                sv_n1.Kd, sv_n1.Pr, sv_n1.Pm, sv_n1.etai, sv_n1.etat,
                sv_c2.matID, sv_c2.x, sv_c2.n_s, sv_c2.n_g, L2_c, sv_c2.o,
                sv_c2.Kd, sv_c2.Pr, sv_c2.Pm, sv_c2.etai, sv_c2.etat,
                pdfx2_c, J_c, true, Jn, J));
            {
                float3 _conn = x2_c - sv_n1.x; float _cd = length(_conn);
                p_hat_from *= (_cd > EPSILON && IsVisible(sv_n1.x, sv_n1.n_g, _conn / _cd, _cd * 0.999f)) ? J : 0.0f;
            }
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > EPSILON)
                m_c += (min(SPAT_MCAP_GI,load_M_gi(g_Reservoirs_current_gi, id))/M_sum) * (m_num / m_den);
        }
    }
    return m_c;
}
#endif // ENABLE_RAY_QUERY_INLINE


#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat_GI(
    in float M_sum_in,
    in float M_c,
    in float M_n,
    in float p_hat_from,
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float W_n,
    in float F_n
    )
{
    float M_sum = max(M_sum_in, 1.0f);
    // Reconstruct p_n from the neigbour reservoir
    float visReuse = W_n > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * F_n;
    // p_hat_from is in this case the reconnection between the canoncial position and the neighbor sample. Cause we need that later, it is provided
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den>EPSILON)
        return (M_n/M_sum) * (m_num/m_den);
    return 0.0f;
}
#endif // ENABLE_RAY_QUERY_INLINE

