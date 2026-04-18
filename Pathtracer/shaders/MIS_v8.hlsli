// Pairwise MIS: canonical sample (defensive)
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

// Pairwise MIS: neighbor sample (defensive)
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


#ifdef ENABLE_RAY_QUERY_INLINE
// Pairwise MIS: canonical spatial
float PairwiseMIS_Canonical_Spat(
    in float M_sum_in,
    in float p_c,
    in float M_c,
    in uint nIds[SPAT_COUNT_MAX], // IDs of the candidates; early out if id is invalid
    // data needed from the canonical reservoir (we dont want to load the complete struct in here)
    in float3 x2_c,
    in float3 n2s_c,
    in float3 L2_c,
    in float3 V2_c,
    in uint   matID_c,
    in float3 localKd2_c,
    in float  localPr2_c,
    in float  localPm2_c,
    in float  eta_c,
    in float  J_c
    )
{
    float M_sum = max(M_sum_in, 1.0f);
    float m_c = M_c / M_sum;
    float m_num = M_c * p_c;

    [unroll]
    for(int i = 0; i < SPAT_COUNT_MAX; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            uint id = nIds[i];
            uint nInstID = load_instID(g_sample_current, id);
            uint nPrimID = load_primID(g_sample_current, id);
            float2 nBary = load_bary(g_sample_current, id);
            SurfaceVertex sv_n1 = BuildVertexLight(nInstID, nPrimID, nBary,
                load_n1_s_with_instID(g_sample_current, id, nInstID),
                load_uv(g_sample_current, id),
                InitOrigin());

            float Jn = 0.0f;
            float p_hat_from = GetPHat(Reconnect(
                sv_n1.x, sv_n1.n_s, sv_n1.o, sv_n1.matID,
                sv_n1.Kd, sv_n1.Pr, sv_n1.Pm, sv_n1.etai, sv_n1.etat,
                matID_c, x2_c, n2s_c, L2_c, V2_c,
                localKd2_c, localPr2_c, localPm2_c, eta_c,
                Jn));
            {
                float3 _conn = x2_c - sv_n1.x; float _cd = length(_conn);
                float J = JacobianRatio(Jn, J_c);
                p_hat_from *= (_cd > EPSILON && IsVisible(sv_n1.x, sv_n1.n_s, _conn / _cd, _cd * 0.999f)) ? J : 0.0f;
            }
            float m_den = m_num + (M_sum - M_c) * p_hat_from;
            if(m_den > EPSILON)
                m_c += (min(SPAT_MCAP, load_M(g_Reservoirs_current, id)) / M_sum) * (m_num / m_den);
        }
    }
    return m_c;
}
#endif // ENABLE_RAY_QUERY_INLINE


#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat(
    in float M_sum_in,
    in float M_c,
    in float M_n,
    in float p_hat_from,
    // data needed from the canonical reservoir (we dont want to load the complete struct in here)
    in float W_n,
    in float F_n
    )
{
    float M_sum = max(M_sum_in, 1.0f);
    float visReuse = W_n > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * F_n;
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den>EPSILON)
        return (M_n/M_sum) * (m_num/m_den);
    return 0.0f;
}
#endif // ENABLE_RAY_QUERY_INLINE
