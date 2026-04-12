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
// Pairwise MIS: canonical spatial DI
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

    [loop]
    for(int i = 0; i < SPAT_COUNT_MAX_DI; i++){ // Iterate over all spatial neighbor candidates, skip invalid entries
        if(nIds[i] != 0xFFFFFFFF){
            uint nInstID = gb_load_instID(g_gbuf_current, nIds[i]);
            SurfaceVertex sv_n;
            sv_n.x    = gb_load_worldPos(g_gbuf_current, nIds[i], nInstID);
            sv_n.n_s  = gb_load_normal_world(g_gbuf_current, nIds[i], nInstID);
            sv_n.o    = normalize(InitOrigin() - sv_n.x);
            sv_n.matID = gb_load_matID(g_gbuf_current, nIds[i]);
            sv_n.Kd   = gb_load_Kd(g_gbuf_current, nIds[i]);
            sv_n.Pr   = gb_load_Pr(g_gbuf_current, nIds[i]);
            sv_n.Pm   = gb_load_Pm(g_gbuf_current, nIds[i]);
            sv_n.etai = 1.0;
            sv_n.etat = materials[sv_n.matID].Ni;
            float p_hat_from = GetPHat(ReconnectDI(sv_n.x, sv_n.n_s, sv_n.o, sv_n.matID, x2_c, n2_c, L2_c, sv_n.Kd, sv_n.Pr, sv_n.Pm, sv_n.etai, sv_n.etat, objID_c));
            p_hat_from *= JacobianDeterminantDI(x1_c, x2_c, sv_n.x, n2_c, objID_c);
            // Visibility check
            {
                float3 _vd = (objID_c >= 0xFFFFFFFEu) ? normalize(x2_c) : ((x2_c - sv_n.x) / max(length(x2_c - sv_n.x), EPSILON));
                float  _vt = (objID_c >= 0xFFFFFFFEu) ? 10000.0f : (length(x2_c - sv_n.x) * 0.999f);
                p_hat_from *= IsVisible(sv_n.x, sv_n.n_s, _vd, _vt) ? 1.0f : 0.0f;
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
// Pairwise MIS: canonical spatial GI
float PairwiseMIS_Canonical_Spat_GI(
    in float M_sum_in,
    in float p_c,
    in float M_c,
    in uint nIds[SPAT_COUNT_MAX_GI],// IDs of the candidates; early out if id is invalid
    // data needed from the canonical reseroir (we dont want to load the complete struct in here)
    in float3 x2_c,
    in float3 n2s_c,
    in float3 L2_c,
    in float3 V2_c,
    in uint   matID_c,
    in float3 localKd2_c,
    in float  localPr2_c,
    in float  localPm2_c,
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
            uint nInstID = gb_load_instID(g_gbuf_current, id);
            SurfaceVertex sv_n1;
            sv_n1.x    = gb_load_worldPos(g_gbuf_current, id, nInstID);
            sv_n1.n_s  = gb_load_normal_world(g_gbuf_current, id, nInstID);
            sv_n1.o    = normalize(InitOrigin() - sv_n1.x);
            sv_n1.matID = gb_load_matID(g_gbuf_current, id);
            sv_n1.Kd   = gb_load_Kd(g_gbuf_current, id);
            sv_n1.Pr   = gb_load_Pr(g_gbuf_current, id);
            sv_n1.Pm   = gb_load_Pm(g_gbuf_current, id);
            sv_n1.etai = 1.0;
            sv_n1.etat = materials[sv_n1.matID].Ni;

            float Jn = 0.0f;
            float p_hat_from = GetPHat(ReconnectGI(
                sv_n1.x, sv_n1.n_s, sv_n1.o, sv_n1.matID,
                sv_n1.Kd, sv_n1.Pr, sv_n1.Pm, sv_n1.etai, sv_n1.etat,
                matID_c, x2_c, n2s_c, L2_c, V2_c,
                localKd2_c, localPr2_c, localPm2_c,
                Jn));
            {
                float3 _conn = x2_c - sv_n1.x; float _cd = length(_conn);
                float J = JacobianRatio(Jn, J_c);
                p_hat_from *= (_cd > EPSILON && IsVisible(sv_n1.x, sv_n1.n_s, _conn / _cd, _cd * 0.999f)) ? J : 0.0f;
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
    float visReuse = W_n > 0.0f ? 1.0f : 0.0f;
    float p_n = visReuse * F_n;
    float m_num = (M_sum - M_c) * p_n;
    float m_den = m_num + M_c * p_hat_from;
    if(m_den>EPSILON)
        return (M_n/M_sum) * (m_num/m_den);
    return 0.0f;
}
#endif // ENABLE_RAY_QUERY_INLINE

