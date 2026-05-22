//====================================
//PAIRWISE MIS TEMPORAL
//====================================
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


//====================================
//PAIRWISE MIS SPATIAL
//====================================
//PairwiseMIS_Canonical_Spat removed: it was unused, and was the last caller of
//BuildVertexLight (per-triangle reconstruction). Spatial-GI canonical MIS is
//computed inline in Pass_spat_gi_v8_1.


#ifdef ENABLE_RAY_QUERY_INLINE
float PairwiseMIS_Neighbor_Spat(
    in float M_sum_in,
    in float M_c,
    in float M_n,
    in float p_hat_from,
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
#endif
