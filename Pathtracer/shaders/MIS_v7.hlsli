// ReSTIR DI generalized MIS
float GenPairwiseMIS_canonical(
    Reservoir_DI c,
    uint n[spatial_candidate_count],
    SampleData   sample_c,
    bool         rejected[spatial_candidate_count],
    float        M_sum,
    float        M_cap,
    MaterialOptimized matOpt)
{
    float c_M_min = min(M_cap, c.M);
    float c_M_max = M_sum - c_M_min;
    float p_c     = GetP_Hat(sample_c.x1, sample_c.n1, c.x2, c.n2, c.L2, sample_c.o, matOpt, false);
    float c_m_num = c_M_min * p_c;
    float m_c = c_M_min / M_sum;

    [loop]
    for(int j = 0; j < spatial_candidate_count; j++)
    {
        if (!rejected[j])
        {
            float n_M_min   = min(M_cap, g_Reservoirs_current[n[j]].M);
            float p_hat_from = GetP_Hat(g_sample_current[n[j]].x1, g_sample_current[n[j]].n1, c.x2, c.n2, c.L2, g_sample_current[n[j]].o, matOpt, true);

            float m_den = c_m_num + (c_M_max * p_hat_from);
            if (m_den > 0.0f)
            {
                float ratio  = (n_M_min / M_sum) * (c_m_num / m_den);
                m_c += ratio;
            }
        }
    }

    return m_c;
}


float GenPairwiseMIS_noncanonical(
    Reservoir_DI c,
    uint n,
    SampleData   sample_c,
    float        M_sum,
    float        M_cap,
    MaterialOptimized matOpt)
{
    float c_M_min = min(M_cap, c.M);
    float p_c     = GetP_Hat(sample_c.x1, sample_c.n1, c.x2, c.n2, c.L2, sample_c.o, matOpt, false);

    float p_hat_from = GetP_Hat(g_sample_current[n].x1, g_sample_current[n].n1, c.x2, c.n2, c.L2, g_sample_current[n].o, matOpt, false);
    float m_num      = (M_sum - c_M_min) * p_hat_from;
    float m_den      = m_num + (c_M_min * p_c);

    if (m_den > 0.0f) {
        float n_M_min = min(M_cap, g_Reservoirs_current[n].M);
        return (n_M_min / M_sum) * (m_num / m_den);
    }
    else {
        return 0.0f;
    }
}


// Temporal variant Pairwise MIS
inline float GenPairwiseMIS_canonical_temporal(Reservoir_DI c, Reservoir_DI n, float M_sum, float M_cap){
    float m_c = min(M_cap, c.M) / M_sum;
    float m_num = min(M_cap, c.M);

    float m_den = m_num + (M_sum - min(M_cap, c.M));
    if(m_den > 0.0f)
        m_c += (min(M_cap, n.M) / M_sum) * (m_num / m_den);
    return m_c;
}

inline float GenPairwiseMIS_noncanonical_temporal(Reservoir_DI c, Reservoir_DI n, float M_sum, float M_cap){
    float m_num = (M_sum - min(M_cap,c.M));
    float m_den = m_num + min(M_cap,c.M);

    if(m_den > 0.0f)
        return (min(M_cap,n.M) / M_sum) * m_num / m_den;
    else return 0.0f;
}