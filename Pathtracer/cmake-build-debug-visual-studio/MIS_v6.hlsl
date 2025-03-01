// ReSTIR DI generalized MIS
float GenPairwiseMIS_canonical(Reservoir_DI c, Reservoir_DI n[spatial_candidate_count], SampleData sample_c, SampleData sample_n[spatial_candidate_count], bool rejected[spatial_candidate_count], float M_sum, float M_cap){
    float m_c = min(M_cap,c.M) / M_sum;
    float m_num = min(M_cap,c.M) * GetP_Hat(c, c, sample_c, false);

    for(int j = 0; j < spatial_candidate_count; j++){
        if(!rejected[j]){
            float p_hat_from = GetP_Hat(n[j], c, sample_n[j], true);
            float m_den = m_num + ((M_sum - min(M_cap,c.M)) * p_hat_from);
            if(m_den > 0.0f)
                m_c += (min(M_cap,n[j].M)/M_sum) * (m_num / m_den);
        }
    }
    return m_c;
}

float GenPairwiseMIS_noncanonical(Reservoir_DI c, Reservoir_DI n, SampleData sample_c, SampleData sample_n, float M_sum, float M_cap){
    float p_hat_from = GetP_Hat(n, c, sample_n, false);
    float m_num = (M_sum - min(M_cap,c.M)) * p_hat_from;
    float m_den = m_num + (min(M_cap,c.M) * GetP_Hat(c, c, sample_c, false));

    if(m_den > 0.0f)
        return (min(M_cap,n.M) / M_sum) * m_num / m_den;
    else return 0.0f;
}

// Temporal variant Pairwise MIS
float GenPairwiseMIS_canonical_temporal(Reservoir_DI c, Reservoir_DI n, float M_sum, float M_cap){
    float m_c = min(M_cap, c.M) / M_sum;
    float m_num = min(M_cap, c.M);

    float m_den = m_num + (M_sum - min(M_cap, c.M));
    if(m_den > 0.0f)
        m_c += (min(M_cap, n.M) / M_sum) * (m_num / m_den);
    return m_c;
}

float GenPairwiseMIS_noncanonical_temporal(Reservoir_DI c, Reservoir_DI n, float M_sum, float M_cap){
    float m_num = (M_sum - min(M_cap,c.M));
    float m_den = m_num + min(M_cap,c.M);

    if(m_den > 0.0f)
        return (min(M_cap,n.M) / M_sum) * m_num / m_den;
    else return 0.0f;
}