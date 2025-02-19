// ReSTIR DI generalized MIS
float GenPairwiseMIS_canonical(Reservoir c, Reservoir n[spatial_candidate_count], bool rejected[spatial_candidate_count], float M_sum){
    if(M_sum == 0.0f)
        return 0.0f;
    float m_c = min(spatial_M_cap,c.M) / M_sum;
    float m_num = min(spatial_M_cap,c.M) * c.p_hat;

    for(int j = 0; j < spatial_candidate_count; j++){
        if(!rejected[j]){
            float p_hat_from = length(ReconnectDI(n[j].x1,n[j].n1,c.x2,c.n2, c.L2, n[j].o, n[j].s, materials[n[j].mID]));
            float m_den = m_num + (M_sum - min(spatial_M_cap,c.M)) * p_hat_from;
            if(m_den > 0.0f)
                m_c += (min(spatial_M_cap,n[j].M)/M_sum) * (m_num / m_den);
        }
    }
    return m_c;
}


float GenPairwiseMIS_noncanonical(Reservoir c, Reservoir n, float M_sum){
    if(M_sum == 0.0f)
        return 0.0f;
    float p_hat_from = length(ReconnectDI(c.x1,c.n1,n.x2,n.n2,n.L2, c.o, c.s, materials[c.mID]));
    float m_num = (M_sum - min(spatial_M_cap,c.M)) * p_hat_from;
    float m_den = m_num + min(spatial_M_cap,c.M) * c.p_hat;

    if(m_den > 0.0f)
        return (min(spatial_M_cap,n.M) / M_sum) * m_num / m_den;
    else return 0.0f;
}