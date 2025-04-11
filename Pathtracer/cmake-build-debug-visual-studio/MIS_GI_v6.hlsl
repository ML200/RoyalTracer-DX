// ReSTIR GI Pairwise MIS canonical sample
float GenPairwiseMIS_canonical_GI(
    Reservoir_GI c,
    uint n[spatial_candidate_count],
    SampleData   sample_c,
    bool         rejected[spatial_candidate_count],
    float        M_sum,
    float        M_cap,
    MaterialOptimized matOpt)
{
    float c_M_min = min(M_cap, c.M);
    float c_M_max = M_sum - c_M_min;


    MaterialOptimized mat_gi_c = CreateMaterialOptimized(materials[c.mID2], c.mID2);
    float p_c     = LinearizeVector(GetP_Hat_GI(sample_c.x1, sample_c.n1,
                                     c.xn, c.nn,
                                     c.E3, c.Vn,
                                     sample_c.o, matOpt, mat_gi_c, false));
    float c_m_num = c_M_min * p_c;
    float m_c = c_M_min / M_sum;

    [loop]
    for(int j = 0; j < spatial_candidate_count; j++)
    {
        if (!rejected[j])
        {
            MaterialOptimized matGI = CreateMaterialOptimized(materials[c.mID2], c.mID2);
            float n_M_min   = min(M_cap, g_Reservoirs_current_gi[n[j]].M);
            float j_gi = Jacobian_Reconnection(sample_c, g_sample_current[n[j]], c.xn, c.nn, c.Vn);
            float p_hat_from = 0.0f;
            p_hat_from = LinearizeVector(GetP_Hat_GI(g_sample_current[n[j]].x1, g_sample_current[n[j]].n1, c.xn, c.nn, c.E3, c.Vn, g_sample_current[n[j]].o, matOpt, matGI, true)) * j_gi;
            float m_den = c_m_num + (c_M_max * p_hat_from);
            if (m_den > 0.0f)
            {
                float ratio  = (n_M_min / M_sum) * (c_m_num / m_den);
                m_c += ratio;
            }
        }
    }
    return clamp(m_c, 0.0f, 1.0f);
}

// Pairwise MIS neighbour sample
float GenPairwiseMIS_noncanonical_GI(
    Reservoir_GI c,
    uint n,
    SampleData   sample_c,
    float        M_sum,
    float        M_cap,
    MaterialOptimized matOpt)
{
    float c_M_min = min(M_cap, c.M);


    MaterialOptimized mat_gi_c = CreateMaterialOptimized(materials[c.mID2], c.mID2);
    float p_c     = LinearizeVector(GetP_Hat_GI(sample_c.x1, sample_c.n1,
                                     c.xn, c.nn,
                                     c.E3, c.Vn,
                                     sample_c.o, matOpt, mat_gi_c, false));


    MaterialOptimized matGI = CreateMaterialOptimized(materials[c.mID2], c.mID2);
    float j_gi = Jacobian_Reconnection(sample_c, g_sample_current[n], c.xn, c.nn, c.Vn);
    float p_hat_from = 0.0f;
    p_hat_from = LinearizeVector(GetP_Hat_GI(g_sample_current[n].x1, g_sample_current[n].n1, c.xn, c.nn, c.E3, c.Vn, g_sample_current[n].o, matOpt, matGI, false)) * j_gi;
    float m_num      = (M_sum - c_M_min) * p_hat_from;
    float m_den      = m_num + (c_M_min * p_c);

    if (m_den > 0.0f) {
        float n_M_min = min(M_cap, g_Reservoirs_current_gi[n].M);
        return clamp((n_M_min / M_sum) * (m_num / m_den), 0.0f, 1.0f);
    }
    else {
        return 0.0f;
    }
}


// Temporal variant Pairwise MIS
float GenPairwiseMIS_canonical_temporal_GI(
    Reservoir_GI c,
    Reservoir_GI n,
    SampleData sample_c,
    SampleData sample_n,
    float M_sum,
    float M_cap,
    MaterialOptimized matOpt)
{
    float m_c = min(M_cap, c.M) / M_sum;
    float m_num = min(M_cap, c.M);

    float m_den = m_num + (M_sum - min(M_cap, c.M));
    if(m_den > 0.0f)
        m_c += (min(M_cap, n.M) / M_sum) * (m_num / m_den);
    return m_c;
}

float GenPairwiseMIS_noncanonical_temporal_GI(
    Reservoir_GI c,
    Reservoir_GI n,
    SampleData sample_c,
    SampleData sample_n,
    float M_sum,
    float M_cap,
    MaterialOptimized matOpt)
{
    float m_num = (M_sum - min(M_cap,c.M));
    float m_den = m_num + min(M_cap,c.M);

    if(m_den > 0.0f)
        return (min(M_cap,n.M) / M_sum) * m_num / m_den;
    else return 0.0f;
}


// Non-defensive version of the canonical sample weight
/*float GenPairwiseMIS_canonical_temporal_GI(
    Reservoir_GI c,
    Reservoir_GI n,
    SampleData sample_c,
    SampleData sample_n,
    float M_sum,
    float M_cap,
    MaterialOptimized matOpt)
{
    // Clamped confidence weight for the canonical sample
    float c_M_min = min(M_cap, c.M);

    // Compute p_c^- (shifted PDF for the canonical sample)
    MaterialOptimized mat_gi_c = CreateMaterialOptimized(materials[c.mID2], c.mID2);
    float p_c = LinearizeVector(
        GetP_Hat_GI(sample_c.x1, sample_c.n1,
                    c.xn, c.nn,
                    c.E3, c.Vn,
                    sample_c.o, matOpt, mat_gi_c, false)
    );

    // Clamped confidence weight for the neighbor
    float n_M_min = min(M_cap, n.M);

    // Compute p_n^
    MaterialOptimized matGI = CreateMaterialOptimized(materials[c.mID2], c.mID2);
    float j_gi = Jacobian_Reconnection(sample_c, sample_n, c.xn, c.nn, c.Vn);
    float p_n_from = (LinearizeVector(
        GetP_Hat_GI(sample_n.x1, sample_n.n1,
                    c.xn, c.nn,
                    c.E3, c.Vn,
                    sample_n.o, matOpt, matGI, true)
    ) * j_gi);

    float denom = c_M_min * p_c + n_M_min * p_n_from;

    // Non-defensive fraction for the canonical sample
    if (denom > 0.0f)
    {
        return (c_M_min * p_c) / denom;
    }
    else
    {
        return 0.0f;
    }
}

// Non-defensive version of the neighbor sample weight
float GenPairwiseMIS_noncanonical_temporal_GI(
    Reservoir_GI c,
    Reservoir_GI n,
    SampleData sample_c,
    SampleData sample_n,
    float M_sum,
    float M_cap,
    MaterialOptimized matOpt)
{
    // Clamped confidence weight for the canonical
    float c_M_min = min(M_cap, c.M);

    // p_c^
    MaterialOptimized mat_gi_c = CreateMaterialOptimized(materials[c.mID2], c.mID2);
    float p_c = LinearizeVector(
        GetP_Hat_GI(sample_c.x1, sample_c.n1,
                    c.xn, c.nn,
                    c.E3, c.Vn,
                    sample_c.o, matOpt, mat_gi_c, false)
    );

    float j_gi = Jacobian_Reconnection(sample_c, sample_n, c.xn, c.nn, c.Vn);

    float p_n_from = (LinearizeVector(
        GetP_Hat_GI(sample_n.x1, sample_n.n1,
                    c.xn, c.nn,
                    c.E3, c.Vn,
                    sample_n.o, matOpt, mat_gi_c, true)
    ) * j_gi);

    // Clamped confidence weight for the neighbor
    float n_M_min = min(M_cap, n.M);

    float denom = c_M_min * p_c + n_M_min * p_n_from;

    // Non-defensive fraction for the neighbor sample
    if (denom > 0.0f)
    {
        return clamp((n_M_min * p_n_from) / denom, 0.0f, 1.0f);
    }
    else
    {
        return 0.0f;
    }
}*/