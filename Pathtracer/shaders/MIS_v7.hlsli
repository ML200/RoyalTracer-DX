// Initial sampling
float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    return pdf_nee / (M1 * pdf_nee + M2 * pdf_bsdf);
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    return pdf_bsdf / (M1 * pdf_bsdf + M2 * pdf_nee);
}

// temporal reuse
// ReSTIR DI generalized MIS
float PairwiseMIS_canonical_Temp(
    float M_c,
    float p_c,
    float n_c,
    Reservoir_DI c,
    Reservoir_DI n,
    SampleData   sample_c,
    SampleData   sample_n,
    float        M_sum)
{
    float c_M_max = M_sum - M_c;
    float c_m_num = M_c * p_c;
    float m_c = M_c / M_sum;

    float m_den = c_m_num + (c_M_max * n_c);
    if (m_den > 0.0f)
    {
        float ratio  = (n.M_di / M_sum) * (c_m_num / m_den);
        m_c += ratio;
    }

    return m_c;
}


float PairwiseMIS_noncanonical_Temp(
    float M_c,
    float p_c,
    float n_c,
    Reservoir_DI c,
    Reservoir_DI n,
    SampleData   sample_c,
    SampleData   sample_n,
    float        M_sum)
{
    float m_num      = (M_sum - c.M_di) * n_c;
    float m_den      = m_num + (c.M_di * p_c);

    if (m_den > 0.0f) {
        float n_M_min = n.M_di;
        return (n_M_min / M_sum) * (m_num / m_den);
    }
    else {
        return 0.0f;
    }
}