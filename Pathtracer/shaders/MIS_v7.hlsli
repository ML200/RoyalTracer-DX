// Initial sampling
float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    return pdf_nee / (M1 * pdf_nee + M2 * pdf_bsdf);
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    return pdf_bsdf / (M1 * pdf_bsdf + M2 * pdf_nee);
}

// defensive pair-wise MIS, canonical sample
float PairwiseMIS_Canonical_Temp(
    float M_c,      // multiplicity / confidence of canonical
    float M_n,      // multiplicity / confidence of neighbour
    float p_c,      // p̂_c(y_c)
    float p_n,      // p̂←n(y_c)  **was called n_c before**
    float M_sum)    // = M_c + M_n
{
    float num   = M_c * p_c;                  // M_c p_c
    float denom = num + M_n * p_n;            // M_c p_c + M_n p_n

    float m_c = M_c / M_sum;                  // 1st term of Eq. (7.6)
    if (denom > 0.0f)
        m_c += (M_n / M_sum) * (num / denom); // 2nd term

    return m_c;                               // m_c ∈ [0,1]
}

// defensive pair-wise MIS, neighbour sample
float PairwiseMIS_Neighbour_Temp(
    float M_c,
    float M_n,
    float p_c,      // p̂_c(y_c)
    float n_n,      // p̂_n(y_n)  **new variable**
    float M_sum)
{
    float num   = M_n * n_n;                  // M_n n_n
    float denom = num + M_c * p_c;            // M_n n_n + M_c p_c

    return (denom > 0.0f)
           ? (M_n / M_sum) * (num / denom)
           : 0.0f;
}

// ──────────────────────────────────────────────────────────────────────
// Non-defensive pair-wise MIS   –   canonical sample  (y_c)
// m_c = (M_c · p_c) / (M_c · p_c + M_n · p_n)
// ──────────────────────────────────────────────────────────────────────
float PairwiseMIS_Canonical_Temp_NonDef(
    float M_c,      // multiplicity / confidence of canonical
    float M_n,      // multiplicity / confidence of neighbour
    float p_c,      // p̂_c (y_c)
    float p_n,
    float M_sum)      // p̂←n (y_c)
{
    float num   = M_c * p_c;
    float denom = num + M_n * p_n;

    return (denom > 0.0f) ? (num / denom) : 0.0f;   // 0 ≤ m_c ≤ 1
}


// ──────────────────────────────────────────────────────────────────────
// Non-defensive pair-wise MIS   –   neighbour sample  (y_n)
// m_n = (M_n · n_n) / (M_n · n_n + M_c · n_c)
// ──────────────────────────────────────────────────────────────────────
float PairwiseMIS_Neighbour_Temp_NonDef(
    float M_c,
    float M_n,
    float n_c,      // p̂_c (y_n)   ← canonical PDF at neighbour point
    float n_n,
    float M_sum)      // p̂_n (y_n)   ← neighbour’s own PDF
{
    float num   = M_n * n_n;
    float denom = num + M_c * n_c;

    return (denom > 0.0f) ? (num / denom) : 0.0f;   // 0 ≤ m_n ≤ 1
}
