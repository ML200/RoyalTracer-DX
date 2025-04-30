float MIS_Initial_NEE(float pdf_nee, float pdf_bsdf, float M1, float M2){
    return pdf_nee / (M1 * pdf_nee + M2 * pdf_bsdf);
}

float MIS_Initial_BSDF(float pdf_nee, float pdf_bsdf, float M2, float M1){
    return pdf_bsdf / (M1 * pdf_bsdf + M2 * pdf_nee);
}