struct CALL_LT_PDF_PAYLOAD
{
    float3 prev_x; // On Input: Position. On Output: .x holds the PDF result
    float3 prev_n;
    uint   lightID;
    uint   objID;
};

struct CALL_BSDF_EVAL_PAYLOAD{
    bool bs; // should the function also perform sampling?
    float2 iors;
    float3 localKd;
    float localPm;
    float localPr;
    float3 n_s;
    float3 n_g;
    float3 s; // return as well
    float3 o;
    uint matID;
    uint iorpointer;
    uint seed:
};