#include "Includes_raygen_v8.hlsli"

// Simple wrapper for making LT_Pdf_LightTree_Area callable
[shader("callable")]
void Call_BSDF_SampleEval_v8(inout CALL_BSDF_EVAL_PAYLOAD params)
{

    SamplingP sp = CalculateStrategyProbabilities(
            params.matID, o, params.n_s,
            params.iors.x, params.iors.y, params.localKd, params.localPm
        );

    if(params.bs){
        params.s = SampleBRDF(
                               sp, params.matID, params.o, params.n_s, params.n_g,
                               params.localKd, params.localPr, params.localPm,
                               seed, params.iors.x, params.iors.y, params.iorpointer
                           );
    }

    BrdfData bdata = EvaluateAndPdf_COMBINED(
                sp, params.matID, params.n_s, params.n_g, params.s, params.o,
                params.localKd, params.localPr, params.localPm, params.iors.x, params.iors.y
            );

    // Return s and bdata output
    params.localPm = bdata.val;
    params.localPr = bdata.pdf;
}