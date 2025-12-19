#include "Includes_raygen_v8.hlsli"

// Simple wrapper for making LT_Pdf_LightTree_Area callable
[shader("callable")]
void Call_LT_Pdf_v8(inout CALL_LT_PDF_PAYLOAD params)
{
    float result = LT_Pdf_LightTree_Area(
        params.prev_x,
        params.prev_n,
        params.lightID,
        params.objID
    );
    params.prev_x.x = result;
}