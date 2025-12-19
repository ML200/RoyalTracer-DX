#include "Includes_raygen_v8.hlsli"

// Simple wrapper for making LT_Pdf_LightTree_Area callable
[shader("callable")]
void LT_Pdf_v8(inout LightTreePdfPayload params)
{
    float result = LT_Pdf_LightTree_Area(
        params.prev_x,
        params.prev_n,
        params.lightID,
        params.objID
    );
    params.prev_x.x = result;
}