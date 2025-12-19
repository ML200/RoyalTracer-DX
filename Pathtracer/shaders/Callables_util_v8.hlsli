struct LightTreePdfPayload
{
    float3 prev_x; // On Input: Position. On Output: .x holds the PDF result
    float3 prev_n;
    uint   lightID;
    uint   objID;
};