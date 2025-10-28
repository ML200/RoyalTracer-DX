// Corrected Reconnection for x1 -> x2 -> x3
inline float3 ReconnectGIBD_Simple(
    in float3 x1, in float3 n1, in float3 o, in uint mID1,
    in float3 x2, in float3 n2, in uint mID2,
    in float3 x3, in float3 n3, in float3 L3,
    in float pdf, // pdf in SOLID ANGLE at x1 toward x2
    inout float3 first_segment // The x1 x2 segment throughput (without pdf) for path advancement
) {
    // Check if the provided subpath is valid
    if(
    !(length(n1)>0.0f) ||
    !(length(n2)>0.0f) ||
    !(length(n3)>0.0f) ||
    !(length(x1-x2)>0.0f) ||
    !(length(x2-x3)>0.0f)
    )
    {first_segment = 0.0f; return float3(0,0,0);}

    // Directions
    float3 w12 = normalize(x2 - x1);
    float3 w21 = -w12;
    float3 w23 = normalize(x3 - x2);
    float3 w32 = -w23;

    // Distances
    float r12_2 = dot(x2 - x1, x2 - x1);
    float r23_2 = dot(x3 - x2, x3 - x2);

    // Cosines
    float cos1     = max(dot(n1, w12), 0.0f);
    float cos2_12  = max(dot(n2, w21), 0.0f); // cos at x2 for incoming from x1
    float cos2_out = max(dot(n2, w23), 0.0f); // cos at x2 for outgoing to x3
    float cos3     = max(dot(n3, w32), 0.0f); // cos at x3 for outgoing to x2

    // BSDFs
    float3 f1 = BSDF_term(mID1, n1, w21, o);
    float3 f2 = BSDF_term(mID2, n2, w32, w21);

    // Guards
    if (pdf <= 0.0f || r12_2 == 0.0f || r23_2 == 0.0f)
        return 0.0f;

    // Geometric terms
    float G12 = cos1;
    float G23 = cos2_out;

    // Full throughput
    float3 throughput = f1 * G12 * f2 * G23 * L3;

    // Contribution
    float3 contrib = throughput / pdf;

    first_segment = f1 * G12;

    return any(isnan(contrib)) ? 0.0f.xxx : contrib;
}