#pragma once
// Snapshot the trace result into ordinary values immediately. In particular, do
// not carry/reassign the driver's opaque HitObject through a volume-retrace loop
// or across a later reorder point. Traversal and alpha-test behavior are unchanged.
struct OceanPathHit {
    bool hit;
    float distance;
    uint instance;
    uint geometry;
    uint primitive;
    float2 barycentrics;
};
OceanPathHit OceanTracePathHit(RayDesc ray) {
    OceanPathHit result = (OceanPathHit)0;
    result.distance = ray.TMax;
    dx::HitObject hit = TraceRayChecked(SceneBVH,RAY_FLAG_FORCE_OMM_2_STATE,0xFF,ray);
    if (hit.IsHit()) {
        BuiltInTriangleIntersectionAttributes attr;
        hit.GetAttributes(attr);
        result.hit = true;
        result.distance = hit.GetRayTCurrent();
        result.instance = hit.GetInstanceID();
        result.geometry = hit.GetGeometryIndex();
        result.primitive = hit.GetPrimitiveIndex();
        result.barycentrics = attr.barycentrics;
    }
    return result;
}

// One actual path-scattering event between object bounces. No lighting/shadow
// queries here: the scattered ray continues to real geometry and the environment.
OceanFlight OceanSampleWaterSegment(float3 origin, float3 dir, float limit, bool used, inout uint seed) {
    OceanFlight f;
    f.distance = limit; f.weight = 1.0f; f.scattered = false;
    if (!OceanMediumEnabled()) return f;
    limit = OceanBoundaryDistance(origin,dir,limit);
    float3 sigmaA,sigmaS; float g;
    OceanMediumCoefficients(sigmaA,sigmaS,g);
    const float2 u = !used && any(sigmaS > 0.0f) ?
        float2(RandomFloatSingle(seed),RandomFloatSingle(seed)) : 0.0f;
    return OceanFreeFlight(sigmaA,sigmaS,limit,used,u);
}
float3 OceanSampleScatterDirection(float3 incoming, inout uint seed) {
    const float g = LoadPhaseG(OceanParams().materialBase);
    const float mu = OceanSamplePhaseCosine(g,RandomFloatSingle(seed));
    const float phi = 6.28318530718f*RandomFloatSingle(seed);
    const float r = sqrt(max(1.0f-mu*mu,0.0f));
    float3 tangent,bitangent;
    GetOrthoBasis(incoming,tangent,bitangent);
    // Phase value / phase PDF is one. Positive g continues along the incoming ray.
    return normalize(mu*incoming+r*(cos(phi)*tangent+sin(phi)*bitangent));
}
