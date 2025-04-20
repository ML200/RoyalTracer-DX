/*
Camera ray operations, optimized
*/

//Initial ray origin
float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//Initial ray direction
float3 InitDirection(){
    float2 d = (((DispatchRaysIndex().xy) / float2(DispatchRaysDimensions().xy)) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

//Pixel idx for directly writing into the sample data -> more efficient
void SampleCameraRay(uint idx){
    RayDesc ray;
    ray.Origin = InitOrigin();
    ray.Direction = InitDirection();
    ray.TMin = 0.0001;
    ray.TMax = 10000;

    store_o(-ray.Direction, g_sample_current, idx);

    // Trace the camera ray
    HitInfo payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    // Compress and save relevant data: x1, L1, n1, mID and oID
    store_x1(payload.hitPosition, g_sample_current, idx);
    store_n1(payload.hitNormal, g_sample_current, idx);
    store_L1(materials[payload.materialID].Ke, g_sample_current, idx);

}