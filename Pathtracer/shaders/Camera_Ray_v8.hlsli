/*
Camera ray operations, optimized
*/

//Initial ray origin
float3 InitOrigin(){
    return mul(viewI, float4(0, 0, 0, 1)).xyz;
}

//Initial ray direction
float3 InitDirection(uint2 pixel, uint2 imgSize){
    float2 d = (((pixel) / float2(imgSize)) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

//Pixel idx for directly writing into the sample data -> more efficient
SampleData SampleCameraRay(uint idx, uint2 pixel, uint2 imgSize){
    RayDesc ray;
    ray.Origin = InitOrigin();
    ray.Direction = InitDirection(pixel, imgSize);
    ray.TMin = 0.00001;
    ray.TMax = 10000;

    // Trace the camera ray
    HitInfo payload;
    TraceRayInline_HitInfo(SceneBVH, ray, payload, RAY_FLAG_NONE, 0xFF);

    // If we hit a backface, the surface doesnt emit light per definition
    float3 ke = payload.hitBackface ? 0.0f : materials[payload.materialID].Ke;

    //if we miss, ke is the position instead.
    if(payload.materialID == 0xFFFFFFFFu)
        ke = payload.hitPosition;

    SampleData sdata = (SampleData)0;
    sdata.x1 = payload.hitPosition;
    sdata.n1_s = payload.hitNormal;
    sdata.n1_g = payload.hitGNormal;
    sdata.L1 = ke;
    sdata.o = -ray.Direction;
    sdata.objID = payload.objID;
    sdata.matID = payload.materialID;

    storeSampleData(g_sample_current, idx, sdata);

    //return the sample data
    return sdata;
}




