#include "Common_v6.hlsl"
#include "GGX_v6.hlsl"
#include "Lambertian_v6.hlsl"
#include "BRDF_v6.hlsl"
#include "Reservoir_v6.hlsl"

// Raytracing output texture, accessed as a UAV
RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

RWStructuredBuffer<Reservoir> g_Reservoirs_current : register(u2);
RWStructuredBuffer<Reservoir> g_Reservoirs_last : register(u3);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);

#include "Sampler_v6.hlsl"
#include "MIS_v6.hlsl"

// #DXR Extra: Perspective Camera
cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;        // Previous frame's view matrix (can be removed if not used elsewhere)
    float4x4 prevProjection;  // Previous frame's projection matrix (can be removed if not used elsewhere)
    float time;
    float4 frustumLeft;       // Left frustum plane
    float4 frustumRight;      // Right frustum plane
    float4 frustumTop;        // Top frustum plane
    float4 frustumBottom;     // Bottom frustum plane
    float4 prevFrustumLeft;   // Previous frame's frustum planes (can be removed if not used elsewhere)
    float4 prevFrustumRight;
    float4 prevFrustumTop;
    float4 prevFrustumBottom;
}

//Generate the initial
[shader("raygeneration")]
void RayGen() {
    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint pixelIdx = launchIndex.y * DispatchRaysDimensions().x + launchIndex.x;
    float2 dims = float2(DispatchRaysDimensions().xy);

    // #DXR Extra: Perspective Camera
    float aspectRatio = dims.x / dims.y;

    // Initialize the ray origin and direction
    float3 init_orig = mul(viewI, float4(0, 0, 0, 1));


    // SEEDING
    const uint prime1_x = 73856093u;
    const uint prime2_x = 19349663u;
    const uint prime3_x = 83492791u;
    const uint prime1_y = 37623481u;
    const uint prime2_y = 51964263u;
    const uint prime3_y = 68250729u;
    const uint prime_time_x = 293803u;
    const uint prime_time_y = 423977u;

    // Initialize once, to reduce allocs with several samples per frame
    HitInfo payload;
    uint2 seed;


    //_______________________________SETUP__________________________________
    // Every sample recieves a new seed
    seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(samples + 1) * prime3_x ^ uint(time) * prime_time_x;
    seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(samples + 1) * prime3_y ^ uint(time) * prime_time_y;

    // Calculate the initial ray direction. Jitter is used to randomize the rays intersection point on subpixel level
    float jitterX = 0.0f;//RandomFloat(seed);
    float jitterY = 0.0f;//RandomFloat(seed);
    float2 d = (((launchIndex.xy + float2(jitterX, jitterY)) / dims.xy) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 init_dir = mul(viewI, float4(target.xyz, 0));


    //_________________________PATH_VARIABLES______________________________
    float3 direction = normalize(init_dir);
    float3 origin = init_orig;


    //____________________________CAMERA_RAY________________________________
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0001;
    ray.TMax = 10000;

    // Trace the camera ray
    bool performSampling = true;

    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    Material hitMaterial;
    if(payload.materialID != 4294967294){
        hitMaterial = materials[payload.materialID];
        if(length(materials[payload.materialID].Ke) > 0.0f){
            performSampling = false;
        }
    }
    else{
        hitMaterial = g_DefaultMissMaterial;
        performSampling = false;
    }

	//_______________________________RESERVOIRS__________________________________
    Reservoir reservoir = {
        (float3)0.0f,  // x1
        (float3)0.0f,  // n1
        (float3)0.0f,  // x2
        (float3)0.0f,  // n2
        (float)0.0f,  // w_sum
        (float)0.0f,  // p_hat of the stored sample
        (float)0.0f,  // W
        (float)0.0f,  // M
        (float)0.0f,    // V
        (float3)0.0f,  // final color (L1), mostly 0
        (float3)0.0f,  // reconnection color (L2)
        (uint)0,  // s
        (float3)0.0f,  //o
        (uint)0  // mID
    };
	reservoir.x1 = payload.hitPosition;
	reservoir.n1 = payload.hitNormal;
	reservoir.L1 = (float3)hitMaterial.Ke;
	reservoir.o = -direction;
	reservoir.mID = payload.materialID;

    //_______________________________PATH_SAMPLING__________________________________
    // Perform y bounces
    if(performSampling){
        /*
        Pathtracing concept:
        To support advanced algorithms like ReSTIR, the path tracing will be always structured the same. Most of the work will be
        done in the raygeneration shader. Sampling works always the same:
        - Evaluate direct NEE: Perform light sampling using NEE. This can be done with or without visibility check which will significantly speed up RIS
        - Evaluate direct BSDF: Sample the BSDF to evaluate direct light contribution
        - Evaluate indirect BSDF: Sample the BSDF to trace a ray with several bounces accumulating the pdf (ReSTIR GI/PT, later)
        */
        //_______________________________RIS_DIRECT_ILLUMINATION__________________________________
        SampleRIS(
            10,
            1,
            -direction,
            reservoir,
            payload,
            seed
            );
        //_______________________________VISIBILITY_PASS__________________________________
        //for(int v = 0; v < 9000; v++){
        if(VisibilityCheck(reservoir.x1, reservoir.n1, normalize(reservoir.x2-reservoir.x1), length(reservoir.x2-reservoir.x1)) == 0.0f){
            reservoir.p_hat = 0.0f;
        }//}
    }
	g_Reservoirs_current[pixelIdx] = reservoir;
}
