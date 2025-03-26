#pragma warning(disable: 1234)

#include "Common_v6.hlsl"
#include "Reservoir_v6.hlsl"

#define TILE_WIDTH 8
#define TILE_HEIGHT 8

// Raytracing output texture, accessed as a UAV
RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

RWStructuredBuffer<Reservoir_DI> g_Reservoirs_current : register(u2);
RWStructuredBuffer<Reservoir_DI> g_Reservoirs_last : register(u3);
RWStructuredBuffer<Reservoir_GI> g_Reservoirs_current_gi : register(u4);
RWStructuredBuffer<Reservoir_GI> g_Reservoirs_last_gi : register(u5);
RWStructuredBuffer<SampleData> g_sample_current : register(u6);
RWStructuredBuffer<SampleData> g_sample_last : register(u7);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);

#include "GGX_v6.hlsl"
#include "Lambertian_v6.hlsl"
#include "BRDF_v6.hlsl"
#include "Sampler_v6.hlsl"
#include "MIS_v6.hlsl"
#include "Path_Sampler_v6.hlsl"

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
    float2 dims = float2(DispatchRaysDimensions().xy);
    uint pixelIdx = MapPixelID(dims, launchIndex);

    // #DXR Extra: Perspective Camera
    float aspectRatio = dims.x / dims.y;

    // Initialize the ray origin and direction
    float3 init_orig = mul(viewI, float4(0, 0, 0, 1)).xyz;


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
    uint2 seed;
    //_______________________________SETUP__________________________________
    // Every sample recieves a new seed
    seed.x = launchIndex.y * prime1_x ^ launchIndex.x * prime2_x ^ uint(1) * prime3_x ^ uint(time) * prime_time_x;
    seed.y = launchIndex.x * prime1_y ^ launchIndex.y * prime2_y ^ uint(1) * prime3_y ^ uint(time) * prime_time_y;

    // Calculate the initial ray direction. Jitter is used to randomize the rays intersection point on subpixel level
    float jitterX = 0.0f;//RandomFloat(seed);
    float jitterY = 0.0f;//RandomFloat(seed);
    float2 d = (((launchIndex.xy + float2(jitterX, jitterY)) / dims.xy) * 2.f - 1.f);
    float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 init_dir = mul(viewI, float4(target.xyz, 0)).xyz;


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
    HitInfo payload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    // Use a more memory efficient and aligned material model with lower precision
    uint mID = payload.materialID;
    bool performSampling = true;
    if(length(materials[mID].Ke) > 0.0f){
        performSampling = false;
    }

    MaterialOptimized matOpt = {
        materials[mID].Kd, materials[mID].Pr_Pm_Ps_Pc,
        materials[mID].Ks, materials[mID].Ke, mID
    };

    if(mID == 4294967294)
        matOpt = g_DefaultMissMaterial;

	//_______________________________RESERVOIRS__________________________________
    Reservoir_DI reservoir = {
        /* Row 0: */ float3(0.0f, 0.0f, 0.0f), 0.0f,
        /* Row 1: */ float3(0.0f, 0.0f, 0.0f), 0.0f,
        /* Row 2: */ { float3(0.0f, 0.0f, 0.0f), uint16_t(0), uint16_t(0)}
    };

    Reservoir_GI reservoir_GI = {
        float3(0.0f, 0.0f, 0.0f), // xn
        float3(0.0f, 0.0f, 0.0f), // nn
        float3(0.0f, 0.0f, 0.0f), // Vn
        0,                       // k
        0,                       // mID2
        0.0f,                    // w_sum
        0.0f,                    // W
        float3(0.0f, 0.0f, 0.0f), // f
        0,                       // M
        0,                       // s
        float3(0.0f, 0.0f, 0.0f),  // E3
        1.0f, //j
        uint2(0, 0)              // seed
    };

    SampleData sdata = {
        float3(0, 0, 0),  // x1
        mID,                  // mID
        float3(0, 0, 0),  // n1
        matOpt.Ke,   // L1
        float3(0, 0, 0),   // o
    };

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
            4,
            1,
            -direction,
            reservoir,
            payload,
            matOpt,
            seed
            );
        //_______________________________VISIBILITY_PASS__________________________________
        sdata.x1 = payload.hitPosition;
        sdata.n1 = normalize(payload.hitNormal);
        sdata.o = -direction;
        sdata.mID = mID;

        float p_hat = GetP_Hat(sdata.x1, sdata.n1, reservoir.x2, reservoir.n2, reservoir.L2, sdata.o, reservoir.s, matOpt, true);
        reservoir.W = GetW(reservoir, p_hat);

        //for(int p = 0; p< 20000; p++)
            //p_hat = GetP_Hat(sdata.x1, sdata.n1, reservoir.x2, reservoir.n2, reservoir.L2, sdata.o, reservoir.s, matOpt, true);

        // Perform path sampling (simpliefied for now)
        SamplePathSimple(reservoir_GI, payload.hitPosition, payload.hitNormal, -direction, matOpt, seed);
        reservoir_GI.M = 1.0f;
    }
	g_Reservoirs_current[pixelIdx] = reservoir;
    g_Reservoirs_current_gi[pixelIdx] = reservoir_GI;
    g_sample_current[pixelIdx] = sdata;
}
