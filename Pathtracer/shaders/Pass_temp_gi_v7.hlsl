cbuffer Push : register(b1)
{
    uint2 gImageSize;
}
#define ENABLE_RAY_QUERY_INLINE // Activate support for inline ray tracing

#define gImageWidth  (gImageSize.x)
#define gImageHeight (gImageSize.y)
#define IMG_W        (gImageSize.x)
#define IMG_H        (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di  : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di     : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi  : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi     : register(u5);
RWByteAddressBuffer g_InitialBSDFRays : register(u9);

StructuredBuffer<STriVertex>          BTriVertex        : register(t2);
StructuredBuffer<int>                 indices           : register(t1);
RaytracingAccelerationStructure       SceneBVH          : register(t0);
StructuredBuffer<InstanceProperties>  instanceProps     : register(t3);
StructuredBuffer<uint>                materialIDs       : register(t4);
StructuredBuffer<Material>            materials         : register(t5);
StructuredBuffer<LightTriangle>       g_EmissiveTriangles : register(t6);
StructuredBuffer<float>               g_AliasProb       : register(t7);
StructuredBuffer<uint>                g_AliasIdx        : register(t8);

// Needs access to all structured/random buffers
#include "Sample_data.hlsli"
#include "Initial_bsdf.hlsli"
#include "GGX_v7.hlsli"
#include "Lambertian_v7.hlsli"
#include "BSDF_v7.hlsli"

cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float time;
}
// These includes need access to ALL previous buffers
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "Camera_ray_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  GI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Load the sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    if(all(sdata.L1 < EPSILON)){
        // Load current reservoir
        Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 2);

        // If the material is below a certain roughness, do reprojection of the reflected point in buffer + objID)
        uint tempPixelIdx;
        if(materials[sdata.matID].Pr_Pm_Ps_Pc.x <= 0.0f){
            float3 x_init = load_x2_init(g_InitialBSDFRays, pixelIdx);
            uint objID_init = load_objID_init(g_InitialBSDFRays, pixelIdx);
            tempPixelIdx = MapPixelID(dims, GetBestReprojectedPixel_s(x_init, objID_init, prevView, prevProjection, dims));
        }
        else{
            // Get the reprojected pixel position
            tempPixelIdx = MapPixelID(dims, GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID, seed.x));
        }
        if(tempPixelIdx != 0xFFFFFFFF){
            // Get the reprojected sample data
            SampleData sdata_r = loadSampleData(g_sample_last, tempPixelIdx);
            // Get the reprojected reservoir
            Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_last_gi, tempPixelIdx);
            // Check wether the reservoir is valid for merge
            bool candidateAcceptedDI =
                (all(sdata_r.L1 < EPSILON) &&
                IsValidReservoir_GI(rdi_r) &&
                !RejectNormal_GI(sdata.n1, sdata_r.n1, 0.5f) &&
                !RejectDistance_GI(sdata.x1, sdata_r.x1, sdata.n1, 0.02f) &&
                //!(rdi.W_gi > 1.0f/1e-10 || rdi.W_gi < 1e-10) &&
                //!RejectLength_GI(rdi.x2_gi, rdi.n2_gi, rdi_r.x2_gi, rdi_r.n2_gi, sdata.x1, EPSILON) &&
                //!RejectDistance_GI(sdata.x1, sdata_r.x1, mul(viewI, float4(0, 0, 0, 1)).xyz, 0.1f) &&
                (sdata_r.matID == sdata.matID));

            // Merge the reservoirs
            if(candidateAcceptedDI){
                // Calculate the canonical target function
                float visReuse_c = rdi.W_gi > 0.0f ? 1.0f : 0.0f;
                float p_c = GetPHat(ReconnectGI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.matID_gi, rdi.x2_gi, rdi.n2_gi, rdi.L2_gi, rdi.V2_gi)) * visReuse_c;
                float p_n = GetPHat(ReconnectGI(sdata_r.x1, sdata_r.n1, sdata_r.o, sdata_r.matID, rdi.matID_gi, rdi.x2_gi, rdi.n2_gi, rdi.L2_gi, rdi.V2_gi));// * VisibilityCheckCP(sdata_r.x1, rdi.x2_gi, sdata_r.n1); // would require last frame AS and we dont store it, can be ommited for minimal added bias
                float n_c = GetPHat(ReconnectGI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_gi, rdi_r.L2_gi, rdi_r.V2_gi)) * VisibilityCheckCP(sdata.x1, rdi_r.x2_gi, sdata.n1);
                float visReuse = rdi_r.W_gi > 0.0f ? 1.0f : 0.0f;
                float n_n = GetPHat(ReconnectGI(sdata_r.x1, sdata_r.n1, sdata_r.o, sdata_r.matID, rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_gi, rdi_r.L2_gi, rdi_r.V2_gi)) * visReuse;
                float M_c = min(TEMP_MCAP_GI,rdi.M_gi);
                float M_n = min(TEMP_MCAP_GI,rdi_r.M_gi);
                float M_sum = M_c + M_n;
                // Calculate the MIS weights
                float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_n, M_sum);
                float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_c, n_n, M_sum);

                // Calculate the reservoirs weights
                float w_c = mis_c * p_c * rdi.W_gi;
                float w_n = mis_n * n_c * rdi_r.W_gi;

                // Adjust wsum of the existing reservoir
                rdi.w_sum_gi = w_c;

                // Update the reservoir
                float p_hat_final = p_c;
                if(UpdateReservoirGI(rdi, w_n, rdi_r.M_gi, rdi_r.x2_gi, rdi_r.n2_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.matID_gi, rdi_r.objID_gi, seed)){
                    p_hat_final = n_c;
                }

                // Calculate new W
                //float p_hat = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di));
                if (p_hat_final > 0.0f && rdi.w_sum_gi > 0.0f) {
                    float W = rdi.w_sum_gi / p_hat_final;
                    // NaN/Inf protection
                    if (isnan(W) || isinf(W)) {
                        W = 0.0f;
                    }
                    rdi.W_gi = W;
                }
                else
                    rdi.W_gi = 0.0f;

                // Store the merged reservoir
                storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
                /*store_x2_gi(rdi.x2_gi, g_Reservoirs_current_gi, pixelIdx, rdi.objID_gi);
                store_n2_gi(rdi.n2_gi, g_Reservoirs_current_gi, pixelIdx, rdi.objID_gi);
                store_L2_gi(rdi.L2_gi, g_Reservoirs_current_gi, pixelIdx);
                store_V2_gi(rdi.V2_gi, g_Reservoirs_current_gi, pixelIdx);
                store_W_gi(rdi.W_gi, g_Reservoirs_current_gi, pixelIdx);
                store_M_gi(rdi.M_gi, g_Reservoirs_current_gi, pixelIdx);
                store_objID_gi(rdi.objID_gi, g_Reservoirs_current_gi, pixelIdx);
                store_matID_gi(rdi.matID_gi, g_Reservoirs_current_gi, pixelIdx);*/
            }
        }
    }
}