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
//  TEMPORAL  DI
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
        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);
        // Get the reprojected pixel position
        uint tempPixelIdx = MapPixelID(dims, GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID));
        if(tempPixelIdx != uint(-1)){
            // Get the reprojected sample data
            SampleData sdata_r = loadSampleData(g_sample_last, tempPixelIdx);
            // Get the reprojected reservoir
            Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_last_di, tempPixelIdx);
            // Check wether the reservoir is valid for merge
            bool candidateAcceptedDI =
                (all(sdata_r.L1 < EPSILON) &&
                IsValidReservoir_DI(rdi_r) &&
                !RejectNormal_DI(sdata.n1, sdata_r.n1, 0.5f) &&
                !RejectDistance_DI(sdata.x1, sdata_r.x1, mul(viewI, float4(0, 0, 0, 1)).xyz, 0.1f) &&
                (sdata_r.matID == sdata.matID));

            // Merge the reservoirs
            if(candidateAcceptedDI){
                // Calculate the canonical target function
                float visReuse_c = rdi.W_di > 0.0f ? 1.0f : 0.0f;
                float p_c = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di)) * visReuse_c;
                float p_n = GetPHat(ReconnectDI(sdata_r.x1, sdata_r.n1, sdata_r.o, sdata_r.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di));// * VisibilityCheckCP(sdata_r.x1, rdi.x2_di, sdata_r.n1); // would require last frame AS and we dont store it, can be ommited for minimal added bias
                float n_c = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di)) * VisibilityCheckCP(sdata.x1, rdi_r.x2_di, sdata.n1);
                float visReuse = rdi_r.W_di > 0.0f ? 1.0f : 0.0f;
                float n_n = GetPHat(ReconnectDI(sdata_r.x1, sdata_r.n1, sdata_r.o, sdata_r.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di)) * visReuse;
                float M_c = min(TEMP_MCAP_DI,rdi.M_di);
                float M_n = min(TEMP_MCAP_DI,rdi_r.M_di);
                float M_sum = M_c + M_n;
                // Calculate the MIS weights
                float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_n, M_sum);
                float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_c, n_n, M_sum);

                // Calculate the reservoirs weights
                float w_c = mis_c * p_c * rdi.W_di;
                float w_n = mis_n * n_c * rdi_r.W_di;

                // Adjust wsum of the existing reservoir
                rdi.w_sum_di = w_c;

                // Update the reservoir
                // Get a random seed
                uint2 seed = GetSeed(pixelIdx, time, 2);
                float p_hat_final = p_c;
                if(UpdateReservoirDI(rdi, w_n, rdi_r.M_di, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, rdi_r.objID_di, seed)){
                    p_hat_final = n_c;
                }

                // Calculate new W
                //float p_hat = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di));
                if (p_hat_final > EPSILON && rdi.w_sum_di > EPSILON && rdi.w_sum_di < 1e10f) {
                    float W = rdi.w_sum_di / p_hat_final;
                    // NaN/Inf protection
                    if (isnan(W) || isinf(W)) {
                        W = 0.0f;
                    }
                    rdi.W_di = W;
                }
                else
                    rdi.W_di = 0.0f;

                // Store the merged reservoir
                store_x2_di(rdi.x2_di, g_Reservoirs_current_di, pixelIdx, rdi.objID_di);
                store_n2_di(rdi.n2_di, g_Reservoirs_current_di, pixelIdx, rdi.objID_di);
                store_L2_di(rdi.L2_di, g_Reservoirs_current_di, pixelIdx);
                store_W_di(rdi.W_di, g_Reservoirs_current_di, pixelIdx);
                store_M_di(rdi.M_di, g_Reservoirs_current_di, pixelIdx);
                store_objID_di(rdi.objID_di, g_Reservoirs_current_di, pixelIdx);
            }
        }
    }
}
