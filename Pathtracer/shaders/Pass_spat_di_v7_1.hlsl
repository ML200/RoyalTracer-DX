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
RWTexture2DArray<half4> gScratchPing         : register(u8);

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
#include "Inline_RT.hlsli"
#include "Camera_ray_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL  DI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Load the sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    // Store SampleData temporally
    storeSampleData(g_sample_last, pixelIdx, sdata);

    if(all(sdata.L1 < EPSILON)){
        // Load current reservoir
        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 2);

        // ########################################### NODE #############################################################
        // Based on the quality of the current canonical sample, reduce the number of spatial reuses.
        float conf = min(60.0f, rdi.M_di) / TEMP_MCAP_DI;
        uint nbrBudget = SPAT_COUNT_MIN_DI +
                 uint((1.0f - conf) * float(SPAT_COUNT_MAX_DI - SPAT_COUNT_MIN_DI) + 0.5f);
        uint radiusBudget = SPAT_RAD_MIN +
                         uint((1.0f - conf) * float(SPAT_RAD_MAX - SPAT_RAD_MIN) + 0.5f);


        // Array to hold valid neighbor IDs
        uint nIds[SPAT_COUNT_MAX_DI];
        float M_sum = 0.0f;
        // loop over all candidates and select those that are best
        [unroll(SPAT_COUNT_MAX_DI)]
        for(uint i = 0; i < SPAT_COUNT_MAX_DI; i++){
            if(i < nbrBudget){
                // loop until we find a valid candidate
                nIds[i] = 0xFFFFFFFF; // Set the id to invalid for that neighbor, until a valid one is found
                [unroll(SPAT_TRIS_DI)]
                for(uint j = 0; j < SPAT_TRIS_DI; j++){
                    // Get the candidate ID
                    uint iID = GetRandomPixelCircleWeighted(radiusBudget, dims.x, dims.y, launchIndex.x, launchIndex.y, seed);
                    // Only load data required for the comparison (L2 + M)
                    Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_current_di, iID);
                    // Check wether the reservoir is valid for merge (Later, replace this with a weight -> the neighbor with the highest weight is selected)
                    bool candidateAcceptedDI =
                        IsValidReservoir_DI_opt(rdi_r.n2_di, rdi_r.M_di) &&
                        (all(load_L1(g_sample_current, iID) < EPSILON) &&
                        !RejectNormal_DI(sdata.n1, load_n1(g_sample_current, iID), 0.9f) &&
                        !RejectDistance_DI(sdata.x1, load_x1(g_sample_current, iID), mul(viewI, float4(0, 0, 0, 1)).xyz, 0.1f) &&
                        (load_matID(g_sample_current, iID) == sdata.matID));
                    if(candidateAcceptedDI){
                        nIds[i] = iID;
                        M_sum += min(SPAT_MCAP_DI, rdi_r.M_di);
                        break;
                    }
                }
            }
            else
                nIds[i] = 0xFFFFFFFF; // Fill the rest with invalid pixels -> fast loop over it (divergence, but fuck it)
        }
        // Reorganize the list of samples for better thread coherency
        uint writeIdx = 0;
        [unroll(SPAT_COUNT_MAX_DI)]
        for (uint readIdx = 0; readIdx < SPAT_COUNT_MAX_DI; ++readIdx)
        {
            uint id = nIds[readIdx];
            if (id != 0xFFFFFFFF)
                nIds[writeIdx++] = id;
        }
        [unroll(SPAT_COUNT_MAX_DI)]
        for (uint i = writeIdx; i < SPAT_COUNT_MAX_DI; ++i)
            nIds[i] = 0xFFFFFFFF;
        // ########################################### NODE #############################################################

        // Calculate M_sum for all valid candidates
        M_sum += min(SPAT_MCAP_DI, rdi.M_di);
        float M_c = min(SPAT_MCAP_DI, rdi.M_di);
        rdi.M_di = M_c;

        //float debug = 0.0f;

        // Calculate canonical pixel p_hat before loading the expensive data
        float visReuse = rdi.W_di > 0.0f ? 1.0f : 0.0f;
        float3 contrib_c = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di) * visReuse;
        float p_c = GetPHat(contrib_c);
        float3 contrib_final = contrib_c;
        // Compute the pairwise MIS weight for the canonical sample
        float mis_c = PairwiseMIS_Canonical_Spat_DI(M_sum, p_c, M_c, nIds, rdi.x2_di, rdi.n2_di, rdi.L2_di);
        //debug += mis_c;
        // Adjust the weight in the canonical reservoir
        rdi.w_sum_di = mis_c * p_c * rdi.W_di;

        // ########################################### NODE #############################################################
        // Iterate through all valid neighbors and update the canonical reservoir with them
        [unroll(SPAT_COUNT_MAX_DI)]
        for(int i = 0; i < SPAT_COUNT_MAX_DI; i++){
            if(nIds[i] != 0xFFFFFFFF){
                // Calculate p_hat for the neighbor using the canonical sample position
                Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_current_di, nIds[i]);
                float3 contrib_n = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di) * VisibilityCheckCP(sdata.x1, rdi_r.x2_di, sdata.n1);
                float p_hat_from = GetPHat(contrib_n);
                // Calculate the samples MIS weight
                float mis_n = PairwiseMIS_Neighbor_Spat_DI(M_sum, M_c, min(SPAT_MCAP_DI ,rdi_r.M_di), p_c, p_hat_from, nIds[i], rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di);
                //debug += mis_n;
                // Calculate the sample weight
                float w_n = mis_n * p_hat_from * rdi_r.W_di;

                // Update the reservoir
                if(UpdateReservoirDI(rdi, w_n, min(SPAT_MCAP_DI ,rdi_r.M_di), rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, rdi_r.objID_di, seed)){
                    contrib_final = contrib_n;
                }

            }
        }
        // ########################################### NODE #############################################################

        // Calculate new W
        //float p_hat = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di));
        float p_hat_final = GetPHat(contrib_final);
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
        storeReservoirDI(g_Reservoirs_last_di, pixelIdx, rdi);

        // Store the final output
        gScratchPing[uint3(tid.xy, 1)] = float4(contrib_final * rdi.W_di, 0);

        // DEBUG
        /*float3 heat;
        heat.r = step(debug, 0.9);          // red when <1
        heat.g = saturate(1 - abs(debug-1)); // green at exactly 1
        heat.b = step(1.1, debug);          // blue when >1
        gOutput[uint3(tid.xy, 0)] = float4(heat, 1);*/

    }
    else
        gScratchPing[uint3(tid.xy, 1)] = float4(sdata.L1, 0);
}