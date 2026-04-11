/******************************************************************************
 * Pass_npg_debug.slang — Neural Path Guiding: training quality visualization
 *
 * Shows the learned mixture PDF evaluated at the GI reservoir direction.
 * This is the paper's primary metric: higher PDF at the actual light
 * direction = better variance reduction.
 *
 * Visualization:
 *   - Brightness: log(PDF+1) at the GI direction (brighter = better learned)
 *   - Red channel: dominant component weight (peaked = confident)
 *   - Green channel: PDF match quality
 *   - Blue channel: selection probability (how much NPG is trusted)
 *****************************************************************************/
#define COMPUTE_PASS
#include "Includes_v8.slang"
#include "NPG_common.slang"

RWByteAddressBuffer g_npgWeights : register(u24);

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    uint2 imgSize = uint2(gImageWidth, gImageHeight);
    uint pixelIdx = MapPixelID(imgSize, DTid.xy);

    float3 debugColor = float3(0, 0, 0);

    bool isEmitter = load_isEmitter(g_sample_current, pixelIdx);
    uint instID = load_instID(g_sample_current, pixelIdx);

    if (!isEmitter && instID != 0xFFFFFFFFu)
    {
        uint primID = load_primID(g_sample_current, pixelIdx);
        float2 bary = load_bary(g_sample_current, pixelIdx);
        float3 camPos = mul(viewI, float4(0, 0, 0, 1)).xyz;
        float3 surfPos = ReconstructPosition(instID, primID, bary);
        float3 surfNormal = load_n1_s_with_instID(g_sample_current, pixelIdx, instID);
        float3 wo = normalize(camPos - surfPos);

        // Build tangent frame
        float3 T, B;
        NPG_BuildTBN(surfNormal, T, B);
        float3 wo_local = NPG_WorldToLocal(wo, T, B, surfNormal);

        // Evaluate model
        float npgInput[NPG_INPUT_DIM];
        NPG_BuildInput(surfPos, wo_local, 0.5f, npgInput);
        NASGMixture mix = NPG_Forward(npgInput, g_npgWeights);

        // Evaluate PDF at the GI reservoir direction (the training target)
        Reservoir_GI rgi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
        float3 giDir = normalize(rgi.x2_gi - surfPos);

        if (!any(isnan(giDir)))
        {
            float3 giDir_local = NPG_WorldToLocal(giDir, T, B, surfNormal);
            if (giDir_local.z > 0.0f)
            {
                float pdfAtGI = NASGMixture_PDF(mix, normalize(giDir_local));

                // log1p scaling for visibility (PDF can range from 0 to very large)
                float logPdf = log(1.0f + pdfAtGI * 2.0f * NPG_PI);

                // Find dominant component weight
                float maxWeight = 0.0f;
                [unroll] for (uint i = 0; i < NPG_NUM_COMPONENTS; ++i)
                    maxWeight = max(maxWeight, mix.comps[i].weight);

                // R: concentration (how peaked the dominant component is)
                // G: PDF quality (main signal — brighter = better learned)
                // B: selection probability (trust level)
                debugColor.r = saturate(maxWeight * NPG_NUM_COMPONENTS - 1.0f); // 0 if uniform, 1 if single dominant
                debugColor.g = saturate(logPdf / 3.0f); // normalized log PDF
                debugColor.b = mix.selectionProb;
            }
        }
    }

    gOutput[uint3(DTid.xy, 3)] = float4(debugColor, 1.0f);
}
