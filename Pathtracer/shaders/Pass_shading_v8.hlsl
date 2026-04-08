#include "Includes_v8.hlsli"


// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gOutput[uint3(DTid.xy, 0)] = float4(0, 0, 0, 0);

    float3 output_DI  = gScratchPing[uint3(DTid.xy, 1)];
    float3 output_GI  = gScratchPing[uint3(DTid.xy, 2)];
    float3 sunDirect  = gScratchPing[uint3(DTid.xy, 3)].rgb;

    float3 accumulation = output_DI + output_GI + sunDirect;

    bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const float MAX_SAMPLES     = 1000.0;

    float4 prev        = gPermanentData[DTid.xy];   // rgb = running avg, a = N
    float3 prevAvg     = prev.rgb;
    float  prevSamples = prev.a;

    float3 newAvg;
    float  newSamples;
    if (cameraChanged)
    {
        // Camera moved: reset running average and sample count
        newAvg     = accumulation;
        newSamples = 1.0h;
    }
    else
    {
        newSamples = min(prevSamples + 1.0h, MAX_SAMPLES);
        float invN  = 1.0h / newSamples;
        newAvg     = mad(accumulation - prevAvg, invN, prevAvg);
    }

    // store back
    gPermanentData[DTid.xy] = float4(newAvg, newSamples);
    gOutput[uint3(DTid.xy, 2)] = float4(newAvg, 1.0f);

    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, DTid.xy);

    gOutput[uint3(DTid.xy, 0)] = float4(accumulation, 1.0f);

    // Variables for bias hint (set in each branch, used after)
    uint  biasInstID;
    float2 biasMV = float2(0, 0);
    bool  isEmitterSurface = false;

    bool isEmissiveOrSky = load_isEmitter(g_sample_current, pixelIdx);
    if (isEmissiveOrSky)
    {
        // Emitters have a valid surface; sky sentinel (instID==0xFFFF) does not
        uint emInstID = load_instID(g_sample_current, pixelIdx);
        bool hasPosition = (emInstID != 0xFFFFu);

        if (hasPosition)
        {
            // Emitter surface: compute depth + motion vectors like regular geometry
            uint emPrimID = load_primID(g_sample_current, pixelIdx);
            float2 emBary = load_bary(g_sample_current, pixelIdx);
            float3 emPos  = ReconstructPosition(emInstID, emPrimID, emBary);
            g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(emPos);

            float2 curPix = DTid.xy;
            float2 prevPix = GetLastFramePixelCoordinates_Float(
                emPos, prevView, prevProjection, dims, emInstID) - jitter;
            bool validPrev = (prevPix.x >= 0 && prevPix.y >= 0 &&
                              prevPix.x < IMG_W && prevPix.y < IMG_H);
            float2 emMV = validPrev ? float2(prevPix - curPix) : float2(0, 0);
            g_dlssMVec[curPix] = emMV;
            biasMV = emMV;
            isEmitterSurface = true;

            // Provide real geometric normal so DLSS-RR can do proper edge detection
            float3 emNormal = load_n1_g_with_instID(g_sample_current, pixelIdx, emInstID);
            g_dlssNormals[DTid.xy] = float4(emNormal, 0.0f);
        }
        else
        {
            // Sky: no surface, clamped to cameraFar to stay within DLSS-RR's declared range
            g_dlssDepth[DTid.xy] = 10000.0f;
            g_dlssMVec[DTid.xy] = float2(0.0f, 0.0f);
            g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        biasInstID = emInstID;

        // Emitters are diffuse light sources — use roughness=1, diffuse albedo=1
        // to route through DLSS-RR's diffuse denoiser (not the specular denoiser,
        // which is view-dependent and causes direction-dependent trailing).
        g_dlssSpecularAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        g_dlssDiffuseAlbedo[DTid.xy] = float4(1.0f, 1.0f, 1.0f, 0.0f);
        g_dlssRoughness[DTid.xy] = 1.0f;
        g_dlssSpecHitDist[DTid.xy] = hasPosition ? 0.0f : 10000.0f;
        g_dlssSpecMVec[DTid.xy] = float2(0.0f, 0.0f);
        gOutput[uint3(DTid.xy, 4)] = float4(0, 0, 0, 1);
        gOutput[uint3(DTid.xy, 5)] = float4(0, 0, 0, 1);

        // Tonemap bright emitters before DLSS-RR to prevent ghosting/ringing
        float3 dlssColor = accumulation / (1.0f + accumulation);
        g_dlssInput[DTid.xy] = float4(saturate(dlssColor), 1.0f);
    }
    else{
        // Reconstruct surface for DLSS
        uint sInstID = load_instID(g_sample_current, pixelIdx);
        uint sPrimID = load_primID(g_sample_current, pixelIdx);
        float2 sBary = load_bary(g_sample_current, pixelIdx);
        SurfaceVertex sv = BuildVertex(sInstID, sPrimID, sBary, mul(viewI, float4(0, 0, 0, 1)).xyz);
        sv.etai = load_etai(g_sample_current, pixelIdx);
        sv.etat = load_etat(g_sample_current, pixelIdx);

        // DLSS RR input data:
        g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(sv.x);

        g_dlssNormals[DTid.xy] = float4(sv.n_s, 0.0f);
        g_dlssDiffuseAlbedo[DTid.xy] = float4(sv.Kd, 1.0f);
        g_dlssRoughness[DTid.xy] = sv.Pr;

        float2 curPix = DTid.xy;
        float2 prevPix = GetLastFramePixelCoordinates_Float(sv.x, prevView, prevProjection, dims, sInstID) - jitter;

        bool validPrev = (prevPix.x >= 0 && prevPix.y >= 0 && prevPix.x < IMG_W && prevPix.y < IMG_H);

        float2 mvPixels = validPrev ? float2(prevPix - curPix) : float2(0.0, 0.0);

        g_dlssMVec[curPix] = mvPixels;
        gOutput[uint3(DTid.xy, 10)] = float4(abs(mvPixels), 0.0f, 1.0f);

        biasInstID = sInstID;
        biasMV = mvPixels;

        // Specular albedo
        float3 specularAlbedo = EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, dot(sv.o, sv.n_s));
        g_dlssSpecularAlbedo[DTid.xy] = float4(specularAlbedo, 0.0f);

        // Post-spatial GI reservoir for stable hit distance
        Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_last_gi, pixelIdx);
        g_dlssSpecHitDist[DTid.xy] = length(rdi.x2_gi - sv.x);

        // ── Specular motion vector ───────────────────────────────────────
        // Uses the deterministic perfect-reflection ray traced in raygen (scratch slice 4).
        float specularity = Luma(specularAlbedo);
        float2 specMV = mvPixels; // default: surface motion vector
        bool validSpecReproj = false;
        {
            float4 reflData = gScratchPing[uint3(DTid.xy, 4)];
            uint   reflInstID = asuint(reflData.w);
            if (specularity > 0.04f && reflInstID < 0xFFFFFFFEu)
            {
                float2 prevRefl = GetLastFramePixelCoordinates_Float(
                    reflData.xyz, prevView, prevProjection, dims, reflInstID) - jitter;
                bool validRefl = (prevRefl.x >= 0 && prevRefl.y >= 0 &&
                                  prevRefl.x < IMG_W && prevRefl.y < IMG_H);
                if (validRefl)
                {
                    specMV = prevRefl - curPix;
                    validSpecReproj = true;
                }
            }
        }
        g_dlssSpecMVec[DTid.xy] = specMV;

        // Debug: slice 5 = spec MV divergence from surface MV
        gOutput[uint3(DTid.xy, 4)] = float4(specularity, specularity, specularity, 1.0f);
        float2 mvDiff = abs(specMV - mvPixels);
        gOutput[uint3(DTid.xy, 5)] = float4(saturate(mvDiff * 0.1f), validSpecReproj ? 1.0f : 0.0f, 1.0f);

        g_dlssInput[DTid.xy] = float4(accumulation, 1.0f);
    }

    // ── Bias hint: instance-ID-based disocclusion detection ──────────────────
    // Compare current instID with the previous frame's instID at the reprojected
    // pixel. Mismatch = different object = disocclusion → tell DLSS-RR to trust
    // the current frame (bias=1). This provides exact object identity info that
    // DLSS-RR cannot derive from depth/normals alone.
    {
        float disoccBias = 1.0f;  // default: trust current (off-screen, first frame)
        float2 reprojPrev = float2(DTid.xy) + biasMV;
        int2 prevPixI = int2(round(reprojPrev));
        if (all(prevPixI >= 0) && all(prevPixI < int2(dims))) {
            uint prevPixelIdx = MapPixelID(uint2(dims), prevPixI);
            uint prevInstID = load_instID(g_sample_last, prevPixelIdx);
            disoccBias = (biasInstID != prevInstID) ? 1.0f : 0.0f;
        }

        // Emitter surfaces have deterministic direct emission — they don't benefit
        // from temporal denoising. Always bias to current frame to prevent DLSS-RR
        // from accumulating bright emitter color that persists as trails when the
        // emitter moves.
        float emitterBias = isEmitterSurface ? 1.0f : 0.0f;

        float bias = max(disoccBias, emitterBias);
        g_dlssBiasHint[DTid.xy] = bias;

        // For disoccluded non-emitter pixels, the GI reservoir is stale (it belonged
        // to the previous frame's surface at this pixel). The specular hit distance
        // from that reservoir is meaningless and can confuse DLSS-RR's temporal
        // weighting. Fall back to primary ray depth as a safe substitute.
        if (disoccBias > 0.5f && !isEmissiveOrSky) {
            g_dlssSpecHitDist[DTid.xy] = g_dlssDepth[DTid.xy];
        }
    }
}
