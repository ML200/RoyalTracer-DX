#include "Includes_v8.hlsli"
#include "Temporal_Merge_v8.hlsli"

//====================================
//TEMPORAL GI  (pass 1/2: candidate pick + k==2 merge)
//====================================
//Owns the reprojection / permutation candidate choice, then runs the shared
//TemporalMergeBody with TEMPORAL_CAN_REPLAY=0: every pixel whose shift needs a
//k>2 prefix replay (in either direction) is parked + appended to the temporal
//replay queue instead, and Pass_temp_replay finishes the identical merge as a
//compacted indirect dispatch. Keeps this full-screen pass RayQuery-only (the
//reuse visibility) with no TraceRay / replay live state.

[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    //emitter / disabled early out
    const uint myFlags = load_flagsWord(g_sample_current, pixelIdx);
    if (myFlags & SD_FLAG_EMITTER)
        return;
    if (!(rs_flags & 2u))
        return;

    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    //====================================
    //BASE REPROJECTION
    //====================================
    //lightweight loads only - all baked, no per-triangle / texture access
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const float3 myPos    = load_x1_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myKd     = load_kd(g_sample_current, pixelIdx);
    float  myPr, myPm;
    load_prpm(g_sample_current, pixelIdx, myPr, myPm);

    //specularity matches DLSS RR EnvBRDFApprox2
    const float3 cameraPos = InitOrigin();
    const float  NoV = saturate(dot(normalize(cameraPos - myPos), myN1s));
    const float  specularity = Luma(EnvBRDFApprox2(myKd, myPr, myPm, NoV));

    //stochastic reprojection, specular vs diffuse MV weighted by specularity
    float4 reflData = gScratchPing[uint3(launchIndex, 4)];
    uint   reflInstID = asuint(reflData.w);
    //valid unless the reflection probe missed (0xFFFFFFFF sentinel).
    bool   reflValid = (reflInstID != 0xFFFFFFFFu);
    float  rSpec = RandomFloatSingle(seed.x);
    bool   useSpecReproj = (rSpec < specularity) && reflValid && !NO_SPEC_REPROJ;

    int2 baseCoord;
    if (useSpecReproj)
    {
        baseCoord = GetBestReprojectedPixel_d(reflData.xyz, prevView, prevProjection, dims_f, reflInstID);
        if (baseCoord.x == -1)
            baseCoord = GetBestReprojectedPixel_d(myPos, prevView, prevProjection, dims_f, myInstID);
    }
    else
    {
        baseCoord = GetBestReprojectedPixel_d(myPos, prevView, prevProjection, dims_f, myInstID);
    }
    if (baseCoord.x == -1 && baseCoord.y == -1)
        baseCoord = (int2)launchIndex;

    //====================================
    //PERMUTED CANDIDATE
    //====================================
    int2 permCoord = baseCoord;
    {
        float u = RandomFloatSingle(permSeed);
        uint  permRnd = (uint)min(u * 16.0f, 15.0f);

        ApplyPermutationSampling(permCoord, permRnd);

        if (permCoord.x < 0 || permCoord.y < 0 ||
            permCoord.x >= (int)IMG_W || permCoord.y >= (int)IMG_H)
        {
            return;
        }
    }

    //shared merge body (direct shifts inline; replay shifts -> temporal queue)
    TemporalMergeBody(pixelIdx, launchIndex, permCoord, false);
}
