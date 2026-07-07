#include "Includes_v8.hlsli"
#include "Temporal_Merge_v8.hlsli"

//====================================
//TEMPORAL GI  (pass 1/3: candidate pick + resolve + route)
//====================================
//Owns the reprojection / permutation candidate choice, then runs
//TemporalMergeBody: writes the forward + reverse job descriptors Pass_shift_v8
//reads (dispatched right after as a single Depth=2 DispatchRays — the SAME
//one binary spatial's Depth=Ntn+1 dispatch also uses). Pass_temp_merge
//(stage 3/3) does the actual MIS combine + reservoir write once both this pass
//and the shift dispatch have finished. Keeps this full-screen pass free of
//TraceRay/RayQuery and any replay-walk live state entirely — it's pure
//candidate resolution + routing, no reconnection math at all.
//
//PESSIMISTIC STATUS: SPM_w0 is written to TEMP_STATUS_DEAD as the very FIRST
//thing, before any early-return check (emitter, tempGI-off, out-of-bounds
//permutation). TemporalMergeBody only overwrites it to TEMP_STATUS_OK once a
//candidate actually resolves; every other exit — including ones added here in
//the future — leaves a coherent status for Pass_temp_merge without needing a
//matching store at each return site.

[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    g_pathStateBuffer.Store(SPM_w0(pixelIdx), TEMP_STATUS_DEAD);
    //Pessimistic job clear, same reasoning as w0 above:
    //Pass_shift_v8's temporal_shift Depth-slices are dispatched
    //unconditionally by the host (the dispatch has no per-mode gate)
    //and check ONLY slotZ (not w0) to decide whether a job exists, so every
    //early-return path here — including TemporalMergeBody's own internal
    //"candidate dead" return — must leave "no job" behind, not whatever a
    //past frame (tempGI on, or a since-resolved candidate) last wrote.
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), SP_UNDEF);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 1u), SP_UNDEF);

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

    //shared merge body (resolves + writes the forward/reverse job descriptors
    //Pass_shift_v8 reads next — no queue, no inline shifting here)
    TemporalMergeBody(pixelIdx, launchIndex, permCoord, false);
}
