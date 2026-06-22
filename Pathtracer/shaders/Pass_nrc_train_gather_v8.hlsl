//====================================
//NRC X1 TRAINING GATHER (POST-RESTIR)
//====================================
//Emits the NRC training row for the PRIMARY hit (x1) from the converged ReSTIR
//reservoir instead of the single path sample raygen used to store. Runs AFTER
//the full reuse chain (temporal + spatial), so the reservoir at this pixel is
//the low-variance, denoised GI estimate the renderer actually displays.
//
//Why this replaces raygen's x1 vertex:
//  - raygen's path-traced x1 row was ONE Monte-Carlo realization of the GI at
//    x1 -> high variance, which the cache over/under-shoots on noisy surfaces.
//  - the reservoir's F*W is an unbiased GRIS estimate of the integrated GI at
//    x1 (W carries the full pdf accounting) AND it is temporally accumulated /
//    spatially pooled -> far lower variance. Same expectation, less noise =>
//    a more stable cache on the visible primary surface.
//Raygen still trains the deeper vertices (v2+) from the path; only the x1
//(vIdx 0) emit was suppressed there. See Pass_raygen_v8.hlsl.
//
//Target convention (MUST match raygen's deeper rows so the network sees one
//consistent mapping across vertices):
//  raygen target[v] = (full outgoing radiance at v) / reflSum(v)
//  here  target[0]  = (F*W + sun/sky direct at x1) / reflSum1
//F*W (reservoir) = emissive-tri DI + point-light DI + all indirect bounces.
//Scratch slot 3 (directAtX1, written by raygen) = sun + sky direct at x1, which
//raygen splits OUT of the reservoir. Their sum is the full outgoing radiance at
//x1 (minus self-emission / sharp reflection, both ~0 on a diffuse non-emitter),
//i.e. exactly what the cache must predict when it fires (the cache short-
//circuits a path BEFORE NEE at the fire vertex, so it owns direct + indirect).
//The CUDA fill kernel divides L_nee by reflSum=alpha+beta (raw[10..15]); we
//store L_nee = full outgoing radiance with an RR tail (radiance 0) so the
//backward fill yields target[0] = outgoing / reflSum1 directly.

#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "Nrc_v8.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x >= IMG_W || dtid.y >= IMG_H) return;
    //gated like raygen's training-record writes: no cache => no consumer
    if (!NrcIsEnabled() || !NrcIsTrainOn()) return;

    const uint2 pixel = dtid.xy;

    //Match raygen's training-pixel selection EXACTLY (same per-frame tile
    //offset from `time`, same NrcClassifyPixel hash) so the x1 rows land at the
    //same subsampled pixels raygen emits its v2 rows at. One path each => up to
    //two rows per training pixel, which is what kFixedTrainingRecords (the
    //adaptive-tile target) is sized for.
    const uint  tileSide   = NrcTrainingTileSide();
    const uint2 tileOffset = uint2(asuint(time) % tileSide,
                                   (asuint(time) / tileSide) % tileSide);
    bool isTraining, isUnbiased;
    NrcClassifyPixel(pixel, asuint(time), tileSide, tileOffset, isTraining, isUnbiased);
    if (!isTraining) return;
    //NRC_CLASS_TRAIN_UNBIASED pixels keep their CACHE-FREE path-traced x1 row
    //from raygen (their paths never fire the cache), which anchors the cache
    //against the self-reinforcement loop the reservoir-sourced x1 introduces.
    //Skip them here so x1 is never emitted twice. NrcClassifyPixel is the same
    //deterministic hash raygen used this frame, so the two passes always agree.
    if (isUnbiased) return;

    const uint pixelIdx = MapPixelID(gImageSize, pixel);
    if (pixelIdx == 0xFFFFFFFFu) return;

    //x1 must be a real, diffuse-ish surface. Sky / emitter primaries carry no
    //cacheable outgoing radiance; specular x1 (roughness < gate) is a view-
    //dependent reflection the position-dominant cache cannot represent. Mirror
    //raygen's NRC_TRAIN_ROUGHNESS_MIN emit gate so the two sources agree on
    //which surfaces are cache-eligible.
    if (load_isEmitter(g_sample_current, pixelIdx)) return;

    float localPr, localPm;
    load_prpm(g_sample_current, pixelIdx, localPr, localPm);
    if (localPr < NRC_TRAIN_ROUGHNESS_MIN) return;

    const uint   instID = load_instID(g_sample_current, pixelIdx);
    const float3 n1_s   = load_n1_s_with_instID(g_sample_current, pixelIdx, instID);
    const float3 x1     = load_x1_with_instID(g_sample_current, pixelIdx, instID);
    const float3 localKd = load_kd(g_sample_current, pixelIdx);
    const bool   backface = load_backface(g_sample_current, pixelIdx);

    //outgoing direction x1 -> camera (shifted/floating-origin frame, same as
    //the position; viewI[3] is the shifted camera, matching load_x1)
    const float3 camPos  = viewI[3].xyz;
    const float3 viewDir = normalize(camPos - x1);

    //reflectance factorisation matches raygen's cache term / debug query
    const float3 alpha = localKd * (1.0f - localPm);
    const float3 betaC = lerp(float3(0.04f, 0.04f, 0.04f), localKd, localPm);

    //Full outgoing radiance at x1 = reservoir GI (F*W) + split-out sun/sky
    //direct (scratch slot 3). NOTE the feedback path: F*W folds in any cache
    //contribution that ReSTIR selected at deeper vertices, so training x1 on it
    //is a (mild) self-reinforcement loop. The unbiased path-traced v2+ rows and
    //the kTargetMax clamp anchor it; watch indirect regions for slow brightness
    //drift (see NrcLayout.h kUnbiasedDenom note). To make x1 GI-only, drop the
    //slot-3 term (but then x1 targets become inconsistent with the deeper rows
    //that include direct lighting -> bias).
    //
    //Read g_Reservoirs_LAST, not _current: the reuse chain writes its CONVERGED
    //(post-temporal+spatial) reservoir into _last (Pass_spat_gi_v8_1 / cell
    //storeReservoir(g_Reservoirs_last, ...)) and emits the DISPLAYED indirect GI
    //(gScratchPing slot 2) as that reservoir's F*W. g_Reservoirs_current still
    //holds only the post-temporal, PRE-spatial reservoir (Pass_temp_gi writes it,
    //the spatial passes never write back to it). Reading _current here would
    //train the cache on a different, higher-variance, edge-differently-biased
    //quantity than the one the renderer displays and that the cache is later
    //injected/queried against -- exactly the consistency we are after. _last is
    //freshly written THIS frame by the spatial pass (which runs before this
    //gather and stays enabled even during the DLSS res-change window, where only
    //the temporal bits 0x1/0x2 are masked, not spatGI 0x8), so it is always the
    //current-frame, current-layout value; read-only here, so the history
    //ping-pong is undisturbed.
    float3 giRadiance         = load_F(g_Reservoirs_last, pixelIdx)
                              * load_W(g_Reservoirs_last, pixelIdx);
    //RS_FLAG_NO_REUSE_VIS defers the reconnection shadow ray to the spatial resolve,
    //so the stored reservoir F is UNSHADOWED while the displayed slot-2 GI is F*W*V.
    //Re-apply that one visibility here so the cache trains on the SAME shadowed GI it
    //displays; otherwise x1 targets drift brighter than the image -> indirect feedback.
    if (REUSE_VIS_OFF)
    {
        const Reservoir rRes = loadReservoir(g_Reservoirs_last, pixelIdx);
        giRadiance *= ResolveReuseVis(pixelIdx, rRes, giRadiance);
    }
    const float3 sunSkyDirect = gScratchPing[uint3(pixel, 3)].rgb;
    const float3 outgoing     = NrcCleanRadiance(giRadiance + sunSkyDirect);

    //One-vertex "path": vIdx 0 = x1, L_nee = full outgoing radiance, RR tail
    //(radiance 0). The backward fill then gives target[0] = outgoing / reflSum1.
    //emitMask bit 0 set + numVertices 1; kTrainingDepthMask bit 0 is set so the
    //fill kernel emits this row (raygen's x1 vertex has bit 0 CLEAR so it does
    //not, keeping path-traced x1 out of the batch).
    const uint pathId = NrcAllocateTrainingPath();
    if (pathId == NRC_INVALID_PATH) return;

    NrcStoreTrainingVertex(
        pathId, 0u,
        x1, viewDir, n1_s, localPr, alpha, betaC, backface,
        PackRGB9E5(outgoing),   // L_nee = full outgoing radiance at x1
        0u);                     // betaLocal unused (RR tail radiance is 0)
    NrcStorePathTail(pathId, NRC_TAIL_RR, NRC_INVALID_SLOT, 0u);
    NrcStorePathHead(pathId, 1u, 1u);  // numVertices=1, emitMask bit0 set (published last)
}
