#include "Includes_v8.hlsli"
#include "PtDefer_v8.hlsli"

// Light sampling for the deferred vertex: pick one triangle from the light tree, resolve the point
// on it and the sun sample, and trace both shadow rays. No material evaluation happens here. Each
// part of the light record is stored as soon as it is known, so almost nothing stays live across
// the tree walk and the two traversals.
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(tid.xy >= imgSize)) return;
    const uint2 pixel    = tid.xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    const uint info = DvLoadInfo(pixelIdx);
    if ((info & DV_VALID) == 0u) return;
    if ((DvLoadVertexFlags(pixelIdx) & DVF_PERFORM_NEE) == 0u) return;

    float3 pos, n;
    DvLoadVertexSurface(pixelIdx, pos, n);
    const uint ps        = DvLoadPs(pixelIdx);
    const uint depth     = PtPsDepth(ps);
    const uint s         = PtPsSample(ps);
    const uint pathSeed  = PtPathSeed(pixel, s);
    const bool blue      = PtPsDiffDepth(ps) == 0u;
    const uint blueIndex = (uint)time * PtSampleCount() + s;
    const bool useLearnedLights = LTC_UseSurfaceLearning();

    // --- the pick: its token and the tree pdf of a pending emitter hit go out right away ---
    LT_Sample pick;
    pick.id = LT_SENTINEL;
    pick.inst = LT_SENTINEL;
    pick.pdf = 0.0f;
    pick.learningToken = 0u;
    if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) == 0u)
    {
        uint sNee = RcBounceSeed(pathSeed, depth, RC_STREAM_NEE);
        pick = LT_SampleLight(pos, n, sNee, useLearnedLights);
    }
    float emitterPdfArea = 0.0f;
    if (((info >> DV_END_SHIFT) & DV_END_MASK) == DV_END_EMITTER && (info & DV_MIS_NONE) == 0u)
    {
        const DvEmitter e = DvLoadEmitter(pixelIdx);
        emitterPdfArea = LT_Pdf_LightTree_Area(pos, n, e.lightID, e.inst, useLearnedLights);
    }
    DvStoreLightPick(pixelIdx, pick.learningToken, emitterPdfArea);

    // --- the point on the picked triangle ---
    uint sPoint = RcBounceSeed(pathSeed, depth, PT_STREAM_LIGHT_POINT);
    const LT_LightSampleResult light = LT_SamplePointOnLightTree(pos, pick, sPoint);
    const float3 toLight = light.position - pos;
    const float3 L = toLight * rsqrt(max(dot(toLight, toLight), 1e-20f));
    const bool lightValid = dot(n, L) > 1e-6f && dot(light.normal, -L) > 1e-6f && light.pdfSolidAngle > 1e-20f;
    DvStoreLightMesh(pixelIdx, light.objID, lightValid ? light.pdfSolidAngle : 0.0f,
        light.position, light.normal, light.emission);

    // --- the sun ---
    float2 rSun = float2(RandomFloatSingle(sPoint), RandomFloatSingle(sPoint));
    if (blue) rSun = PtBlue2(pixel, blueIndex, depth, BN_PAIR_SUN);
    const SunSampleResult sun = SampleSun(rSun, pos + sceneOriginWorld);
    const bool sunValid = dot(n, sun.direction) > 1e-6f && sun.pdf > 1e-20f;
    DvStoreLightSun(pixelIdx, sun.direction, sunValid ? sun.radiance : float3(0, 0, 0), sun.pdf);

    // --- visibility: one traversal for both targets, each result stored as it arrives ---
    [loop]
    for (uint target = 0u; target < 2u; ++target)
    {
        float3 vis = 0.0f;
        if (target == 0u ? lightValid : sunValid)
        {
            const float3 to  = target == 0u ? light.position : pos + sun.direction * RAY_TMAX_PLANET;
            const float3 toN = target == 0u ? light.normal : -sun.direction;
            vis = VisibilityTransmittance(pos, n, to, toN);
        }
        DvStoreLightVisibility(pixelIdx, target, vis);
    }
}
