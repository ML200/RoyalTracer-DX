#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - PASSTHROUGH  (one deferred visibility ray, no reuse cell)
//====================================
//Extracted out of the old monolithic shift pass so that Pass_shift_v8 (the
//unified reconnection-or-replay mapping, round-8) never has to carry this
//branch: a passthrough pixel has NO spatial reuse cell at all (select's
//search found nothing), so there is no "job" to route through the shift
//job-descriptor model — it just needs its own reservoir's shadow ray, which
//select's compute pass can't cast (RayQuery stays in raygen passes in this
//codebase) and the shared shift shader has no reason to know about.
//
//direct k==2 ONLY: a replay canonical's reconnection segment starts at its
//(never re-derived here) replayed prefix vertex — x1 -> x_k is a
//non-segment through geometry. Its F was generated fully shadowed, so
//passVis stays 1. Runs full-screen, once (NOT looped — there is exactly one
//passthrough job per pixel, if any).

[shader("raygeneration")]
void Pass_spmis_passthrough_v8()
{
    if (!SPMIS_SPATIAL_MODE) return;

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    const uint w0     = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
    const uint status = SPM_hdrStatus(w0);
    if (status != SPM_STATUS_PASS) return;

    const float3 camPos = InitOrigin();
    Reservoir    rdi     = loadReservoir(g_Reservoirs_current, pixelIdx);
    const float3 myPos   = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv_me = BuildVertex(g_sample_current, pixelIdx, myPos, camPos);

    const float  W    = (rdi.W > 0.0f) ? rdi.W : 0.0f;
    const float3 outC = rdi.F * W;
    float passVis = 1.0f;
    if (GetPHat(outC) > 0.0f && !IsVolumeVertex(rdi.matID) &&
        RcK(rdi.rcInfo) == 2u && !RcEnvReplay(rdi.rcInfo))
        //deferred passthrough stores a single float -> achromatic transmittance here (the
        //per-channel thin-glass tint is carried by the inline RIS-target visibility).
        passVis = Luma(ReconnectVis(sv_me.x, sv_me.n_s, rdi.matID, rdi.x2, rdi.n2_s));
    g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(passVis));
}
