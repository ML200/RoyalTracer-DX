#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"

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

        passVis = Luma(ReconnectVis(sv_me.x, sv_me.n_s, rdi.matID, rdi.x2, rdi.n2_s));
    g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(passVis));
}
