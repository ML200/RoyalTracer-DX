#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PER-FRAME SKY LUT BAKE
//====================================
//Two tiny dispatches recorded by Renderer::RecordSkyLUTBake at the top of
//every frame, BEFORE any pass that consumes the LUTs. Rebaking per frame
//(total ~17K texels) sidesteps all invalidation logic for the cbuffer
//params the integrals depend on (turbidity, multi-scatter factor, cloud
//layer sliders, ...) — the whole bake costs less than a single pixel of
//the per-pixel marches it replaces.
//
//  mainTransmittance → g_skyTransmittanceLUT (t49), 256x64 RGBA16F.
//    Atmosphere transmittance to space over the Bruneton (r, mu) domain.
//    Consumed by TransmittanceToSun (SunSampler_v8.hlsli), which used to
//    march this integral per call.
//
//  mainAmbient → g_cloudAmbientLUT (t50), 128x2 RGBA16F.
//    Cloud ambient probe scatter over sun-zenith cosine at the fixed
//    cloud-top probe radius. Row 0 = zenith probe, row 1 = horizon probe.
//    Replicates the old per-pixel IntegrateScattering probes in
//    EvaluateAtmosphereAndClouds, minus the per-step cloud shadow term
//    (a position-dependent quantity a 1D LUT cannot carry — the consumer
//    applies the global-cover proxy instead) and minus the terrain block
//    (equally position-dependent, and negligible at cloud-top altitude).
//
//Bound via a private root signature (CBV b0 + UAV table u25/u26) — the
//global heap isn't set up when this records. Everything the entry points
//touch is cbuffer constants + pure ALU, so none of the scene SRVs that
//Includes_v8.hlsli declares need bindings here.

RWTexture2D<float4> gTransmittanceLUTOut : register(u25);
RWTexture2D<float4> gAmbientLUTOut       : register(u26);

//====================================
//TRANSMITTANCE LUT
//====================================
[numthreads(8, 8, 1)]
void mainTransmittance(uint3 DTid : SV_DispatchThreadID)
{
    const uint2 dims = uint2((uint)SKY_TRANSMITTANCE_LUT_W,
                             (uint)SKY_TRANSMITTANCE_LUT_H);
    if (DTid.x >= dims.x || DTid.y >= dims.y) return;

    float2 uv = (float2(DTid.xy) + 0.5f) / float2(dims);

    float r, mu;
    TransmittanceLutRMuFromUv(uv, r, mu);

    float3 tr = ComputeTransmittanceToTopRMu(r, mu);
    gTransmittanceLUTOut[DTid.xy] = float4(tr, 1.0f);
}

//====================================
//CLOUD AMBIENT LUT
//====================================
//Marched sun transmittance for the probe integral — must NOT call the
//LUT-backed TransmittanceToSun here (it reads the texture the sibling
//dispatch writes this same frame). Mirrors its geometric planet block;
//skips the terrain block (see header).
float3 BakeSunTransmittance(float3 Q, float3 L)
{
    float t0, t1;
    if (!RaySphereIntersect(Q, L, ATMOS_TOP_RADIUS, t0, t1) || t1 <= 0.0f)
        return float3(1, 1, 1);

    float tG0, tG1;
    float RbBlock = ATMOS_BOTTOM_RADIUS - ATMOS_SUN_BLOCK_BIAS_KM;
    if (RaySphereIntersect(Q, L, RbBlock, tG0, tG1) && tG0 > 0.0f && tG0 < t1)
        return float3(0, 0, 0);

    float r  = length(Q);
    float mu = clamp(dot(Q, L) / max(r, 1e-4f), -1.0f, 1.0f);
    return ComputeTransmittanceToTopRMu(r, mu);
}

[numthreads(64, 1, 1)]
void mainAmbient(uint3 DTid : SV_DispatchThreadID)
{
    const uint kW = (uint)CLOUD_AMBIENT_LUT_SIZE;
    if (DTid.x >= kW || DTid.y >= 2u) return;

    // Texel index -> sun-zenith cosine, matching CloudAmbientLutU's
    // texel-center inset (x = i/(N-1) spans the full [-1, 1]).
    float x       = (float)DTid.x / (float)(kW - 1u);
    float sunCosZ = x * 2.0f - 1.0f;

    // Canonical frame: up = +Y, sun in the XY plane. The atmosphere is
    // spherically symmetric so only the relative geometry matters.
    const float3 up = float3(0, 1, 0);
    float sinT = sqrt(saturate(1.0f - sunCosZ * sunCosZ));
    float3 L   = float3(sinT, sunCosZ, 0.0f);

    // Probe radius — must match the cloudProbePos anchor in
    // EvaluateAtmosphereAndClouds.
    float probeR = ATMOS_BOTTOM_RADIUS + CLOUD_LAYER_TOP_KM
                 + CLOUD_TOP_VARIATION_KM + 0.5f;
    float3 O = up * probeR;

    // Row 0: zenith probe. Row 1: horizon probe — perpendicular to the sun
    // azimuth with the same 0.05 up-bias the runtime construction used
    // (perp = cross(up, sunHoriz); sign is irrelevant by mirror symmetry,
    // and at sun-zenith the sky is azimuthally symmetric so the fixed perp
    // replaces the old view-dependent fallback exactly).
    float3 V = up;
    if (DTid.y == 1u)
        V = SafeNormalize(float3(0.0f, 0.05f, -0.95f));

    float tV0, tV1;
    if (!RaySphereIntersect(O, V, ATMOS_TOP_RADIUS, tV0, tV1) || tV1 <= 0.0f)
    {
        gAmbientLUTOut[DTid.xy] = float4(0, 0, 0, 1);
        return;
    }
    float tMin = max(0.0f, tV0);
    float tMax = tV1;

    // Neither probe direction can hit the planet (zenith and +3° above
    // horizontal from cloud-top altitude), so no ground clip.
    float totalDist = tMax - tMin;

    float cosTheta = dot(V, L);
    float phR = PhaseRayleigh(cosTheta);
    float phM = PhaseMieTwoLobe(cosTheta);

    // Mirrors IntegrateScattering's sqrt-spaced view loop. Step count never
    // below the runtime's slider so the LUT is at least as clean as the
    // old per-pixel probes were.
    const int N = max(ATMOS_VIEW_STEPS, 32);

    float3 totalInScatter = float3(0, 0, 0);
    float3 throughput     = float3(1, 1, 1);

    [loop]
    for (int i = 0; i < N; ++i)
    {
        float u0 = (float)i / (float)N;
        float u1 = (float)(i + 1) / (float)N;
        float s0 = u0 * u0;
        float s1 = u1 * u1;
        float tMid = tMin + (s0 + s1) * 0.5f * totalDist;
        float ds   = (s1 - s0) * totalDist;

        float3 P  = O + V * tMid;
        float alt = max(0.0f, length(P) - ATMOS_BOTTOM_RADIUS);

        MediumSample med = SampleMedium(alt);
        float3 segTr = exp(-med.extinction * ds);
        float3 sunTr = BakeSunTransmittance(P, L);

        float3 Pnorm = SafeNormalize(P);
        float sunCosZP   = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f,
            1.0f - (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS) / dot(P, P)));
        float earthShadow = SunDiskFractionAboveHorizon(sunCosZP, cosHorizon);

        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;

        float3 scatterInteg;
        scatterInteg.x = (med.extinction.x > 1e-10f)
            ? scatterPhase.x * (1.0f - segTr.x) / med.extinction.x : scatterPhase.x * ds;
        scatterInteg.y = (med.extinction.y > 1e-10f)
            ? scatterPhase.y * (1.0f - segTr.y) / med.extinction.y : scatterPhase.y * ds;
        scatterInteg.z = (med.extinction.z > 1e-10f)
            ? scatterPhase.z * (1.0f - segTr.z) / med.extinction.z : scatterPhase.z * ds;

        float3 sunIllum = ATMOS_SOLAR_IRRADIANCE * earthShadow * sunTr;
        totalInScatter += throughput * sunIllum * scatterInteg;

        throughput *= segTr;
    }

    totalInScatter *= ATMOS_MULTI_SCATTER_FACTOR;

    gAmbientLUTOut[DTid.xy] = float4(totalInScatter, 1.0f);
}
