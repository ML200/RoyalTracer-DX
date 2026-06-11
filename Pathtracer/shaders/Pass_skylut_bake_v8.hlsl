#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PER-FRAME SKY LUT BAKE
//====================================
//Three tiny dispatches recorded by Renderer::RecordSkyLUTBake at the top of
//every frame, BEFORE any pass that consumes the LUTs. Rebaking per frame
//(total ~18K texels) sidesteps all invalidation logic for the cbuffer
//params the integrals depend on (turbidity, multi-scatter factor, cloud
//layer sliders, ...) — the whole bake costs less than a single pixel of
//the per-pixel marches it replaces.
//
//  mainTransmittance → g_skyTransmittanceLUT (t49), 256x64 RGBA16F.
//    Atmosphere transmittance to space over the Bruneton (r, mu) domain.
//    Consumed by TransmittanceToSun (SunSampler_v8.hlsli), which used to
//    march this integral per call.
//
//  mainMultiScatter → g_skyMultiScatterLUT (t51), 32x32 RGBA16F.
//    Hillaire 2020 multiple-scattering transfer Psi_ms over (sun-zenith
//    cosine, normalized altitude). READS the transmittance UAV (u25), so
//    the C++ side MUST dispatch mainTransmittance first with a UAV
//    barrier between.
//
//  mainAmbient → g_cloudAmbientLUT (t50), 128x2 RGBA16F.
//    Cloud ambient probe scatter over sun-zenith cosine at the fixed
//    cloud-top probe radius. Row 0 = zenith probe, row 1 = horizon probe.
//    Replicates the old per-pixel IntegrateScattering probes in
//    EvaluateAtmosphereAndClouds, minus the per-step cloud shadow term
//    (a position-dependent quantity a 1D LUT cannot carry — the consumer
//    applies the local-cover proxy instead) and minus the terrain block
//    (equally position-dependent, and negligible at cloud-top altitude).
//    READS the multi-scatter UAV (u27) so the probes carry 2nd+ order
//    light — the C++ side MUST dispatch mainMultiScatter first and put a
//    UAV barrier between the two.
//
//Bound via a private root signature (CBV b0 + UAV table u25..u27) — the
//global heap isn't set up when this records. Everything the entry points
//touch is cbuffer constants + pure ALU, so none of the scene SRVs that
//Includes_v8.hlsli declares need bindings here.

RWTexture2D<float4> gTransmittanceLUTOut : register(u25);
RWTexture2D<float4> gAmbientLUTOut       : register(u26);
RWTexture2D<float4> gMultiScatterLUTOut  : register(u27);

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
//SUN TRANSMITTANCE FOR THE PROBE / MS INTEGRALS
//====================================
//Reads the transmittance LUT the sibling mainTransmittance dispatch wrote
//THIS frame, via manual-bilinear UAV loads (no sampler on UAVs) — the C++
//records mainTransmittance first and puts a UAV barrier before any
//consumer dispatch. Mirrors TransmittanceToSun's geometric planet block;
//skips the terrain block (see header).
//
//This used to call ComputeTransmittanceToTopRMu inline (a 64-step
//SampleMedium march) "to avoid reading a texture the sibling writes" —
//fine for mainAmbient's 256x32 calls, catastrophic once mainMultiScatter
//multiplied it by 1024 texels x 64 dirs x 20 steps: ~84M SampleMedium
//calls on a 1024-thread dispatch. At that occupancy nothing hides the
//latency and the bake serialized ~1-2 ms of near-idle GPU at the top of
//every frame, behind barriers that block all later passes. Four UAV
//loads replace the 64-step integral; the explicit barrier makes it safe.
float3 BakeSunTransmittance(float3 Q, float3 L)
{
    float t0, t1;
    if (!RaySphereIntersect(Q, L, ATMOS_TOP_RADIUS, t0, t1) || t1 <= 0.0f)
        return float3(1, 1, 1);

    float tG0, tG1;
    float RbBlock = ATMOS_BOTTOM_RADIUS - ATMOS_SUN_BLOCK_BIAS_KM;
    if (RaySphereIntersect(Q, L, RbBlock, tG0, tG1) && tG0 > 0.0f && tG0 < t1)
        return float3(0, 0, 0);

    float r  = clamp(length(Q), ATMOS_BOTTOM_RADIUS, ATMOS_TOP_RADIUS);
    float mu = clamp(dot(Q, L) / max(r, 1e-4f), -1.0f, 1.0f);
    float2 uv = TransmittanceLutUvFromRMu(r, mu);

    // Continuous texel coords; LutCoordFromUnitRange's texel-center inset
    // makes u*W - 0.5 = x*(W-1), same scheme as AmbientMsPsi below.
    float fx = uv.x * SKY_TRANSMITTANCE_LUT_W - 0.5f;
    float fy = uv.y * SKY_TRANSMITTANCE_LUT_H - 0.5f;
    uint  x0 = (uint)clamp(fx, 0.0f, SKY_TRANSMITTANCE_LUT_W - 2.0f);
    uint  y0 = (uint)clamp(fy, 0.0f, SKY_TRANSMITTANCE_LUT_H - 2.0f);
    float wx = saturate(fx - (float)x0);
    float wy = saturate(fy - (float)y0);
    float3 v00 = gTransmittanceLUTOut[uint2(x0,      y0     )].rgb;
    float3 v10 = gTransmittanceLUTOut[uint2(x0 + 1u, y0     )].rgb;
    float3 v01 = gTransmittanceLUTOut[uint2(x0,      y0 + 1u)].rgb;
    float3 v11 = gTransmittanceLUTOut[uint2(x0 + 1u, y0 + 1u)].rgb;
    return lerp(lerp(v00, v10, wx), lerp(v01, v11, wx), wy);
}

//====================================
//MULTIPLE-SCATTERING LUT (Hillaire 2020, eq. 10)
//====================================
//Per texel: estimate the 2nd-order in-scatter L2 and the scattering
//transfer fms by marching SKY_MS_DIRS uniform sphere directions from the
//probe point, both under the uniform-phase approximation; the geometric
//series of higher orders collapses to Psi_ms = L2 / (1 - fms). Sun
//transmittance per step comes from the transmittance LUT (UAV reads, see
//BakeSunTransmittance above — the C++ barriers guarantee ordering); the
//planet block + earth shadow inside the per-step sun term are what make
//the LUT carry twilight correctly. Ground-albedo bounce term omitted —
//the engine renders no analytic planet surface.
//
//Cost: 32x32 texels x 64 dirs x 20 steps x ~4 UAV loads — tens of µs on
//a 1024-thread dispatch, and the result is read billions of times.

#define SKY_MS_LUT_DIM 32u
#define SKY_MS_DIRS    64u
#define SKY_MS_STEPS   20

// Deterministic uniform sphere coverage (golden-angle spiral).
inline float3 MsFibonacciDir(uint i, uint n)
{
    float z   = 1.0f - (2.0f * (float)i + 1.0f) / (float)n;
    float s   = sqrt(max(0.0f, 1.0f - z * z));
    float phi = (float)i * 2.39996323f;
    return float3(cos(phi) * s, z, sin(phi) * s);
}

[numthreads(8, 8, 1)]
void mainMultiScatter(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= SKY_MS_LUT_DIM || DTid.y >= SKY_MS_LUT_DIM) return;

    const float Rb = ATMOS_BOTTOM_RADIUS;
    const float Rt = ATMOS_TOP_RADIUS;

    // Texel -> (sunCosZ, r), inverse of MultiScatterPsi's mapping.
    float2 uv      = (float2(DTid.xy) + 0.5f) / (float)SKY_MS_LUT_DIM;
    float  sunCosZ = LutUnitRangeFromCoord(uv.x, (float)SKY_MS_LUT_DIM) * 2.0f - 1.0f;
    float  r       = Rb + LutUnitRangeFromCoord(uv.y, (float)SKY_MS_LUT_DIM) * (Rt - Rb);
    r = clamp(r, Rb + 1e-3f, Rt - 1e-3f);

    float  sinT = sqrt(saturate(1.0f - sunCosZ * sunCosZ));
    float3 L    = float3(sinT, sunCosZ, 0.0f);
    float3 O    = float3(0.0f, r, 0.0f);

    const float kUniformPhase = 1.0f / (4.0f * PI);

    float3 L2  = float3(0, 0, 0);   // 2nd-order in-scatter, uniform phase
    float3 fms = float3(0, 0, 0);   // scattering transfer

    [loop]
    for (uint d = 0; d < SKY_MS_DIRS; ++d)
    {
        float3 V = MsFibonacciDir(d, SKY_MS_DIRS);

        float tA0, tA1;
        if (!RaySphereIntersect(O, V, Rt, tA0, tA1) || tA1 <= 0.0f) continue;
        float tMax = tA1;
        float tG0, tG1;
        if (RaySphereIntersect(O, V, Rb, tG0, tG1) && tG0 > 0.0f)
            tMax = min(tMax, tG0);
        if (tMax <= 0.0f) continue;

        float  ds         = tMax / (float)SKY_MS_STEPS;
        float3 throughput = float3(1, 1, 1);

        [loop]
        for (int s = 0; s < SKY_MS_STEPS; ++s)
        {
            float  t   = ((float)s + 0.5f) * ds;
            float3 P   = O + V * t;
            float  alt = max(0.0f, length(P) - Rb);

            MediumSample med  = SampleMedium(alt);
            float3 segTr      = exp(-med.extinction * ds);
            float3 sunTr      = BakeSunTransmittance(P, L);

            float3 Pnorm       = SafeNormalize(P);
            float  cosHorizon  = -sqrt(max(0.0f,
                1.0f - (Rb * Rb) / dot(P, P)));
            float  earthShadow = SunDiskFractionAboveHorizon(dot(Pnorm, L),
                                                             cosHorizon);

            // Analytic per-segment integration, same trick as the view
            // marches: integ = (1 - segTr) / extinction.
            float3 integ;
            integ.x = (med.extinction.x > 1e-10f)
                ? (1.0f - segTr.x) / med.extinction.x : ds;
            integ.y = (med.extinction.y > 1e-10f)
                ? (1.0f - segTr.y) / med.extinction.y : ds;
            integ.z = (med.extinction.z > 1e-10f)
                ? (1.0f - segTr.z) / med.extinction.z : ds;

            float3 scatterIso = med.scatterR + med.scatterM;

            L2  += throughput * scatterIso * kUniformPhase
                 * (ATMOS_SOLAR_IRRADIANCE * earthShadow * sunTr) * integ;
            fms += throughput * scatterIso * integ;

            throughput *= segTr;
        }
    }

    L2  /= (float)SKY_MS_DIRS;
    fms /= (float)SKY_MS_DIRS;

    float3 psi = L2 / max(1.0f - fms, 1e-3f);
    gMultiScatterLUTOut[DTid.xy] = float4(psi, 1.0f);
}

// Psi_ms lookup for the ambient probe march below — manual bilinear over
// the sibling dispatch's UAV (u27, no sampler on UAVs). Continuous texel
// coord x*(DIM-1) matches the runtime MultiScatterPsi texel-center-inset
// mapping exactly. Requires the C++ UAV barrier between the dispatches.
float3 AmbientMsPsi(float r, float sunCosZ)
{
    float xR  = saturate((r - ATMOS_BOTTOM_RADIUS)
                       / (ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float xMu = saturate(sunCosZ * 0.5f + 0.5f);
    float fx  = xMu * (float)(SKY_MS_LUT_DIM - 1u);
    float fy  = xR  * (float)(SKY_MS_LUT_DIM - 1u);
    uint  x0  = min((uint)fx, SKY_MS_LUT_DIM - 2u);
    uint  y0  = min((uint)fy, SKY_MS_LUT_DIM - 2u);
    float wx  = saturate(fx - (float)x0);
    float wy  = saturate(fy - (float)y0);
    float3 v00 = gMultiScatterLUTOut[uint2(x0,      y0     )].rgb;
    float3 v10 = gMultiScatterLUTOut[uint2(x0 + 1u, y0     )].rgb;
    float3 v01 = gMultiScatterLUTOut[uint2(x0,      y0 + 1u)].rgb;
    float3 v11 = gMultiScatterLUTOut[uint2(x0 + 1u, y0 + 1u)].rgb;
    return lerp(lerp(v00, v10, wx), lerp(v01, v11, wx), wy);
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
        float  rP  = length(P);
        float  alt = max(0.0f, rP - ATMOS_BOTTOM_RADIUS);

        MediumSample med = SampleMedium(alt);
        float3 segTr = exp(-med.extinction * ds);
        float3 sunTr = BakeSunTransmittance(P, L);

        float3 Pnorm = SafeNormalize(P);
        float sunCosZP   = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f,
            1.0f - (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS) / dot(P, P)));
        float earthShadow = SunDiskFractionAboveHorizon(sunCosZP, cosHorizon);

        // Single scatter (slider boost) + Hillaire 2nd+ order from the
        // sibling MS dispatch — mirrors the runtime integrators so the
        // cloud ambient probes carry the same energy, including twilight.
        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;
        float3 scatterIso   = med.scatterR + med.scatterM;
        float3 sunIllum     = ATMOS_SOLAR_IRRADIANCE * earthShadow * sunTr
                            * ATMOS_MULTI_SCATTER_FACTOR;
        float3 psiMS        = AmbientMsPsi(rP, sunCosZP);
        float3 rate         = scatterPhase * sunIllum
                            + scatterIso * psiMS * ATMOS_SOLAR_IRRADIANCE;

        float3 scatterInteg;
        scatterInteg.x = (med.extinction.x > 1e-10f)
            ? rate.x * (1.0f - segTr.x) / med.extinction.x : rate.x * ds;
        scatterInteg.y = (med.extinction.y > 1e-10f)
            ? rate.y * (1.0f - segTr.y) / med.extinction.y : rate.y * ds;
        scatterInteg.z = (med.extinction.z > 1e-10f)
            ? rate.z * (1.0f - segTr.z) / med.extinction.z : rate.z * ds;

        totalInScatter += throughput * scatterInteg;

        throughput *= segTr;
    }

    // Multi-scatter factor moved onto the per-step single-scatter term; the
    // Psi_ms contribution is real multiple scattering and must not be
    // double-boosted by the artistic slider.

    gAmbientLUTOut[DTid.xy] = float4(totalInScatter, 1.0f);
}
