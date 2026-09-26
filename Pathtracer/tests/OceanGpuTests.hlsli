#include "OceanMath.hlsli"

// Runs inside the material GPU harness.
float OceanRegressionError()
{
    float error = 0.0f;
    // Direct-light widening must not narrow rough materials or own refraction.
    error = max(error, abs(OceanHighlightRoughness(0.4f) - 0.4f));
    if (OceanHighlightRoughness(0.0f) <= 0.0f ||
        !OceanDirectLightingOwnsRay(true, 0.5f) || OceanDirectLightingOwnsRay(true, -0.5f) ||
        OceanDirectLightingOwnsRay(false, 0.5f)) return 1.0f;
    if (GGXUsesDeltaSampling(10u,0.0f) || !GGXUsesDeltaSampling(5u,0.0f)) return 1.0f;
    // Off-diagonal inverse-transpose terms.
    const float2 gradient = float2(0.3f, -0.2f);
    const float3 stretch = float3(1.2f, 0.8f, 0.25f);
    const float2 slope = OceanWarpedSlope(gradient, stretch);
    const float2 recovered = float2(stretch.x * slope.x + stretch.z * slope.y,
                                     stretch.z * slope.x + stretch.y * slope.y);
    error = max(error, length(recovered - gradient));
    if (!all(isfinite(OceanWarpedSlope(gradient, float3(0, 0, 1))))) return 1.0f;
    const float3 extinction = float3(0.3f, 0.06f, 0.02f);
    const float3 t1 = CalculateAbsorptionThroughput(extinction, 1.0f);
    const float3 t10 = CalculateAbsorptionThroughput(extinction, 10.0f);
    if (!all(t10 < t1) || !(t10.r < t10.g && t10.g < t10.b)) return 1.0f;
    error = max(error, length(t10 - pow(t1, 10.0f)));
    const float3 n = float3(0, 1, 0);
    const float f0 = FresnelDielectricTIR(n, n, 1.0f, 1.333f).x;
    // Broadened highlight off-mirror; the continuation lobe keeps its sharp peak.
    const float3 offMirrorLight = normalize(float3(0.04f,1,0));
    const GGXResult sharpPeak = EvalGGXAll(10u,n,n,n,n,(half)1.0f,(half)1.333f,1.0f,(half)0.0f,(half)0.0f);
    const GGXResult widePeak = EvalGGXAll(10u,n,n,n,n,(half)1.0f,(half)1.333f,1.0f,(half)OceanHighlightRoughness(0.0f),(half)0.0f);
    const GGXResult sharpSide = EvalGGXAll(10u,n,n,n,offMirrorLight,(half)1.0f,(half)1.333f,1.0f,(half)0.0f,(half)0.0f);
    const GGXResult wideSide = EvalGGXAll(10u,n,n,n,offMirrorLight,(half)1.0f,(half)1.333f,1.0f,(half)OceanHighlightRoughness(0.0f),(half)0.0f);
    if (!(sharpPeak.f.x > widePeak.f.x && wideSide.f.x > sharpSide.f.x)) return 1.0f;
    error = max(error, abs(f0 - pow((1.333f - 1.0f) / (1.333f + 1.0f), 2.0f)));
    float3 transmitted;
    if (!RefractVector(n, n, 1.0f / 1.333f, transmitted)) return 1.0f;
    error = max(error, length(transmitted + n));
    if (RefractVector(normalize(float3(0.98f, 0.2f, 0)), n, 1.333f, transmitted)) return 1.0f;

    // GGX sampling, including smooth water and TIR.
    uint seed = 0x51a7e123u;
    uint reflections = 0u;
    uint offMirror = 0u;
    uint coreReflections = 0u;
    [loop] for (uint i = 0u; i < 2048u; ++i) {
        bool refracted = false;
        const float3 wi = SampleBRDF_GGX(10u, n, n, n, 1.0f, 1.333f,
            refracted, seed, 1.0f, 0.0f, 0.0f, true);
        if (!all(isfinite(wi)) || (refracted ? wi.y >= 0.0f : wi.y <= 0.0f)) return 1.0f;
        reflections += refracted ? 0u : 1u;
        offMirror += !refracted && length(wi.xz) > 1e-5f ? 1u : 0u;
        if (!refracted) {
            const float3 h = normalize(n + wi);
            // Normal incidence: half the GGX slope mass within tan(theta) = alpha.
            coreReflections += dot(h.xz,h.xz) <= 1e-6f*h.y*h.y ? 1u : 0u;
        }
    }
    if (abs(float(reflections) / 2048.0f - 0.7f) > 0.04f) return 1.0f;
    // H must not collapse to N.
    if (offMirror < reflections * 9u / 10u) return 1.0f;
    if (abs(float(coreReflections)/max(float(reflections),1.0f) - 0.5f) > 0.06f) return 1.0f;
    bool refracted = false;
    const float3 tir = SampleBRDF_GGX(10u, normalize(float3(0.98f, 0.2f, 0)), n, n,
        1.333f, 1.0f, refracted, seed, 1.0f, 0.0f, 0.0f, true);
    if (refracted || tir.y <= 0.0f) return 1.0f;
    error = max(error, abs(GGXReflectPick(10u, 0.9f, 0.1f) - 0.9f));
    error = max(error, abs(GGXReflectPick(10u, 1.0f, 0.0f) - 1.0f));
    error = max(error, GGXReflectPick(10u, 0.0f, 1.0f));

    // Water proposal changes PDFs only; both hemispheres against glass.
    const float proposalRatio = GGXReflectPick(10u, f0, 1.0f - f0) / GGXReflectPick(5u, f0, 1.0f - f0);
    const float transmissionRatio = (1.0f - GGXReflectPick(10u, f0, 1.0f - f0)) /
        (1.0f - GGXReflectPick(5u, f0, 1.0f - f0));
    [unroll] for (uint side = 0u; side < 2u; ++side) {
        const float3 wi = side == 0u ? n : -n;
        const GGXResult water = EvalGGXAll(10u, n, n, n, wi, (half)1.0f, (half)1.333f, 1.0f, (half)0.25f, (half)0.0f);
        const GGXResult glass = EvalGGXAll(5u, n, n, n, wi, (half)1.0f, (half)1.333f, 1.0f, (half)0.25f, (half)0.0f);
        error = max(error, length(water.f - glass.f));
        if (!(glass.pdf > 0.0f && water.pdf > 0.0f)) return 1.0f;
        error = max(error, abs(water.pdf / glass.pdf - (side == 0u ? proposalRatio : transmissionRatio)));
    }

    // Transmittance composes; HG normalised; free flight incl. clear water.
    error = max(error, length(OceanMediumTransmittance(extinction, 3.0f) *
        OceanMediumTransmittance(extinction, 7.0f) - OceanMediumTransmittance(extinction, 10.0f)));
    error = max(error, abs(OceanPhase(0.3f, 0.0f) - 1.0f/(4.0f*PI)));
    const OceanFlight noEvent = OceanFreeFlight(0.1f.xxx,0.2f.xxx,2.0f,false,float2(0.1f,0.99f));
    const OceanFlight eventFlight = OceanFreeFlight(0.1f.xxx,0.2f.xxx,2.0f,false,float2(0.1f,0.1f));
    if (noEvent.scattered || !eventFlight.scattered) return 1.0f;
    error = max(error,length(noEvent.weight-1.0f));
    error = max(error,length(eventFlight.weight-(2.0f/3.0f)));
    const OceanFlight spent = OceanFreeFlight(0.1f.xxx,0.2f.xxx,2.0f,true,float2(0.1f,0.1f));
    if (spent.scattered) return 1.0f;
    error = max(error,length(spent.weight-exp(-0.6f)));
    const OceanFlight clear = OceanFreeFlight(0.0f,0.0f,1000.0f,false,float2(0.5f,0.1f));
    if (clear.scattered) return 1.0f;
    error = max(error,length(clear.weight-1.0f));
    if (!OceanScatterUsedAfterSurface(true,true,true,true,false) ||
        OceanScatterUsedAfterSurface(true,true,true,false,false) ||
        OceanScatterUsedAfterSurface(true,false,true,true,false) ||
        OceanScatterUsedAfterSurface(true,true,false,true,false)) return 1.0f;
    const float qReflection = OceanReflectionProbability(f0, 1.0f-f0);
    error = max(error, abs(qReflection*(f0/qReflection) - f0));
    error = max(error, abs((1.0f-qReflection)*((1.0f-f0)/(1.0f-qReflection)) - (1.0f-f0)));
    return error;
}
