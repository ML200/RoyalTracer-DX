// ============================================================================
// Sun + Physically-Based Atmospheric Sky
// Based on: Bruneton 2008/2017 parameterization (exact Earth coefficients)
//           Hillaire 2020 single-scatter ray march approach
//
// Drop-in replacement — public API identical:
//   SunSampleResult SampleSun(float2 u)
//   float          GetSunPdf(float3 rayDir)
//   float3         EvaluateSun(float3 rayDir)
//   float3         EvaluateSky(float3 rayDir)
//
// Approach: single-scattering ray march with Rayleigh + Mie + ozone,
// two-lobe Cornette-Shanks Mie phase for balanced halo/horizon,
// Bruneton piecewise-linear ozone profile,
// and a simple constant multi-scattering energy correction.
// ============================================================================

#define TAU     (2.0f * PI)
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

// ---------------------------------------------------------------------------
// External time input
// ---------------------------------------------------------------------------
#ifndef SUN_FRAMECOUNT
#define SUN_FRAMECOUNT time
#endif

// ---------------------------------------------------------------------------
// Location / date
// ---------------------------------------------------------------------------
#ifndef SUN_LATITUDE_DEG
#define SUN_LATITUDE_DEG   48.5200f
#endif

#ifndef SUN_LONGITUDE_DEG
#define SUN_LONGITUDE_DEG  11.4050f
#endif

#ifndef SUN_DAY_OF_YEAR
#define SUN_DAY_OF_YEAR    172.0f
#endif

// ---------------------------------------------------------------------------
// Simulation control
// ---------------------------------------------------------------------------
#ifndef SUN_FPS
#define SUN_FPS            90.0f
#endif

#ifndef SUN_SIM_SPEED
#define SUN_SIM_SPEED      10.0f
#endif

#ifndef SUN_START_UTC_HOURS
#define SUN_START_UTC_HOURS 6.0f
#endif

#ifndef SUN_NIGHT_SPEEDUP
#define SUN_NIGHT_SPEEDUP  2.0f
#endif

// ---------------------------------------------------------------------------
// Sun appearance
// ---------------------------------------------------------------------------
#ifndef SUN_ANGULAR_DEG
#define SUN_ANGULAR_DEG    0.53f
#endif

#ifndef SUN_INTENSITY_VAL
#define SUN_INTENSITY_VAL  5.0f
#endif

#ifndef SUN_COLOR_VAL
#define SUN_COLOR_VAL      float3(1.0f, 0.99f, 0.98f)
#endif

#ifndef SUN_DIST_INF
#define SUN_DIST_INF       1e7f
#endif

#ifndef SUN_HORIZON_DEG
#define SUN_HORIZON_DEG    -0.833f
#endif

#ifndef SUN_LIMB_DARKENING
#define SUN_LIMB_DARKENING 1
#endif

// ---------------------------------------------------------------------------
// Atmosphere: Bruneton 2017 standard Earth parameters
// ALL lengths in km.  ALL coefficients in 1/km.
// ---------------------------------------------------------------------------

#ifndef ATMOS_BOTTOM_RADIUS
#define ATMOS_BOTTOM_RADIUS     6360.0f
#endif

#ifndef ATMOS_TOP_RADIUS
#define ATMOS_TOP_RADIUS        6420.0f
#endif

// ---- Rayleigh (air molecules) ----
#ifndef ATMOS_RAYLEIGH_SCATTER
#define ATMOS_RAYLEIGH_SCATTER  float3(0.005802f, 0.013558f, 0.033100f)
#endif

#ifndef ATMOS_RAYLEIGH_SCALE_H
#define ATMOS_RAYLEIGH_SCALE_H  8.0f
#endif

// ---- Mie (aerosols) ----
#ifndef ATMOS_MIE_SCATTER_BASE
#define ATMOS_MIE_SCATTER_BASE  float3(0.003996f, 0.003996f, 0.003996f)
#endif

#ifndef ATMOS_MIE_EXTINCT_BASE
#define ATMOS_MIE_EXTINCT_BASE  float3(0.004440f, 0.004440f, 0.004440f)
#endif

#ifndef ATMOS_MIE_SCALE_H
#define ATMOS_MIE_SCALE_H       1.2f
#endif

// Two-lobe Mie phase function:
//   Primary lobe:   narrow forward scatter (the sun halo core)
//   Secondary lobe: wider forward scatter (soft glow around sun)
// The two-lobe approach gives a visible halo on short zenith paths
// without over-accumulating scatter on long horizon paths.
#ifndef ATMOS_MIE_G_PRIMARY
#define ATMOS_MIE_G_PRIMARY     0.76f
#endif

#ifndef ATMOS_MIE_G_SECONDARY
#define ATMOS_MIE_G_SECONDARY   0.35f
#endif

// Weight of secondary lobe (0 = single lobe, 1 = all secondary)
#ifndef ATMOS_MIE_LOBE2_WEIGHT
#define ATMOS_MIE_LOBE2_WEIGHT  0.15f
#endif

// Turbidity multiplier on Mie (1.0 = very clear, 2-3 = clear, 5+ = hazy)
#ifndef SUN_TURBIDITY
#define SUN_TURBIDITY           2.0f
#endif

// ---- Ozone (absorption only) ----
#ifndef ATMOS_OZONE_ABSORPTION
#define ATMOS_OZONE_ABSORPTION  float3(0.000650f, 0.001881f, 0.000085f)
#endif

// ---- Ray march quality ----
#ifndef ATMOS_VIEW_STEPS
#define ATMOS_VIEW_STEPS        12
#endif

#ifndef ATMOS_LIGHT_STEPS
#define ATMOS_LIGHT_STEPS       8
#endif

// ---- Solar irradiance at top of atmosphere ----
#ifndef ATMOS_SOLAR_IRRADIANCE
#define ATMOS_SOLAR_IRRADIANCE  float3(1.0f, 1.0f, 1.0f)
#endif

// ---- Multi-scattering energy correction ----
#ifndef ATMOS_MULTI_SCATTER_FACTOR
#define ATMOS_MULTI_SCATTER_FACTOR  1.1f
#endif

// ---- Final sky exposure ----
#ifndef SKY_INTENSITY
#define SKY_INTENSITY           6.0f
#endif

// ---------------------------------------------------------------------------
// Night sky
// ---------------------------------------------------------------------------
#ifndef SKY_TWILIGHT_DEG
#define SKY_TWILIGHT_DEG        18.0f
#endif

#ifndef SKY_STAR_GRID
#define SKY_STAR_GRID           900.0f
#endif

#ifndef SKY_STAR_DENSITY
#define SKY_STAR_DENSITY        0.997f
#endif

#ifndef SKY_STAR_INTENSITY
#define SKY_STAR_INTENSITY      3.0f
#endif

#ifndef SKY_STAR_SCALE
#define SKY_STAR_SCALE          1.0f
#endif

#ifndef SKY_SIDEREAL_RATIO
#define SKY_SIDEREAL_RATIO      1.00273790935f
#endif

#ifndef SKY_GROUND_DARKEN
#define SKY_GROUND_DARKEN       0.04f
#endif

#ifndef SKY_NIGHT_BASE
#define SKY_NIGHT_BASE          float3(0.00015f, 0.00020f, 0.00035f)
#endif

// ---------------------------------------------------------------------------
// Star config
// ---------------------------------------------------------------------------
#ifndef SKY_STAR_LAYERS
#define SKY_STAR_LAYERS         3
#endif

#ifndef SKY_STAR_SCINTILLATION
#define SKY_STAR_SCINTILLATION  0.04f
#endif

// How many degrees past civil twilight (-6°) the brightest stars persist.
// At magnitude=1.0, stars fade at elevDeg = -6 + SKY_STAR_DAWN_LINGER.
// Dim stars (magnitude~0) still fade at the normal deep twilight threshold.
// 8.0 means the very brightest stars linger until sun is about -6+8 = +2°
// (just above horizon). Realistic: Venus/Sirius visible well into dawn.
#ifndef SKY_STAR_DAWN_LINGER
#define SKY_STAR_DAWN_LINGER    10.0f
#endif

// ---------------------------------------------------------------------------
// World orientation
// ---------------------------------------------------------------------------
#ifndef WORLD_NORTH
#define WORLD_NORTH             normalize(float3(0, 0, 1))
#endif

static const float3 WORLD_UP = float3(0, 1, 0);

// ============================================================================
// Structs
// ============================================================================
struct SunSampleResult
{
    float3 direction;
    float3 radiance;
    float  pdf;
    float  dist;
};

struct SunState
{
    float3 dirWS;
    float  elevRad;
    float  cosThetaMax;
    float  omega;
    float  pdf;
    float3 radiance;
    float3 tint;
    float  visible;
};

struct MediumSample
{
    float3 scatterR;        // Rayleigh scattering [1/km]
    float3 scatterM;        // Mie scattering [1/km]
    float3 extinction;      // Total extinction [1/km]
};

// ============================================================================
// Utility
// ============================================================================
inline float3 SafeNormalize(float3 v)
{
    return (dot(v, v) > 0.0f) ? normalize(v) : float3(0, 1, 0);
}

float GetSunSolidAngle(float thetaMaxRad)
{
    return TAU * (1.0f - cos(thetaMaxRad));
}

void GetOrthoBasis(float3 N, out float3 T, out float3 B)
{
    float3 Up = abs(dot(N, WORLD_UP)) < 0.999f ? WORLD_UP : float3(1, 0, 0);
    T = SafeNormalize(cross(Up, N));
    B = cross(N, T);
}

inline float Smooth01(float x)
{
    x = saturate(x);
    return x * x * (3.0f - 2.0f * x);
}

// ============================================================================
// Hashing (improved distribution for stars)
// ============================================================================
inline float Hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

inline float2 Hash22(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac(float2((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y));
}

// ============================================================================
// Solar position (NOAA)
// ============================================================================
inline float SolarDeclinationRad(float dayOfYear, float timeHours)
{
    float gamma = TAU / 365.0f * (dayOfYear - 1.0f + (timeHours - 12.0f) / 24.0f);
    return 0.006918f
         - 0.399912f * cos(gamma) + 0.070257f * sin(gamma)
         - 0.006758f * cos(2.0f * gamma) + 0.000907f * sin(2.0f * gamma)
         - 0.002697f * cos(3.0f * gamma) + 0.001480f * sin(3.0f * gamma);
}

inline void SunriseSunsetHours(float latRad, float declRad, float h0Rad,
                               out float sunriseH, out float sunsetH, out float dayLenH)
{
    float sinLat = sin(latRad), cosLat = cos(latRad);
    float sinD   = sin(declRad), cosD  = cos(declRad);
    float denom  = max(1e-6f, cosLat * cosD);
    float cosH0  = (sin(h0Rad) - sinLat * sinD) / denom;

    if (cosH0 >= 1.0f)  { dayLenH = 0.0f;  sunriseH = 12.0f; sunsetH = 12.0f; return; }
    if (cosH0 <= -1.0f) { dayLenH = 24.0f; sunriseH = 0.0f;  sunsetH = 24.0f; return; }

    float H0 = acos(clamp(cosH0, -1.0f, 1.0f));
    dayLenH  = 2.0f * H0 * RAD2DEG / 15.0f;
    sunriseH = 12.0f - 0.5f * dayLenH;
    sunsetH  = 12.0f + 0.5f * dayLenH;
}

inline float WarpToSolarHours(float u, float sunriseH, float sunsetH, float dayLenH, float nightSpeedUp)
{
    if (dayLenH <= 1e-3f || dayLenH >= 23.999f) return u * 24.0f;

    float nightLenH = 24.0f - dayLenH;
    float wDay   = dayLenH;
    float wNight = nightLenH / max(1.0f, nightSpeedUp);
    float wTot   = wDay + wNight;

    float s = u * wTot;
    float t = (s < wDay) ? (sunriseH + s) : (sunsetH + (s - wDay) * nightSpeedUp);
    return frac(t / 24.0f) * 24.0f;
}

inline float3 ENU_ToWorld(float east, float north, float up)
{
    float3 N = SafeNormalize(float3(WORLD_NORTH.x, 0.0f, WORLD_NORTH.z));
    float3 E = SafeNormalize(cross(WORLD_UP, N));
    return SafeNormalize(east * E + north * N + up * WORLD_UP);
}

inline float GetSolarTimeHours()
{
    float tReal = (float)SUN_FRAMECOUNT / SUN_FPS;
    float tSim  = tReal * SUN_SIM_SPEED;
    float u = frac((SUN_START_UTC_HOURS / 24.0f) + (tSim / 86400.0f) + (SUN_LONGITUDE_DEG / 360.0f));

    float latRad   = SUN_LATITUDE_DEG * DEG2RAD;
    float h0Rad    = SUN_HORIZON_DEG * DEG2RAD;
    float declNoon = SolarDeclinationRad(SUN_DAY_OF_YEAR, 12.0f);

    float sunriseH, sunsetH, dayLenH;
    SunriseSunsetHours(latRad, declNoon, h0Rad, sunriseH, sunsetH, dayLenH);

    return WarpToSolarHours(u, sunriseH, sunsetH, dayLenH, SUN_NIGHT_SPEEDUP);
}

inline void GetSunDirAndElev(out float3 dirWS, out float elevRad)
{
    float latRad  = SUN_LATITUDE_DEG * DEG2RAD;
    float solarH  = GetSolarTimeHours();
    float declRad = SolarDeclinationRad(SUN_DAY_OF_YEAR, solarH);
    float H = (solarH - 12.0f) * 15.0f * DEG2RAD;

    float sinLat = sin(latRad), cosLat = cos(latRad);
    float sinD   = sin(declRad), cosD  = cos(declRad);

    float east  =  cosD * sin(H);
    float north =  cosLat * sinD - sinLat * cosD * cos(H);
    float up    =  sinLat * sinD + cosLat * cosD * cos(H);

    dirWS   = ENU_ToWorld(east, north, up);
    elevRad = asin(clamp(up, -1.0f, 1.0f));
}

// ============================================================================
// Atmosphere medium sampling
// ============================================================================

inline float DensityOzone(float altKm)
{
    return (altKm < 25.0f)
        ? max(0.0f, altKm / 15.0f - 2.0f / 3.0f)
        : max(0.0f, -altKm / 15.0f + 8.0f / 3.0f);
}

inline MediumSample SampleMedium(float altKm)
{
    MediumSample m;

    float dR = exp(-altKm / ATMOS_RAYLEIGH_SCALE_H);
    float dM = exp(-altKm / ATMOS_MIE_SCALE_H);
    float dO = DensityOzone(altKm);

    float turbScale = max(1.0f, SUN_TURBIDITY);

    m.scatterR = ATMOS_RAYLEIGH_SCATTER * dR;
    m.scatterM = ATMOS_MIE_SCATTER_BASE * dM * turbScale;

    float3 extinctR = ATMOS_RAYLEIGH_SCATTER * dR;
    float3 extinctM = ATMOS_MIE_EXTINCT_BASE * dM * turbScale;
    float3 absorbO  = ATMOS_OZONE_ABSORPTION * dO;

    m.extinction = extinctR + extinctM + absorbO;

    return m;
}

// ============================================================================
// Phase functions
// ============================================================================
inline float PhaseRayleigh(float cosTheta)
{
    return (3.0f / (16.0f * PI)) * (1.0f + cosTheta * cosTheta);
}

// Cornette-Shanks phase function
inline float PhaseMieCS(float cosTheta, float g)
{
    float g2 = g * g;
    float num = 3.0f * (1.0f - g2) * (1.0f + cosTheta * cosTheta);
    float denom = (8.0f * PI) * (2.0f + g2) * pow(max(1e-4f, 1.0f + g2 - 2.0f * g * cosTheta), 1.5f);
    return num / denom;
}

// Two-lobe Mie phase: narrow forward peak + wider glow
// This ensures a visible halo even on short (near-zenith) paths,
// while the wider lobe doesn't accumulate as aggressively on long
// horizon paths because it's spread over a larger solid angle.
inline float PhaseMieTwoLobe(float cosTheta)
{
    float p1 = PhaseMieCS(cosTheta, ATMOS_MIE_G_PRIMARY);
    float p2 = PhaseMieCS(cosTheta, ATMOS_MIE_G_SECONDARY);
    return lerp(p1, p2, ATMOS_MIE_LOBE2_WEIGHT);
}

// ============================================================================
// Ray-sphere intersection (origin = sphere center)
// ============================================================================
inline bool RaySphereIntersect(float3 ro, float3 rd, float radius, out float t0, out float t1)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) { t0 = 0.0f; t1 = 0.0f; return false; }
    disc = sqrt(disc);
    t0 = -b - disc;
    t1 = -b + disc;
    return true;
}

// ============================================================================
// Transmittance from P toward L to atmosphere boundary
// ============================================================================
inline float3 TransmittanceToSun(float3 P, float3 L, float Rb, float Rt)
{
    float t0, t1;
    if (!RaySphereIntersect(P, L, Rt, t0, t1)) return float3(1, 1, 1);

    float tMax = t1;
    if (tMax <= 0.0f) return float3(1, 1, 1);
    float tMin = max(0.0f, t0);

    float ds = (tMax - tMin) / (float)ATMOS_LIGHT_STEPS;
    float3 od = float3(0, 0, 0);

    for (int i = 0; i < ATMOS_LIGHT_STEPS; i++)
    {
        float t = tMin + ((float)i + 0.5f) * ds;
        float3 Q = P + L * t;
        float alt = length(Q) - Rb;

        if (alt < 0.0f) return float3(0, 0, 0);

        MediumSample med = SampleMedium(alt);
        od += med.extinction * ds;
    }

    return exp(-od);
}

// ============================================================================
// Main scattering integral
// ============================================================================
float3 IntegrateScattering(float3 viewDir, float3 sunDir, out float3 transmittanceOut)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;

    float3 O = float3(0, Rb + 0.0002f, 0);
    float3 V = SafeNormalize(viewDir);
    float3 L = SafeNormalize(sunDir);

    float tV0, tV1;
    if (!RaySphereIntersect(O, V, Rt, tV0, tV1))
    {
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    float tMin = max(0.0f, tV0);
    float tMax = tV1;

    float tG0, tG1;
    if (RaySphereIntersect(O, V, Rb, tG0, tG1))
    {
        if (tG0 > 0.0f) tMax = min(tMax, tG0);
    }

    if (tMax <= tMin)
    {
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    // --- Non-uniform step distribution ---
    // Use sqrt-spaced steps: denser near the observer (low altitude, high density)
    // and sparser in the upper atmosphere. This improves horizon accuracy
    // without needing more total steps.
    float totalDist = tMax - tMin;

    float cosTheta = dot(V, L);
    float phR = PhaseRayleigh(cosTheta);
    float phM = PhaseMieTwoLobe(cosTheta);

    float3 totalInScatter = float3(0, 0, 0);
    float3 throughput     = float3(1, 1, 1);

    for (int i = 0; i < ATMOS_VIEW_STEPS; i++)
    {
        // sqrt spacing: samples cluster near the start of the ray
        float u0 = (float)i / (float)ATMOS_VIEW_STEPS;
        float u1 = (float)(i + 1) / (float)ATMOS_VIEW_STEPS;
        float s0 = u0 * u0;
        float s1 = u1 * u1;
        float tMid = tMin + (s0 + s1) * 0.5f * totalDist;
        float ds   = (s1 - s0) * totalDist;

        float3 P = O + V * tMid;
        float alt = length(P) - Rb;

        if (alt < 0.0f) break;

        MediumSample med = SampleMedium(alt);

        float3 segTr = exp(-med.extinction * ds);

        float3 sunTr = TransmittanceToSun(P, L, Rb, Rt);

        // Earth shadow
        float3 Pnorm = SafeNormalize(P);
        float sunCosZ = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f, 1.0f - (Rb * Rb) / dot(P, P)));
        float earthShadow = (sunCosZ > cosHorizon) ? 1.0f : 0.0f;

        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;

        // Analytically integrated in-scatter over the segment
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

    transmittanceOut = throughput;
    return totalInScatter;
}

// Full-path transmittance from observer along a direction
float3 AtmosphericTransmittance(float3 dir)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;

    float3 O = float3(0, Rb + 0.0002f, 0);
    float3 D = SafeNormalize(dir);

    float t0, t1;
    if (!RaySphereIntersect(O, D, Rt, t0, t1)) return float3(1, 1, 1);

    float tMax = t1;
    if (tMax <= 0.0f) return float3(1, 1, 1);
    float tMin = max(0.0f, t0);

    float ds = (tMax - tMin) / (float)ATMOS_VIEW_STEPS;
    float3 od = float3(0, 0, 0);

    for (int i = 0; i < ATMOS_VIEW_STEPS; i++)
    {
        float t = tMin + ((float)i + 0.5f) * ds;
        float3 Q = O + D * t;
        float alt = length(Q) - Rb;
        if (alt < 0.0f) return float3(0, 0, 0);

        MediumSample med = SampleMedium(alt);
        od += med.extinction * ds;
    }

    return exp(-od);
}

// ============================================================================
// Sun limb darkening (Neckel & Labs 1994 — simplified)
// ============================================================================
inline float3 LimbDarkening(float mu)
{
    float u = 1.0f - mu;
    float u2 = u * u;
    float3 ld;
    ld.x = 1.0f - 0.47f * u - 0.23f * u2;
    ld.y = 1.0f - 0.60f * u - 0.20f * u2;
    ld.z = 1.0f - 0.78f * u - 0.12f * u2;
    return max(ld, 0.0f);
}

// ============================================================================
// SunState
// ============================================================================
inline SunState ComputeSunState()
{
    SunState S;

    float3 d;
    float  elev;
    GetSunDirAndElev(d, elev);

    float thetaMax = (SUN_ANGULAR_DEG * 0.5f) * DEG2RAD;

    S.dirWS       = d;
    S.elevRad     = elev;
    S.cosThetaMax = cos(thetaMax);
    S.omega       = GetSunSolidAngle(thetaMax);

    float horizonRad = SUN_HORIZON_DEG * DEG2RAD;
    S.visible = (S.elevRad > horizonRad) ? 1.0f : 0.0f;

    float3 Tr = AtmosphericTransmittance(S.dirWS);
    S.tint    = SUN_COLOR_VAL * Tr;

    S.pdf      = (S.visible > 0.0f) ? (1.0f / S.omega) : 0.0f;
    S.radiance = (S.visible > 0.0f) ? (S.tint * SUN_INTENSITY_VAL / S.omega) : float3(0, 0, 0);

    return S;
}

// ============================================================================
// Public API — Sun
// ============================================================================
SunSampleResult SampleSun(float2 u)
{
    SunState S = ComputeSunState();

    SunSampleResult r;
    r.dist = SUN_DIST_INF;

    if (S.pdf <= 0.0f)
    {
        r.direction = S.dirWS;
        r.radiance  = float3(0, 0, 0);
        r.pdf       = 0.0f;
        return r;
    }

    float z = 1.0f - u.y * (1.0f - S.cosThetaMax);
    float sinTheta = sqrt(max(0.0f, 1.0f - z * z));
    float phi = TAU * u.x;

    float3 localDir = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, z);

    float3 T, B;
    GetOrthoBasis(S.dirWS, T, B);

    r.direction = SafeNormalize(localDir.x * T + localDir.y * B + localDir.z * S.dirWS);
    r.pdf       = S.pdf;

#if SUN_LIMB_DARKENING
    float cosAngle = dot(r.direction, S.dirWS);
    float mu = saturate((cosAngle - S.cosThetaMax) / max(1e-6f, 1.0f - S.cosThetaMax));
    mu = sqrt(mu);
    float3 ld = LimbDarkening(mu);
    r.radiance = S.radiance * ld / 0.85f;
#else
    r.radiance = S.radiance;
#endif

    return r;
}

float GetSunPdf(float3 rayDir)
{
    SunState S = ComputeSunState();
    if (S.pdf <= 0.0f) return 0.0f;

    float3 d = SafeNormalize(rayDir);
    return (dot(d, S.dirWS) >= S.cosThetaMax) ? S.pdf : 0.0f;
}

float3 EvaluateSun(float3 rayDir)
{
    SunState S = ComputeSunState();
    if (S.pdf <= 0.0f) return float3(0, 0, 0);

    float3 d = SafeNormalize(rayDir);
    float cosAngle = dot(d, S.dirWS);

    if (cosAngle < S.cosThetaMax) return float3(0, 0, 0);

#if SUN_LIMB_DARKENING
    float mu = saturate((cosAngle - S.cosThetaMax) / max(1e-6f, 1.0f - S.cosThetaMax));
    mu = sqrt(mu);
    float3 ld = LimbDarkening(mu);
    return S.radiance * ld / 0.85f;
#else
    return S.radiance;
#endif
}

// ============================================================================
// Stars
// ============================================================================

inline float3 RotateAroundAxis(float3 v, float3 axis, float angle)
{
    float s = sin(angle), c = cos(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
}

// Star color from approximate B-V color index
inline float3 StarColor(float rand01)
{
    float bv = lerp(-0.2f, 1.4f, rand01);

    float3 col;
    float t;
    if (bv < 0.0f)
    {
        t = saturate((bv + 0.3f) / 0.3f);
        col = lerp(float3(0.65f, 0.75f, 1.0f), float3(0.80f, 0.85f, 1.0f), t);
    }
    else if (bv < 0.4f)
    {
        t = bv / 0.4f;
        col = lerp(float3(0.95f, 0.95f, 1.0f), float3(1.0f, 0.96f, 0.88f), t);
    }
    else if (bv < 0.8f)
    {
        t = (bv - 0.4f) / 0.4f;
        col = lerp(float3(1.0f, 0.96f, 0.88f), float3(1.0f, 0.86f, 0.65f), t);
    }
    else
    {
        t = saturate((bv - 0.8f) / 0.6f);
        col = lerp(float3(1.0f, 0.86f, 0.65f), float3(1.0f, 0.70f, 0.45f), t);
    }

    return col;
}

// Atmospheric scintillation — elevation dependent, TAA stable
inline float Scintillation(float seed, float elevFactor, float frameCount)
{
    float strength = SKY_STAR_SCINTILLATION * (1.0f - elevFactor * elevFactor);

    float phase = seed * 100.0f;
    float t = frameCount * 0.002f;
    float flicker = sin(t * 1.7f + phase)
                  * sin(t * 2.9f + phase * 0.7f)
                  * sin(t * 0.5f + phase * 1.3f);

    return 1.0f - strength * 0.5f * (flicker + 1.0f);
}

// Single star layer
// sunElevDeg: sun elevation in degrees (used for per-star twilight fade)
float3 EvaluateStarLayer(float3 vStar, float gridScale, float density,
                         float brightnessScale, float sunElevDeg)
{
    float2 uv;
    uv.x = atan2(vStar.z, vStar.x) / TAU + 0.5f;
    uv.y = acos(clamp(vStar.y, -1.0f, 1.0f)) / PI;

    // Pole compensation: at the poles, UV cells cover a tiny solid angle,
    // causing stars to cluster and appear shrunken. Two corrections:
    //   1) Aggressively reject stars where cells are small (raised floor)
    //   2) Scale the distance metric so surviving stars keep their angular size
    float sinTheta = sqrt(max(1e-6f, 1.0f - vStar.y * vStar.y));
    float poleCompensation = saturate(sinTheta);

    // Hard cutoff very near the pole where cells degenerate
    if (poleCompensation < 0.08f) return float3(0, 0, 0);

    float2 g    = uv * float2(gridScale, gridScale * 0.5f);
    float2 cell = floor(g);
    float2 f    = frac(g);

    float r0 = Hash12(cell);
    // Floor raised to 0.3 — at sinTheta < 0.3 (within ~17° of pole),
    // stars are progressively culled. This prevents visible clustering.
    float adjustedDensity = 1.0f - (1.0f - density) * poleCompensation;
    float present = step(adjustedDensity, r0);
    if (present < 0.5f) return float3(0, 0, 0);

    // Jittered position within cell
    float2 starPos = Hash22(cell + 5.0f);
    starPos = lerp(0.15f, 0.85f, starPos);

    // Anisotropic pole correction for distance:
    // Near the pole, grid cells are compressed in x (longitude) but not y (latitude).
    // Without correction, length(f - starPos) treats both axes equally, making
    // stars appear stretched vertically. We scale delta.x by sinTheta so that
    // the distance is computed in angular-proportional coordinates — this makes
    // the Gaussian point spread circular on the sky at all latitudes.
    float2 delta = f - starPos;
    delta.x *= max(0.15f, poleCompensation);
    float dist = length(delta);

    // Power-law magnitude: few bright, many dim
    float magRand = Hash12(cell + 13.0f);
    float magnitude = pow(magRand, 2.5f);

    // --- Per-star twilight fade ---
    // Dim stars (magnitude~0): fade threshold at deep twilight (-SKY_TWILIGHT_DEG)
    // Bright stars (magnitude~1): linger until sun is at -6 + SKY_STAR_DAWN_LINGER degrees
    // This creates the natural effect of a few bright stars persisting into dawn/dusk
    // while the bulk of the sky gradually empties.
    float dimThreshold    = -SKY_TWILIGHT_DEG;                        // e.g. -18°
    float brightThreshold = -6.0f + SKY_STAR_DAWN_LINGER;            // e.g. +2°
    float starThreshold   = lerp(dimThreshold, brightThreshold, magnitude);
    float fadeRange        = max(1.0f, abs(starThreshold - dimThreshold) * 0.4f + 2.0f);
    float starTwilight    = Smooth01(saturate((starThreshold - sunElevDeg) / fadeRange));

    if (starTwilight <= 0.0f) return float3(0, 0, 0);

    float baseRadius = lerp(0.05f, 0.08f, magnitude);

    // Gaussian core + soft halo
    float core = exp(-dist * dist / max(1e-6f, baseRadius * baseRadius * 0.08f));
    float halo = magnitude * exp(-dist * dist / max(1e-6f, baseRadius * baseRadius * 0.5f)) * 0.3f;
    float brightness = saturate(core + halo);

    float3 col = StarColor(Hash12(cell + 29.0f));

    float elevFactor = saturate(dot(vStar, WORLD_UP));
    float scint = Scintillation(Hash12(cell + 37.0f), elevFactor, (float)SUN_FRAMECOUNT);

    return brightness * magnitude * col * brightnessScale * scint * starTwilight;
}

// Multi-layer star field
float3 EvaluateStars(float3 rayDir, float elevDeg)
{
    float3 v = SafeNormalize(rayDir);

    // Sidereal rotation
    float solarH = GetSolarTimeHours();
    float sidFrac = frac((solarH / 24.0f) * SKY_SIDEREAL_RATIO);
    float3 vStar = RotateAroundAxis(v, WORLD_UP, -TAU * sidFrac);

    // Fade out below horizon (keep this — stars shouldn't render underground)
    float upDot = dot(v, WORLD_UP);
    float horizonW = Smooth01(saturate(upDot / 0.02f));

    if (horizonW <= 0.0f) return float3(0, 0, 0);

    // Note: twilight fade is now per-star inside EvaluateStarLayer,
    // so bright stars naturally persist into dawn/dusk while dim ones vanish first.

    float3 stars = float3(0, 0, 0);

    // Layer 0: Sparse bright stars (these linger longest at dawn)
    stars += EvaluateStarLayer(vStar, SKY_STAR_GRID * 0.5f,
                               0.985f, 1.6f, elevDeg);

    // Layer 1: Medium density
    stars += EvaluateStarLayer(vStar, SKY_STAR_GRID * 1.0f,
                               SKY_STAR_DENSITY, 1.0f, elevDeg);

#if SKY_STAR_LAYERS >= 3
    // Layer 2: Dense dim background (vanishes first at dawn)
    stars += EvaluateStarLayer(vStar, SKY_STAR_GRID * 2.2f,
                               0.994f, 0.35f, elevDeg);
#endif

    return stars * SKY_STAR_INTENSITY * SKY_STAR_SCALE * horizonW;
}

// ============================================================================
// Public API — Sky
// ============================================================================
float3 EvaluateSky(float3 rayDir)
{
    SunState S = ComputeSunState();

    float3 v     = SafeNormalize(rayDir);
    float  upDot = dot(v, WORLD_UP);
    float  elevDeg = S.elevRad * RAD2DEG;

    bool isGround = (upDot <= 0.0f);
    float3 vEval = isGround ? SafeNormalize(float3(v.x, 1e-4f, v.z)) : v;

    // Physical atmospheric scattering
    float3 viewTr;
    float3 scatter = IntegrateScattering(vEval, S.dirWS, viewTr);

    float3 daySky = scatter * SKY_INTENSITY;

    // Night sky
    float tw = Smooth01(saturate((elevDeg + SKY_TWILIGHT_DEG) / SKY_TWILIGHT_DEG));

    float mu = saturate(dot(vEval, WORLD_UP));
    float3 nightBase = SKY_NIGHT_BASE * lerp(1.6f, 1.0f, pow(mu, 0.7f));
    nightBase *= SKY_INTENSITY;

    float3 stars = EvaluateStars(v, elevDeg);

    float3 sky = lerp(nightBase, daySky, tw) + stars;

    if (isGround)
    {
        float t = Smooth01(saturate((-upDot) / 0.35f));
        return lerp(sky, sky * SKY_GROUND_DARKEN, t);
    }

    return sky;
}
