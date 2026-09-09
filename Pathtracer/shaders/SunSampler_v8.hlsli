#define TAU     (2.0f * PI)
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

#ifndef SUN_FRAMECOUNT
#define SUN_FRAMECOUNT time
#endif

#ifndef SUN_LATITUDE_DEG
#define SUN_LATITUDE_DEG   48.5200f
#endif

#ifndef SUN_LONGITUDE_DEG
#define SUN_LONGITUDE_DEG  11.4050f
#endif

#ifndef SUN_DAY_OF_YEAR
#define SUN_DAY_OF_YEAR    172.0f
#endif

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

#ifndef ATMOS_BOTTOM_RADIUS
#define ATMOS_BOTTOM_RADIUS     6360.0f
#endif

#ifndef ATMOS_TOP_RADIUS
#define ATMOS_TOP_RADIUS        6420.0f
#endif

// TransmittanceToSun tests its hard geometric planet block against a sphere
// this many km below ATMOS_BOTTOM_RADIUS. A view sample on the planet surface
// then sits above the block sphere, so its sun ray is not hard zeroed the
// moment the sun touches the geometric horizon (the dark band that otherwise
// appears along the horizon at sunset). cosHorizon and
// SunDiskFractionAboveHorizon keep using the true radius, so the visible
// terminator is unchanged; the hard cutoff just sinks deep enough into the
// shadow that the earth shadow smoothstep has already faded the sample's
// contribution to zero there. Replaces the old global 5 km radius drop.
#ifndef ATMOS_SUN_BLOCK_BIAS_KM
#define ATMOS_SUN_BLOCK_BIAS_KM 5.0f
#endif

#ifndef ATMOS_RAYLEIGH_SCATTER
#define ATMOS_RAYLEIGH_SCATTER  float3(0.005802f, 0.013558f, 0.033100f)
#endif

#ifndef ATMOS_RAYLEIGH_SCALE_H
#define ATMOS_RAYLEIGH_SCALE_H  8.0f
#endif

#ifndef ATMOS_MIE_SCATTER_BASE
#define ATMOS_MIE_SCATTER_BASE  float3(0.003996f, 0.003996f, 0.003996f)
#endif

#ifndef ATMOS_MIE_EXTINCT_BASE
#define ATMOS_MIE_EXTINCT_BASE  float3(0.004440f, 0.004440f, 0.004440f)
#endif

#ifndef ATMOS_MIE_SCALE_H
#define ATMOS_MIE_SCALE_H       1.2f
#endif

//two-lobe Mie, primary narrow halo, secondary soft glow
#ifndef ATMOS_MIE_G_PRIMARY
#define ATMOS_MIE_G_PRIMARY     0.76f
#endif

#ifndef ATMOS_MIE_G_SECONDARY
#define ATMOS_MIE_G_SECONDARY   0.35f
#endif

//secondary lobe weight, 0=single lobe, 1=all secondary
#ifndef ATMOS_MIE_LOBE2_WEIGHT
#define ATMOS_MIE_LOBE2_WEIGHT  0.15f
#endif

//turbidity on Mie, 1=very clear, 2-3=clear, 5+=hazy
#ifndef SUN_TURBIDITY
#define SUN_TURBIDITY           2.0f
#endif

#ifndef ATMOS_OZONE_ABSORPTION
#define ATMOS_OZONE_ABSORPTION  float3(0.000650f, 0.001881f, 0.000085f)
#endif

#ifndef ATMOS_VIEW_STEPS
#define ATMOS_VIEW_STEPS        12
#endif

#ifndef ATMOS_LIGHT_STEPS
#define ATMOS_LIGHT_STEPS       8
#endif

#ifndef ATMOS_SOLAR_IRRADIANCE
#define ATMOS_SOLAR_IRRADIANCE  float3(1.0f, 1.0f, 1.0f)
#endif

// 1.0 = physical now that the Psi_ms LUT supplies real multiple scattering
// (the old 1.1 was the flat stand-in). Boost applies to single scatter only.
#ifndef ATMOS_MULTI_SCATTER_FACTOR
#define ATMOS_MULTI_SCATTER_FACTOR  1.0f
#endif

#ifndef SKY_INTENSITY
#define SKY_INTENSITY           6.0f
#endif

#ifndef SKY_TWILIGHT_DEG
#define SKY_TWILIGHT_DEG        18.0f
#endif

// Star skybox knobs are editor-driven via the SunSettings cbuffer tail.
#define SKY_STAR_INTENSITY      skyStarIntensity
#define SKY_STAR_GAMMA          skyStarGamma
#define SKY_STAR_THRESHOLD      skyStarThreshold

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

// Per-channel OD scaling that hides stars under bright daytime scatter.
// The gate MUST be atmospheric scatter, not sun elevation — orbital daytime
// views still see stars (scatter ≈ 0 there).
#ifndef SKY_STAR_SCATTER_SHIELD
#define SKY_STAR_SCATTER_SHIELD 1000.0f
#endif

// gSkyStars (equirect, celestial frame) bound in Includes_v8.hlsli at t40.
// Sampled via WorldToCelestial(rayDir) so it rotates with sidereal time.
#ifndef SKY_STAR_TEXTURE_SRGB
#define SKY_STAR_TEXTURE_SRGB 0
#endif

#ifndef SKY_STAR_TEXTURE_FLIP_U
#define SKY_STAR_TEXTURE_FLIP_U 0
#endif

#define SKY_STAR_LOD_BIAS       skyStarLodBias

#ifndef ATMOS_GROUND_ALBEDO
#define ATMOS_GROUND_ALBEDO     float3(0.4f, 0.4f, 0.4f)
#endif

#ifndef WORLD_NORTH
#define WORLD_NORTH             normalize(float3(0, 0, 1))
#endif

static const float3 WORLD_UP = float3(0, 1, 0);

// World Y = altitude above surface tangent point; world XZ = tangent-plane.
#ifndef WORLD_UNITS_PER_KM
#define WORLD_UNITS_PER_KM      1000.0f
#endif

static const float SKY_OBSERVER_MIN_RADIUS = ATMOS_BOTTOM_RADIUS + 0.0002f;

// Per-invocation observer position in planet space. Raygen sets this near
// the top; default = surface tangent point for callers that skip the set.
static float3 g_skyObserverPlanet = float3(0.0f, SKY_OBSERVER_MIN_RADIUS, 0.0f);

#define SKY_UNDERGROUND_EPS_KM 0.001f
static bool g_skyObserverUnderground = false;

inline bool SkyObserverIsUnderground() { return g_skyObserverUnderground; }

inline float3 WorldToPlanet(float3 worldPos)
{
    float scale = 1.0f / WORLD_UNITS_PER_KM;
    return float3(worldPos.x * scale,
                  ATMOS_BOTTOM_RADIUS + worldPos.y * scale,
                  worldPos.z * scale);
}

inline void SetSkyObserver(float3 worldPos)
{
    float3 P = WorldToPlanet(worldPos);
    float r = length(P);
    g_skyObserverUnderground =
        (r < ATMOS_BOTTOM_RADIUS - SKY_UNDERGROUND_EPS_KM);
    if (r < SKY_OBSERVER_MIN_RADIUS) P *= SKY_OBSERVER_MIN_RADIUS / max(1e-6f, r);
    g_skyObserverPlanet = P;
}

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
    float3 scatterR;
    float3 scatterM;
    float3 extinction;
};

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

inline float Hash12(float2 p)
{
    float3 p3 = frac(float3(p.xyx) * float3(0.1031f, 0.1030f, 0.0973f));
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

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

// Local Sidereal Time built from the sun's GetSolarTimeHours so stars lock
// to sun motion (and SUN_NIGHT_SPEEDUP warping). SKY_SIDEREAL_RATIO ≈ Earth's
// 1.00273 makes stars drift ~4 min/day ahead of the sun.
inline float GetLSTRad()
{
    return GetSolarTimeHours() * SKY_SIDEREAL_RATIO * (TAU / 24.0f)
         + SUN_LONGITUDE_DEG * DEG2RAD;
}

// World → celestial equatorial (RA fixed). Inverse of GetSunDirAndElev's
// projection so sun and stars share one time base + transform. Output:
// y = celestial pole, (x,z) = celestial equator. EvaluateStars takes
// Dec = asin(y), RA = atan2(z, x).
float3 WorldToCelestial(float3 vWorld)
{
    float latRad = SUN_LATITUDE_DEG * DEG2RAD;
    float cosL = cos(latRad), sinL = sin(latRad);

    float3 Nw = SafeNormalize(float3(WORLD_NORTH.x, 0.0f, WORLD_NORTH.z));
    float3 Ew = SafeNormalize(cross(WORLD_UP, Nw));
    float east  = dot(vWorld, Ew);
    float north = dot(vWorld, Nw);
    float up    = dot(vWorld, WORLD_UP);

    // ENU → instantaneous equatorial (matches v_eq = (cosδ sinH, cosδ cosH, sinδ)).
    float X_i = east;
    float Y_i = -sinL * north + cosL * up;
    float Z_i =  cosL * north + sinL * up;

    // Instantaneous → RA-fixed: rotate by LST around the celestial pole.
    float lst = GetLSTRad();
    float cl = cos(lst), sl = sin(lst);
    float X_f =  sl * X_i + cl * Y_i;
    float Y_f = -cl * X_i + sl * Y_i;
    float Z_f =  Z_i;

    // Pack with pole on +Y for the equirect lookup.
    return float3(X_f, Z_f, Y_f);
}

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

inline float PhaseRayleigh(float cosTheta)
{
    return (3.0f / (16.0f * PI)) * (1.0f + cosTheta * cosTheta);
}

// Cornette-Shanks
inline float PhaseMieCS(float cosTheta, float g)
{
    float g2 = g * g;
    float num = 3.0f * (1.0f - g2) * (1.0f + cosTheta * cosTheta);
    float denom = (8.0f * PI) * (2.0f + g2) * pow(max(1e-4f, 1.0f + g2 - 2.0f * g * cosTheta), 1.5f);
    return num / denom;
}

inline float PhaseMieTwoLobe(float cosTheta)
{
    float p1 = PhaseMieCS(cosTheta, ATMOS_MIE_G_PRIMARY);
    float p2 = PhaseMieCS(cosTheta, ATMOS_MIE_G_SECONDARY);
    return lerp(p1, p2, ATMOS_MIE_LOBE2_WEIGHT);
}

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

// Sun disk fraction above local horizon. Smoothstep width = sun angular
// radius so the gate matches actual disk geometry (replaces the old
// decoupled softness slider).
inline float SunDiskFractionAboveHorizon(float sunCosZ, float cosHorizon)
{
    float kSinSunRadius = sin(SUN_ANGULAR_DEG * 0.5f * DEG2RAD)*sqrt(max(0.0f,1-cosHorizon*cosHorizon));
    return smoothstep(cosHorizon - kSinSunRadius,
                      cosHorizon + kSinSunRadius, sunCosZ);
}


#define SKY_TRANSMITTANCE_LUT_W 256.0f
#define SKY_TRANSMITTANCE_LUT_H 64.0f

// x in [0,1] <-> texel-center-inset texture coord (Bruneton).
inline float LutCoordFromUnitRange(float x, float texSize)
{
    return 0.5f / texSize + x * (1.0f - 1.0f / texSize);
}

inline float LutUnitRangeFromCoord(float u, float texSize)
{
    return saturate((u - 0.5f / texSize) / (1.0f - 1.0f / texSize));
}

// Bruneton mapping: x_r = rho/H, x_mu = (d - dmin)/(dmax - dmin) with
// d = distance to the atmosphere top along the ray, rho = sqrt(r²-Rb²),
// H = sqrt(Rt²-Rb²). Follows the horizon line, so the steep near-horizon
// falloff gets texel density exactly where it changes fastest. Domain is
// mu >= horizon (rays that reach space); below-horizon rays are rejected
// by the geometric block checks before lookup, and the sliver that
// survives the sunken block sphere clamps to the darkest stored column.
inline float2 TransmittanceLutUvFromRMu(float r, float mu)
{
    const float Rb = ATMOS_BOTTOM_RADIUS;
    const float Rt = ATMOS_TOP_RADIUS;
    const float H  = sqrt(max(1e-3f, Rt * Rt - Rb * Rb));
    float rho  = sqrt(max(0.0f, r * r - Rb * Rb));
    float disc = max(0.0f, r * r * (mu * mu - 1.0f) + Rt * Rt);
    float d    = max(0.0f, -r * mu + sqrt(disc));
    float dMin = Rt - r;
    float dMax = rho + H;
    float xMu  = saturate((d - dMin) / max(1e-6f, dMax - dMin));
    float xR   = saturate(rho / H);
    return float2(LutCoordFromUnitRange(xMu, SKY_TRANSMITTANCE_LUT_W),
                  LutCoordFromUnitRange(xR,  SKY_TRANSMITTANCE_LUT_H));
}

// Inverse of the mapping above — used by the bake pass only.
inline void TransmittanceLutRMuFromUv(float2 uv, out float r, out float mu)
{
    const float Rb = ATMOS_BOTTOM_RADIUS;
    const float Rt = ATMOS_TOP_RADIUS;
    const float H  = sqrt(max(1e-3f, Rt * Rt - Rb * Rb));
    float xMu = LutUnitRangeFromCoord(uv.x, SKY_TRANSMITTANCE_LUT_W);
    float xR  = LutUnitRangeFromCoord(uv.y, SKY_TRANSMITTANCE_LUT_H);
    float rho = H * xR;
    r = sqrt(rho * rho + Rb * Rb);
    float dMin = Rt - r;
    float dMax = rho + H;
    float d    = dMin + xMu * (dMax - dMin);
    mu = (d <= 1e-4f) ? 1.0f
       : clamp((H * H - rho * rho - d * d) / (2.0f * r * d), -1.0f, 1.0f);
}

// Reference integral (r, mu) -> transmittance to the atmosphere top.
// Bake-side only; the runtime path samples the LUT instead.
inline float3 ComputeTransmittanceToTopRMu(float r, float mu)
{
    const float Rb = ATMOS_BOTTOM_RADIUS;
    const float Rt = ATMOS_TOP_RADIUS;
    float disc = max(0.0f, r * r * (mu * mu - 1.0f) + Rt * Rt);
    float dTop = max(0.0f, -r * mu + sqrt(disc));

    const int N = 64;
    float  ds = dTop / (float)N;
    float3 od = float3(0, 0, 0);
    [loop]
    for (int i = 0; i < N; ++i)
    {
        float t  = ((float)i + 0.5f) * ds;
        float rT = sqrt(max(Rb * Rb, r * r + t * t + 2.0f * r * t * mu));
        MediumSample med = SampleMedium(max(0.0f, rT - Rb));
        od += med.extinction * ds;
    }
    return exp(-od);
}

// Rb/Rt params kept for call-site compatibility; the LUT itself is baked
// for ATMOS_BOTTOM_RADIUS / ATMOS_TOP_RADIUS (every caller passes those).
inline float3 TransmittanceToSun(float3 P, float3 L, float Rb, float Rt)
{
    float t0, t1;
    if (!RaySphereIntersect(P, L, Rt, t0, t1)) return float3(1, 1, 1);

    float tMax = t1;
    if (tMax <= 0.0f) return float3(1, 1, 1);

    // Geometric planet block — must return zero, not the partial integral.
    // A partial would bleed bright yellow into the earth shadow band; the
    // caller's SunDiskFractionAboveHorizon smooths the visibility edge. The
    // block sphere sinks ATMOS_SUN_BLOCK_BIAS_KM below Rb so a sample on the
    // mesh surface clears it at the sunset horizon.
    float tG0, tG1;
    float RbBlock = Rb - ATMOS_SUN_BLOCK_BIAS_KM;
    if (RaySphereIntersect(P, L, RbBlock, tG0, tG1) && tG0 > 0.0f && tG0 < tMax)
        return float3(0, 0, 0);

    // From-space rays: advance to the atmosphere entry so (r, mu) lands in
    // the LUT domain. The mesh-bias band slightly below Rb clamps to ground.
    float r = length(P);
    if (r > Rt && t0 > 0.0f)
    {
        P += L * t0;
        r  = Rt;
    }
    r = clamp(r, Rb, Rt);
    float mu = clamp(dot(P, L) / max(r, 1e-4f), -1.0f, 1.0f);

    return g_skyTransmittanceLUT.SampleLevel(
        g_sampler_LUT, TransmittanceLutUvFromRMu(r, mu), 0).rgb;
}

//====================================
//ATMOSPHERIC MULTIPLE SCATTERING — PER-FRAME 2D LUT (Hillaire 2020)
//====================================
//g_skyMultiScatterLUT (t51, 32x32 RGBA16F, baked per frame by
//Pass_skylut_bake_v8.hlsl::mainMultiScatter) stores Psi_ms(sunCosZ, r): the
//radiance a point at radius r receives from second-and-higher-order
//atmospheric scattering, per unit sun irradiance, under the isotropic-phase
//approximation ("A Scalable and Production Ready Sky and Atmosphere
//Rendering Technique", eq. 10). Runtime usage per march step:
//    rate_ms = (scatterR + scatterM) * PsiMS * sunIrradiance
//with NO phase function, NO sun transmittance and NO earth-shadow factor —
//all three are integrated inside the LUT. This is what keeps twilight and
//the horizon lit (and desaturated) after the direct term dies; the old flat
//ATMOS_MULTI_SCATTER_FACTOR stand-in kept single-scatter's saturation and
//phase signature at every range. The factor remains as an artistic boost on
//the single-scatter term only (default now 1.0 = physical).
#define SKY_MS_LUT_SIZE 32.0f

// Spend the existing 32x32 table on the lower atmosphere and terminator.
// Linear mu put sunset between samples almost four degrees apart.
inline float2 MultiScatterLutUnitFromRMu(float r,float mu)
{
    return float2(.5f+.5f*sign(mu)*sqrt(abs(mu)),
        sqrt(saturate((r-ATMOS_BOTTOM_RADIUS)/(ATMOS_TOP_RADIUS-ATMOS_BOTTOM_RADIUS))));
}

// Use the centroid of the visible solar disk for a single shadow/transmission
// direction. Multiplying a planet-blocked centre ray by a nonzero disk fraction
// still traces through the ground at sunset. This small-disk approximation
// stays continuous as the centre sets, and does not change the disk's energy.
inline float3 VisibleSunDirection(float3 P,float3 L)
{
    float3 up=normalize(P);float mu=dot(up,L);
    float cosH=-sqrt(max(0.0f,1-ATMOS_BOTTOM_RADIUS*ATMOS_BOTTOM_RADIUS/dot(P,P)));
    float radius=sin(SUN_ANGULAR_DEG*.5f*DEG2RAD);
    float x=(cosH-mu)/max(radius*sqrt(max(0.0f,1-cosH*cosH)),1e-7f);
    if(x<=-1)return L;
    float diskCentroid;
    if(x>=.99f)diskCentroid=lerp(.994f,1.0f,saturate((x-.99f)*100));
    else {
        float root=sqrt(max(0.0f,1-x*x));
        diskCentroid=(2.0f/3.0f)*root*root*root/max(acos(x)-x*root,1e-7f);
    }
    float3 lift=up-L*mu;
    lift*=rsqrt(max(dot(lift,lift),1e-12f));
    return normalize(L+lift*(radius*diskCentroid));
}
inline void MultiScatterLutRMuFromUnit(float2 unit,out float r,out float mu)
{
    float x=unit.x*2-1;
    mu=x*abs(x);
    r=ATMOS_BOTTOM_RADIUS+unit.y*unit.y*(ATMOS_TOP_RADIUS-ATMOS_BOTTOM_RADIUS);
}
inline float3 MultiScatterPsi(float r, float sunCosZ)
{
    float2 unit=MultiScatterLutUnitFromRMu(r,clamp(sunCosZ,-1.0f,1.0f));
    float2 uv = float2(LutCoordFromUnitRange(unit.x, SKY_MS_LUT_SIZE),
                       LutCoordFromUnitRange(unit.y, SKY_MS_LUT_SIZE));
    return g_skyMultiScatterLUT.SampleLevel(g_sampler_LUT, uv, 0).rgb;
}

// Implemented after CumulusCache is included. Only runtime volume transport
// enables this; clear-sky LUT bakes must not read their dependent cloud cache.
float CumulusAtmosphereSunVisibility(float3 P,float3 L);
// Sample a source point using the least-extinguished RGB channel. The shared
// proposal has support for all channels, and the RGB correction integrates a
// constant source exactly in expectation for the cell's fixed extinction.
void AtmosphereSourceQuadrature(float3 extinction,float ds,float u,out float distance,out float3 weight)
{
    float proposal=max(0.0f,min(extinction.x,min(extinction.y,extinction.z)));
    float tau=proposal*ds;
    float mass=tau<.001f ? ds*(1-.5f*tau+tau*tau/6) : (1-exp(-tau))/max(proposal,1e-20f);
    distance=tau<.001f ? ds*(u+.5f*tau*u*(u-1)) : -log(max(1e-20f,1-u*proposal*mass))/proposal;
    weight=mass*exp(-(extinction-proposal)*distance);
}
float3 IntegrateScattering(float3 viewDir, float3 sunDir,
                           out float3 transmittanceOut, out bool hitPlanetOut,
                           float maxDistanceKm = -1.0f,uint stepCount = ATMOS_VIEW_STEPS,
                           bool cloudShadows=false,float sampleJitter=-1.0f)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;

    float3 O = g_skyObserverPlanet;
    float3 V = SafeNormalize(viewDir);
    float3 L = SafeNormalize(sunDir);

    hitPlanetOut = false;

    float tV0, tV1;
    if (!RaySphereIntersect(O, V, Rt, tV0, tV1) || tV1 <= 0.0f)
    {
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    float tMin = max(0.0f, tV0);
    float tMax = maxDistanceKm >= 0.0f ? min(tV1, maxDistanceKm) : tV1;

    // Planet body terminates the ray; throughput is reused by EvaluatePlanetBody.
    float tG0, tG1;
    if (RaySphereIntersect(O, V, Rb, tG0, tG1) && tG0 > 0.0f && tG0 < tMax)
    {
        tMax = tG0;
        hitPlanetOut = true;
    }

    if (tMax <= tMin)
    {
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    float totalDist = tMax - tMin;

    float cosTheta = dot(V, L);
    float phR = PhaseRayleigh(cosTheta);
    float phM = PhaseMieTwoLobe(cosTheta);

    float3 totalInScatter = float3(0, 0, 0);
    float3 throughput     = float3(1, 1, 1);

    stepCount=clamp(stepCount,2u,64u);
    for (uint i = 0; i < stepCount; i++)
    {
        // sqrt spacing — clusters samples near the start of the ray.
        float u0 = (float)i / (float)stepCount;
        float u1 = (float)(i + 1) / (float)stepCount;
        float s0 = u0 * u0;
        float s1 = u1 * u1;
        float tMid = tMin + (s0 + s1) * 0.5f * totalDist;
        float ds   = (s1 - s0) * totalDist;

        float3 P = O + V * tMid;
        float  rP  = length(P);
        float  alt = max(0.0f, rP - Rb);

        MediumSample med = SampleMedium(alt);
        float3 segTr = exp(-med.extinction * ds);
        float3 sourceWeight=0;
        MediumSample sourceMed=med;
        if(sampleJitter>=0.0f) {
            float distance;
            AtmosphereSourceQuadrature(med.extinction,ds,frac(sampleJitter+float(i)*.61803398875f),distance,sourceWeight);
            P=O+V*(tMin+s0*totalDist+distance);
            rP=length(P);
            sourceMed=SampleMedium(max(0.0f,rP-Rb));
        }
        float3 visibleL=VisibleSunDirection(P,L);
        float3 sunTr = TransmittanceToSun(P, visibleL, Rb, Rt);

        float3 Pnorm = SafeNormalize(P);
        float sunCosZ = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f, 1.0f - (Rb * Rb) / dot(P, P)));
        float earthShadow = SunDiskFractionAboveHorizon(sunCosZ, cosHorizon);

        float3 scatterPhase = sourceMed.scatterR * phR + sourceMed.scatterM * phM;
        float3 scatterIso   = sourceMed.scatterR + sourceMed.scatterM;

        // Directional single scattering plus isotropic atmospheric multiple scattering.
        float3 sunIllum = ATMOS_SOLAR_IRRADIANCE * earthShadow * sunTr
                        * ATMOS_MULTI_SCATTER_FACTOR;
        if(cloudShadows && earthShadow>0.0f)
            sunIllum*=CumulusAtmosphereSunVisibility(P,visibleL);
        float3 psiMS = MultiScatterPsi(rP, sunCosZ);
        float3 rate = scatterPhase * sunIllum
                    + scatterIso * psiMS * ATMOS_SOLAR_IRRADIANCE;

        float3 scatterInteg;
        scatterInteg.x = (med.extinction.x > 1e-10f)
            ? rate.x * (1.0f - segTr.x) / med.extinction.x : rate.x * ds;
        scatterInteg.y = (med.extinction.y > 1e-10f)
            ? rate.y * (1.0f - segTr.y) / med.extinction.y : rate.y * ds;
        scatterInteg.z = (med.extinction.z > 1e-10f)
            ? rate.z * (1.0f - segTr.z) / med.extinction.z : rate.z * ds;

        if(sampleJitter>=0.0f)scatterInteg=sourceWeight*rate;
        totalInScatter += throughput * scatterInteg;

        throughput *= segTr;
    }

    // No trailing ATMOS_MULTI_SCATTER_FACTOR: the factor moved into the
    // per-step directional term, and the real multiple scattering now
    // comes from the Psi_ms LUT (which must NOT be double-boosted).

    transmittanceOut = throughput;
    return totalInScatter;
}

#ifndef ATMOS_AERIAL_VIEW_STEPS
#define ATMOS_AERIAL_VIEW_STEPS  4
#endif
#ifndef ATMOS_AERIAL_LIGHT_STEPS
#define ATMOS_AERIAL_LIGHT_STEPS 4
#endif

// Historical "cheap" variant — it existed to cap the inner-march step count
// for aerial perspective. The LUT-backed TransmittanceToSun is now a single
// fetch, so this is a plain alias kept for its call sites.
inline float3 TransmittanceToSunCheap(float3 P, float3 L, float Rb, float Rt)
{
    return TransmittanceToSun(P, L, Rb, Rt);
}

float3 ComputeAerialPerspective(float3 viewDir, float3 sunDir, float hitDistKm,
                                uint2 pixel,
                                out float3 transmittanceOut)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;
    float3 O = g_skyObserverPlanet;
    float3 V = SafeNormalize(viewDir);
    float3 L = SafeNormalize(sunDir);

    float tV0, tV1;
    if (!RaySphereIntersect(O, V, Rt, tV0, tV1) || tV1 <= 0.0f)
    {
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    float tMin = max(0.0f, tV0);
    float tMax = min(hitDistKm, tV1);

    if (tMax <= tMin)
    {
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    float totalDist = tMax - tMin;

    float cosTheta = dot(V, L);
    float phR = PhaseRayleigh(cosTheta);
    float phM = PhaseMieTwoLobe(cosTheta);

    float3 totalInScatter = float3(0, 0, 0);
    float3 throughput     = float3(1, 1, 1);

    uint seed = initRandomData(pixel, uint2(0, 0), (uint)time, 91u);

    // Uniform spacing — scene segments are short and roughly homogeneous.
    [unroll]
    for (int i = 0; i < ATMOS_AERIAL_VIEW_STEPS; i++)
    {
        float u0    = (float)i / (float)ATMOS_AERIAL_VIEW_STEPS;
        float u1    = (float)(i + 1) / (float)ATMOS_AERIAL_VIEW_STEPS;
        float xi    = RandomFloatSingle(seed);
        float tSamp = tMin + lerp(u0, u1, xi) * totalDist;
        float ds    = (u1 - u0) * totalDist;

        float3 P = O + V * tSamp;
        float alt = max(0.0f, length(P) - Rb);
        MediumSample med = SampleMedium(alt);

        float3 segTr = exp(-med.extinction * ds);

        // Smooth earth shadow — the binary version put a visible line in haze.
        float3 Pnorm = SafeNormalize(P);
        float sunCosZ = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f, 1.0f - (Rb * Rb) / dot(P, P)));
        float earthShadow = smoothstep(cosHorizon - 0.005f,
                                       cosHorizon + 0.005f, sunCosZ);

        float3 sunTr = TransmittanceToSunCheap(P, L, Rb, Rt);

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

    transmittanceOut = throughput;
    return totalInScatter;
}

float3 AtmosphericTransmittance(float3 dir)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;

    float3 O = g_skyObserverPlanet;
    float3 D = SafeNormalize(dir);

    float t0, t1;
    if (!RaySphereIntersect(O, D, Rt, t0, t1) || t1 <= 0.0f) return float3(1, 1, 1);

    float tMin = max(0.0f, t0);
    float tMax = t1;

    float tG0, tG1;
    if (RaySphereIntersect(O, D, Rb, tG0, tG1) && tG0 > 0.0f && tG0 < tMax)
        return float3(0, 0, 0);

    float ds = (tMax - tMin) / (float)ATMOS_VIEW_STEPS;
    float3 od = float3(0, 0, 0);

    for (int i = 0; i < ATMOS_VIEW_STEPS; i++)
    {
        float t = tMin + ((float)i + 0.5f) * ds;
        float3 Q = O + D * t;
        float alt = max(0.0f, length(Q) - Rb);

        MediumSample med = SampleMedium(alt);
        od += med.extinction * ds;
    }

    return exp(-od);
}

// Lambertian planet sphere. Exposure uses SUN_INTENSITY_VAL (not the
// SKY_INTENSITY-amplified sky model) so the lit ground reads as a surface.
// UNUSED since 2026-06-11 (no analytic planet surface) — kept for debugging.
inline float3 EvaluatePlanetBody(float3 O, float3 V, float3 L)
{
    float tG0, tG1;
    if (!RaySphereIntersect(O, V, ATMOS_BOTTOM_RADIUS, tG0, tG1)) return float3(0, 0, 0);
    if (tG0 <= 0.0f) return float3(0, 0, 0);

    float3 P     = O + V * tG0;
    float3 N     = SafeNormalize(P);
    float  NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0f) return float3(0, 0, 0);

    float3 sunTr = TransmittanceToSun(P, L, ATMOS_BOTTOM_RADIUS, ATMOS_TOP_RADIUS);

    return (ATMOS_GROUND_ALBEDO / PI) * NdotL * sunTr * ATMOS_SOLAR_IRRADIANCE * SUN_INTENSITY_VAL;
}

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

inline SunState ComputeSunStateInline()
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

    // Horizon altitude-faded: refraction-lifted (-0.833°) at ground, pure
    // geometric horizon at orbit (well below local horizontal).
    float  obsR2          = max(1e-6f, dot(g_skyObserverPlanet, g_skyObserverPlanet));
    float  obsR           = sqrt(obsR2);
    float  cosGeoHorizon  = -sqrt(max(0.0f, 1.0f - (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS) / obsR2));
    float  cosGroundLift  = sin(SUN_HORIZON_DEG * DEG2RAD);
    float  altFade        = saturate((obsR - ATMOS_BOTTOM_RADIUS) / max(1e-6f, ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float  cosEffHorizon  = lerp(cosGroundLift, cosGeoHorizon, altFade);
    float3 observerUp     = g_skyObserverPlanet / obsR;
    float  sunCosFromUp   = dot(d, observerUp);

    // Smoothstep over ~1° prevents "lights terrain" pop at horizon crossing
    // while still reading as a real sunset, not a dimmer.
    const float kHorizonFadeDeg = 1.0f;
    const float visEdge         = sin(kHorizonFadeDeg * DEG2RAD);
    S.visible = smoothstep(-visEdge, +visEdge, sunCosFromUp - cosEffHorizon);

    float3 Tr = AtmosphericTransmittance(S.dirWS);
    S.tint    = SUN_COLOR_VAL * Tr;

    // PDF stays at 1/omega while sun is partly visible for stable MIS;
    // radiance carries the smooth fade.
    S.pdf      = (S.visible > 0.001f) ? (1.0f / S.omega) : 0.0f;
    S.radiance = S.tint * SUN_INTENSITY_VAL * S.visible / S.omega;

    return S;
}

#include "SkyBake_v8.hlsli"

// Pass_pt (SKYBAKE_CONSUMER): every caller there shares the camera observer,
// so the state comes from the per-frame bake instead of a march per call.
#include "CumulusCache_v8.hlsli"

inline SunState ComputeSunState()
{
#if SKYBAKE_CONSUMER
    return SkyBakeLoadSunState();
#else
    return ComputeSunStateInline();
#endif
}

SunSampleResult SampleSun(float2 u, float3 receiverWorld)
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

    r.radiance *= CumulusSunVisibility(WorldToPlanet(receiverWorld), r.direction);
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
    if (SkyObserverIsUnderground()) return 0.0f;
    SunState S = ComputeSunState();
    if (S.pdf <= 0.0f) return float3(0, 0, 0);

    float3 d = SafeNormalize(rayDir);
    float cosAngle = dot(d, S.dirWS);

    if (cosAngle < S.cosThetaMax) return float3(0, 0, 0);

#if SUN_LIMB_DARKENING
    float mu = saturate((cosAngle - S.cosThetaMax) / max(1e-6f, 1.0f - S.cosThetaMax));
    mu = sqrt(mu);
    float3 ld = LimbDarkening(mu);
    return S.radiance * ld / 0.85f * CumulusSunVisibility(g_skyObserverPlanet,d);
#else
    return S.radiance * CumulusSunVisibility(g_skyObserverPlanet,d);
#endif
}

// Above-atmosphere sun (no tint). Use when the caller applies the
// camera-to-sun transmittance externally (primary raygen — combinedTr
// already integrates it). EvaluateSun would double-attenuate.
float3 EvaluateSunUnattenuated(float3 rayDir)
{
    SunState S = ComputeSunState();
    if (S.pdf <= 0.0f) return float3(0, 0, 0);

    float3 d = SafeNormalize(rayDir);
    float cosAngle = dot(d, S.dirWS);

    if (cosAngle < S.cosThetaMax) return float3(0, 0, 0);

    float3 radiance = SUN_COLOR_VAL * SUN_INTENSITY_VAL * S.visible / S.omega;

#if SUN_LIMB_DARKENING
    float mu = saturate((cosAngle - S.cosThetaMax) / max(1e-6f, 1.0f - S.cosThetaMax));
    mu = sqrt(mu);
    float3 ld = LimbDarkening(mu);
    return radiance * ld / 0.85f;
#else
    return radiance;
#endif
}

// Equirect celestial-frame star map. WorldToCelestial locks rotation to
// sidereal time. Horizon / atmosphere gating happens at EvaluateSky scope.
float3 EvaluateStars(float3 rayDir)
{
    float3 v     = SafeNormalize(rayDir);
    float3 vStar = WorldToCelestial(v);

    float dec = asin(clamp(vStar.y, -1.0f, 1.0f));
    float ra  = atan2(vStar.z, vStar.x);
    float2 uv = float2(ra * (1.0f / TAU) + 0.5f,
                       0.5f - dec * (1.0f / PI));

#if SKY_STAR_TEXTURE_FLIP_U
    uv.x = 1.0f - uv.x;
#endif

    // Footprint-based LOD keeps bright single-texel stars stable under
    // sub-pixel jitter (otherwise DLSS RR smears them frame-to-frame).
    float texW, texH, texMips;
    gSkyStars.GetDimensions(0, texW, texH, texMips);
    float pixelAngular = 2.0f / (projection._m11 * float(gImageSize.y));
    float texelAngular = PI / texH;
    float lod = log2(pixelAngular / texelAngular) + SKY_STAR_LOD_BIAS;
    lod = clamp(lod, 0.0f, texMips - 1.0f);

    float3 c = gSkyStars.SampleLevel(g_sampler, uv, lod).rgb;
    c = max(c, 0.0f);

#if SKY_STAR_TEXTURE_SRGB
    // sRGB → linear (fast polynomial, max error ~0.003).
    c = c * (c * (c * 0.305306011f + 0.682171111f) + 0.012522878f);
#endif

    // Black-level lift clamps the bilinear halo around each star to zero,
    // shrinking the visible footprint to the bright centre.
    c = max(c - SKY_STAR_THRESHOLD, 0.0f);

    // Luminance-only power curve: lum^gamma while preserving chroma.
    const float lum    = max(dot(c, float3(0.2126f, 0.7152f, 0.0722f)), 1e-6f);
    const float factor = pow(lum, SKY_STAR_GAMMA - 1.0f);
    c *= factor;

    return c * SKY_STAR_INTENSITY * SKY_STAR_SCALE;
}

// Atmosphere along v from the observer: in-scatter (solar-irradiance units),
// the view transmittance and whether the ray ends on the planet (0 or 1; the
// LUT path blends the two across the horizon). Marched per call, or read from
// the per-frame sky-view LUT (SKYBAKE_CONSUMER, Pass_pt only: its observer is
// the camera for every miss).
inline void SkyAtmosphere(float3 v, float3 sunDir,
                          out float3 scatter, out float3 viewTr, out float hitPlanet)
{
#if SKYBAKE_CONSUMER
    SkyBakeLoadView(v, scatter, viewTr, hitPlanet);
#else
    bool hit;
    scatter   = IntegrateScattering(v, sunDir, viewTr, hit);
    hitPlanet = hit ? 1.0f : 0.0f;
#endif
}

float3 EvaluateSkyBackground(float3 rayDir)
{
#if ATM_DEBUG_RING == 3
    return float3(0.0f, 0.0f, 0.0f);   //sky disabled — ATM_DEBUG_RING
#endif
    //Underground observer: no sky is geometrically visible from inside
    //the planet body.
    if (SkyObserverIsUnderground()) return float3(0.0f, 0.0f, 0.0f);

    SunState S = ComputeSunState();

    float3 v       = SafeNormalize(rayDir);
    float  elevDeg = S.elevRad * RAD2DEG;
    float3 O       = g_skyObserverPlanet;

    //physical scattering, viewTr is the real atmospheric transmittance along
    //the integrated path (from observer to atmosphere top OR planet surface);
    //marched here, or read from the per-frame sky-view LUT under Pass_pt
    float3 scatter, viewTr;
    float  hitPlanet;
    SkyAtmosphere(v, S.dirWS, scatter, viewTr, hitPlanet);

    float3 daySky = scatter * SKY_INTENSITY;

    // NO analytic planet body: below-horizon rays return aerial in-scatter
    // only, fading to black. The scene's own geometry provides any visible
    // ground; the old Lambertian Rb sphere also leaked gray "ground light"
    // into GI bounce misses (the nadir-ring class of artifacts). hitPlanet
    // still clips the march and gates stars/airglow below — only the shaded
    // body is gone.

    // Night base airglow. MUST NOT scale by SKY_INTENSITY — airglow and
    // integrated starlight don't depend on sun brightness, otherwise
    // cranking sunSunIntensity brightens midnight.
    float observerR     = length(O);
    float atmosResidual = saturate((ATMOS_TOP_RADIUS - observerR)
                                   / max(1e-6f, ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float tw            = Smooth01(saturate((elevDeg + SKY_TWILIGHT_DEG) / SKY_TWILIGHT_DEG));
    float mu            = saturate(dot(v, WORLD_UP));
    float3 nightBase    = SKY_NIGHT_BASE * skyNightBaseIntensity
                                         * lerp(1.6f, 1.0f, pow(mu, 0.7f));
    nightBase          *= atmosResidual * (1.0f - hitPlanet);

    // Stars gated by atmospheric scatter only — NO sun-elevation gate, or
    // orbital daytime hides them (scatter ≈ 0 at altitude even with sun up).
    float3 starShield = exp(-scatter * SKY_STAR_SCATTER_SHIELD);
    float3 stars      = (hitPlanet < 1.0f)
                      ? (EvaluateStars(v) * viewTr * starShield * (1.0f - hitPlanet))
                      : float3(0, 0, 0);

    return lerp(nightBase, daySky, tw) + stars;
}

// Components "behind" the unified march's scatter — planet body, stars,
// airglow. Caller multiplies the result by combined T from the march so
// all three get correct attenuation through both media. unifiedInscatter
// drives the star shield (divided by SKY_INTENSITY to recover the
// unamplified magnitude the shield was tuned for).
float3 EvaluateSkyBackgroundBehind(float3 rayDir, SunState S,
                                   bool hitPlanet, float3 unifiedInscatter)
{
#if ATM_DEBUG_RING == 3
    return float3(0.0f, 0.0f, 0.0f);   //sky disabled — ATM_DEBUG_RING
#endif
    float3 v = SafeNormalize(rayDir);
    float3 O = g_skyObserverPlanet;
    float elevDeg = S.elevRad * RAD2DEG;

    float observerR     = length(O);
    float atmosResidual = saturate((ATMOS_TOP_RADIUS - observerR)
                                   / max(1e-6f, ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float tw            = Smooth01(saturate((elevDeg + SKY_TWILIGHT_DEG) / SKY_TWILIGHT_DEG));
    float mu            = saturate(dot(v, WORLD_UP));
    float3 nightBase    = SKY_NIGHT_BASE * skyNightBaseIntensity
                                         * lerp(1.6f, 1.0f, pow(mu, 0.7f));
    nightBase          *= atmosResidual * (hitPlanet ? 0.0f : 1.0f) * (1.0f - tw);

    // NO analytic planet body (see EvaluateSkyBackground) — hitPlanet only
    // gates stars/airglow here.

    float3 stars = float3(0, 0, 0);
    if (!hitPlanet)
    {
        float3 shieldInput = unifiedInscatter / max(SKY_INTENSITY, 1e-6f);
        float3 starShield  = exp(-shieldInput * SKY_STAR_SCATTER_SHIELD);
        stars = EvaluateStars(v) * starShield;
    }

    return nightBase + stars;
}

float3 EvaluateSky(float3 rayDir)
{
    if (cloudEnabled > .5f && !SkyObserverIsUnderground())
        return CumulusEnvironment(rayDir,0u).rgb;
    return EvaluateSkyBackground(rayDir);
}

//Deferred RC_F_ENV_REPLAY finish (merge side of EnvReplayTailPartial — see
//Hybrid_Replay_v8.hlsli). The shift persisted the partial contribution `cPartial`
//and the re-derived offset direction `s` + its marginal BSDF pdf `misPdf`; here
//we do the atmosphere eval (sky + MIS sun disc) the shift used to inline, keeping
//that LUT/march stack OUT of the register-tight RT shift binary. Same algebra as
//the old inline tail — the ONLY difference is the observer, which the CALLER sets
//once per pixel via SetSkyObserver (per-pixel approximation of the offset bounce
//vertex; sky observer is very low-frequency over a pixel neighbourhood).
float3 EnvTailFinish(float3 cPartial, float3 s, float misPdf)
{
    const float  sunSAPdf   = GetSunPdf(s);
    const float3 sunRad     = (sunSAPdf > 0.0f) ? EvaluateSun(s) : float3(0, 0, 0);
    const float  sunMisBsdf = (sunSAPdf > 0.0f)
        ? misPdf / max(misPdf + sunSAPdf, EPSILON) : 0.0f;
    const float3 sky  = EvaluateSky(s);
    const float3 envL = sky + sunRad * sunMisBsdf;

    float3 c = cPartial * envL;
    return (any(isnan(c)) || any(isinf(c))) ? (float3)0.0f : max(c, 0.0f);
}
