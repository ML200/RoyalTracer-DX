//====================================
//TIME AND ANGLE MACROS
//====================================
#define TAU     (2.0f * PI)
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

#ifndef SUN_FRAMECOUNT
#define SUN_FRAMECOUNT time
#endif

//====================================
//SUN DEFAULTS
//====================================
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

//====================================
//ATMOSPHERE BRUNETON 2017
//====================================
//lengths in km, coefficients in 1/km

#ifndef ATMOS_BOTTOM_RADIUS
#define ATMOS_BOTTOM_RADIUS     6360.0f
#endif

#ifndef ATMOS_TOP_RADIUS
#define ATMOS_TOP_RADIUS        6420.0f
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

#ifndef ATMOS_MULTI_SCATTER_FACTOR
#define ATMOS_MULTI_SCATTER_FACTOR  1.1f
#endif

#ifndef SKY_INTENSITY
#define SKY_INTENSITY           6.0f
#endif

//====================================
//NIGHT SKY
//====================================
#ifndef SKY_TWILIGHT_DEG
#define SKY_TWILIGHT_DEG        18.0f
#endif

//Star skybox runtime knobs are pulled from the CameraParams cbuffer
//(skyStarIntensity, skyStarGamma, skyStarLodBias, skyStarThreshold on the
//SunSettings tail), driven by the editor sliders so values can be tuned
//live without a recompile. Names kept as SKY_STAR_* for compatibility with
//the existing call sites.
#define SKY_STAR_INTENSITY      skyStarIntensity
#define SKY_STAR_GAMMA          skyStarGamma
#define SKY_STAR_THRESHOLD      skyStarThreshold

//Additional scale, kept separate so the host can drive an artistic fade
//(e.g. for cinematic shots) without touching INTENSITY.
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

//per-channel optical-depth scaling used to fade stars where the daytime sky is
//bright. Higher = stars hide earlier as scatter ramps up at dawn/dusk. At
//ground noon, blue Rayleigh scatter ≈ 0.02; shield 1000 gives exp(-20) → 0.
//Red scatter is ~5x smaller, so red components reach exp(-4) ≈ 0.018 — close
//enough to invisible after AE+AgX. From orbit scatter ≈ 0 so stars are
//unaffected, which is the whole point: the gate must be the atmosphere, not
//the sun's elevation.
#ifndef SKY_STAR_SCATTER_SHIELD
#define SKY_STAR_SCATTER_SHIELD 1000.0f
#endif

//====================================
//STAR / MILKY WAY TEXTURE
//====================================
//Stars and Milky Way are sourced from an equirectangular sky texture in the
//celestial (RA/Dec) frame, bound as gSkyStars in Includes_v8.hlsli. Sampled
//with WorldToCelestial(rayDir) to stay locked to sidereal rotation. Host
//side: load NASA SVS 4851 Deep Star Maps as Texture2D, expose the SRV at
//register t40, and bind it through the root signature.

//Set to 1 if the source texture is stored sRGB encoded (LDR 8 bit TIFF/JPEG/
//PNG). Leave at 0 for HDR linear formats (BC6H DDS, .exr, .hdr).
#ifndef SKY_STAR_TEXTURE_SRGB
#define SKY_STAR_TEXTURE_SRGB 0
#endif

//Set to 1 if the texture's right ascension increases right to left (mirror
//image, "as seen from inside the celestial sphere"). Most map projections go
//left to right (RA=0 at u=0.5, increasing toward u=1).
#ifndef SKY_STAR_TEXTURE_FLIP_U
#define SKY_STAR_TEXTURE_FLIP_U 0
#endif

//LOD bias is also runtime tunable via the editor (SunSettings.skyStarLodBias).
#define SKY_STAR_LOD_BIAS       skyStarLodBias

//Lambertian albedo of the planet body, used when a ray hits the ground from
//orbit. Ocean-dominated, lit by the same in-shader sun irradiance.
#ifndef ATMOS_GROUND_ALBEDO
#define ATMOS_GROUND_ALBEDO     float3(0.06f, 0.07f, 0.10f)
#endif

//====================================
//WORLD ORIENTATION
//====================================
#ifndef WORLD_NORTH
#define WORLD_NORTH             normalize(float3(0, 0, 1))
#endif

static const float3 WORLD_UP = float3(0, 1, 0);

//====================================
//OBSERVER POSITION
//====================================
//world Y is altitude above the surface tangent point, world XZ are tangent-plane
//offsets (planet curvature is ignored for horizontal motion, which keeps "fly
//straight up to orbit" predictable). Override WORLD_UNITS_PER_KM if your engine
//does not use 1 m world units.
#ifndef WORLD_UNITS_PER_KM
#define WORLD_UNITS_PER_KM      1000.0f
#endif

static const float SKY_OBSERVER_MIN_RADIUS = ATMOS_BOTTOM_RADIUS + 0.0002f;

//per-invocation observer position in planet space (km, planet centered at origin).
//defaults to the surface tangent point so call sites that skip SetSkyObserver
//keep the legacy ground-level behavior. Raygen sets this once near the top.
static float3 g_skyObserverPlanet = float3(0.0f, SKY_OBSERVER_MIN_RADIUS, 0.0f);

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
    if (r < SKY_OBSERVER_MIN_RADIUS) P *= SKY_OBSERVER_MIN_RADIUS / max(1e-6f, r);
    g_skyObserverPlanet = P;
}

//====================================
//SUN STRUCTS
//====================================
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

//====================================
//UTILITY
//====================================
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

//====================================
//HASHING
//====================================
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

//====================================
//SOLAR POSITION
//====================================
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

//====================================
//CELESTIAL FRAME (STARS)
//====================================
//Local Sidereal Time in radians. Built from the same GetSolarTimeHours() the
//sun uses, so star motion is locked in lockstep with sun motion (including
//SUN_NIGHT_SPEEDUP warping). The SKY_SIDEREAL_RATIO multiplier makes stars
//rotate ~1.00273x faster than the sun, which is the real Earth value:
//over a solar day they drift ~4 min ahead, walking through the seasons over
//a year. Longitude offsets which meridian sees which RA first.
inline float GetLSTRad()
{
    return GetSolarTimeHours() * SKY_SIDEREAL_RATIO * (TAU / 24.0f)
         + SUN_LONGITUDE_DEG * DEG2RAD;
}

//Map a world space direction into the celestial equatorial frame with RA
//fixed (stars are stationary in this frame, the diurnal rotation lives in
//the LST term). This is the algebraic inverse of the same projection
//GetSunDirAndElev uses, so the sun's position and the star field share a
//single coordinate transform and a single time base.
//
//Output convention: vStar.y = celestial north pole component, vStar.x and
//vStar.z together span the celestial equator. EvaluateStars consumes this
//directly for the equirectangular skybox lookup: Dec = asin(vStar.y),
//RA = atan2(vStar.z, vStar.x).
float3 WorldToCelestial(float3 vWorld)
{
    float latRad = SUN_LATITUDE_DEG * DEG2RAD;
    float cosL = cos(latRad), sinL = sin(latRad);

    //world to ENU components
    float3 Nw = SafeNormalize(float3(WORLD_NORTH.x, 0.0f, WORLD_NORTH.z));
    float3 Ew = SafeNormalize(cross(WORLD_UP, Nw));
    float east  = dot(vWorld, Ew);
    float north = dot(vWorld, Nw);
    float up    = dot(vWorld, WORLD_UP);

    //ENU to instantaneous equatorial. Matches the sun convention where
    //v_eq = (cos δ sin H, cos δ cos H, sin δ): X_i at H=90°, Y_i at meridian
    //(H=0), Z_i along celestial north pole. Derived by inverting the
    //rotation by colatitude around the east axis that the sun formula uses.
    float X_i = east;
    float Y_i = -sinL * north + cosL * up;
    float Z_i =  cosL * north + sinL * up;

    //instantaneous to RA fixed. Substituting H = LST - α into the sun's
    //v_eq form and solving for the (α, δ) coefficients gives a rotation by
    //LST around the celestial pole axis.
    float lst = GetLSTRad();
    float cl = cos(lst), sl = sin(lst);
    float X_f =  sl * X_i + cl * Y_i;
    float Y_f = -cl * X_i + sl * Y_i;
    float Z_f =  Z_i;

    //pack with celestial pole on +Y so the equirectangular sampling in
    //EvaluateStars treats vStar.y as the latitude axis directly.
    return float3(X_f, Z_f, Y_f);
}

//====================================
//ATMOSPHERE MEDIUM SAMPLING
//====================================
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

//====================================
//PHASE FUNCTIONS
//====================================
inline float PhaseRayleigh(float cosTheta)
{
    return (3.0f / (16.0f * PI)) * (1.0f + cosTheta * cosTheta);
}

//Cornette-Shanks
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

//====================================
//RAY-SPHERE AND TRANSMITTANCE
//====================================
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

//====================================
//SCATTERING INTEGRATION
//====================================
float3 IntegrateScattering(float3 viewDir, float3 sunDir,
                           out float3 transmittanceOut, out bool hitPlanetOut)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;

    float3 O = g_skyObserverPlanet;
    float3 V = SafeNormalize(viewDir);
    float3 L = SafeNormalize(sunDir);

    hitPlanetOut = false;

    //ray vs atmosphere top, observer can be inside (tV0<0) or above (tV0>0)
    float tV0, tV1;
    if (!RaySphereIntersect(O, V, Rt, tV0, tV1) || tV1 <= 0.0f)
    {
        //ray never enters atmosphere (outside, pointing outward)
        transmittanceOut = float3(1, 1, 1);
        return float3(0, 0, 0);
    }

    float tMin = max(0.0f, tV0);
    float tMax = tV1;

    //planet body terminates the ray, throughput accumulates the real
    //atmospheric transmittance from observer to that surface point and is
    //reused by EvaluatePlanetBody for the outgoing leg
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

    for (int i = 0; i < ATMOS_VIEW_STEPS; i++)
    {
        //sqrt spacing, samples cluster near start of ray
        float u0 = (float)i / (float)ATMOS_VIEW_STEPS;
        float u1 = (float)(i + 1) / (float)ATMOS_VIEW_STEPS;
        float s0 = u0 * u0;
        float s1 = u1 * u1;
        float tMid = tMin + (s0 + s1) * 0.5f * totalDist;
        float ds   = (s1 - s0) * totalDist;

        float3 P = O + V * tMid;
        float alt = max(0.0f, length(P) - Rb);

        MediumSample med = SampleMedium(alt);

        float3 segTr = exp(-med.extinction * ds);

        float3 sunTr = TransmittanceToSun(P, L, Rb, Rt);

        //earth shadow
        float3 Pnorm = SafeNormalize(P);
        float sunCosZ = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f, 1.0f - (Rb * Rb) / dot(P, P)));
        float earthShadow = (sunCosZ > cosHorizon) ? 1.0f : 0.0f;

        float3 scatterPhase = med.scatterR * phR + med.scatterM * phM;

        //analytic in-scatter over segment
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

    //throughput is the atmospheric transmittance from observer along the ray,
    //terminating either at the atmosphere top (no planet hit) or at the planet
    //surface. Planet occlusion of stars/space is reported via hitPlanetOut.
    transmittanceOut = throughput;
    return totalInScatter;
}

//====================================
//AERIAL PERSPECTIVE (SCENE RAYS)
//====================================
//Bounded variant of IntegrateScattering used by the shading pass to apply
//atmosphere to scene geometry. Returns the in-scatter accumulated between the
//observer and a hit point at distance hitDistKm (in km), with the matching
//transmittance written to the out parameter. Caller composites as:
//    finalRadiance = sceneRadiance * transmittance + inScatter * SKY_INTENSITY
//
//Cheaper than IntegrateScattering: scene segments are typically short, so a
//4-step march with 4-step transmittance-to-sun is plenty even for tens of km.
//Distant outdoor objects (mountains 50+ km away) still get a meaningful tint.
#ifndef ATMOS_AERIAL_VIEW_STEPS
#define ATMOS_AERIAL_VIEW_STEPS  4
#endif
#ifndef ATMOS_AERIAL_LIGHT_STEPS
#define ATMOS_AERIAL_LIGHT_STEPS 4
#endif

inline float3 TransmittanceToSunCheap(float3 P, float3 L, float Rb, float Rt)
{
    float t0, t1;
    if (!RaySphereIntersect(P, L, Rt, t0, t1)) return float3(1, 1, 1);

    float tMax = t1;
    if (tMax <= 0.0f) return float3(1, 1, 1);
    float tMin = max(0.0f, t0);

    float ds = (tMax - tMin) / (float)ATMOS_AERIAL_LIGHT_STEPS;
    float3 od = float3(0, 0, 0);

    [unroll]
    for (int i = 0; i < ATMOS_AERIAL_LIGHT_STEPS; i++)
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

//Returns in-scatter in units of ATMOS_SOLAR_IRRADIANCE; caller scales by
//SKY_INTENSITY (sunSunIntensity * sunSkyIntensity) to match scene exposure.
float3 ComputeAerialPerspective(float3 viewDir, float3 sunDir, float hitDistKm,
                                out float3 transmittanceOut)
{
    float Rb = ATMOS_BOTTOM_RADIUS;
    float Rt = ATMOS_TOP_RADIUS;
    float3 O = g_skyObserverPlanet;
    float3 V = SafeNormalize(viewDir);
    float3 L = SafeNormalize(sunDir);

    //find the segment of the ray that lies inside the atmosphere shell, clipped
    //to the hit distance. Observer can be inside (tV0<0) or above (tV0>0).
    float tV0, tV1;
    if (!RaySphereIntersect(O, V, Rt, tV0, tV1) || tV1 <= 0.0f)
    {
        //segment never enters the atmosphere
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

    //uniform spacing: scene segments are typically short and roughly homogeneous,
    //so the sqrt-spaced clustering used for full-sky integration is unnecessary
    [unroll]
    for (int i = 0; i < ATMOS_AERIAL_VIEW_STEPS; i++)
    {
        float u0   = (float)i / (float)ATMOS_AERIAL_VIEW_STEPS;
        float u1   = (float)(i + 1) / (float)ATMOS_AERIAL_VIEW_STEPS;
        float tMid = tMin + (u0 + u1) * 0.5f * totalDist;
        float ds   = (u1 - u0) * totalDist;

        float3 P = O + V * tMid;
        float alt = max(0.0f, length(P) - Rb);
        MediumSample med = SampleMedium(alt);

        float3 segTr = exp(-med.extinction * ds);

        //earth shadow at this sample
        float3 Pnorm = SafeNormalize(P);
        float sunCosZ = dot(Pnorm, L);
        float cosHorizon = -sqrt(max(0.0f, 1.0f - (Rb * Rb) / dot(P, P)));
        float earthShadow = (sunCosZ > cosHorizon) ? 1.0f : 0.0f;

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

//====================================
//ATMOSPHERIC TRANSMITTANCE
//====================================
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

    //planet between observer and atmosphere exit blocks the path entirely
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

//====================================
//PLANET BODY
//====================================
//simple Lambertian sphere lit by the in-shader sun. Exposure matches the rest
//of the path-traced scene (SUN_INTENSITY_VAL irradiance) rather than the
//SKY_INTENSITY-amplified sky model, so the lit ground reads as a surface, not
//a glowing emitter. Caller multiplies by the observer→surface atmospheric
//transmittance to attenuate the outgoing leg through the air column.
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

//====================================
//LIMB DARKENING
//====================================
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

//====================================
//SUN STATE
//====================================
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

    //horizon depends on observer altitude: at ground use the refraction-lifted
    //value (-0.833 deg), at the top of the atmosphere drop to the pure geometric
    //horizon, which is well below local horizontal for an orbital observer
    float  obsR2          = max(1e-6f, dot(g_skyObserverPlanet, g_skyObserverPlanet));
    float  obsR           = sqrt(obsR2);
    float  cosGeoHorizon  = -sqrt(max(0.0f, 1.0f - (ATMOS_BOTTOM_RADIUS * ATMOS_BOTTOM_RADIUS) / obsR2));
    float  cosGroundLift  = sin(SUN_HORIZON_DEG * DEG2RAD);
    float  altFade        = saturate((obsR - ATMOS_BOTTOM_RADIUS) / max(1e-6f, ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float  cosEffHorizon  = lerp(cosGroundLift, cosGeoHorizon, altFade);
    float3 observerUp     = g_skyObserverPlanet / obsR;
    float  sunCosFromUp   = dot(d, observerUp);

    //Smooth horizon visibility: a hard step here caused a global "lights
    //out" pop when the sun crossed the horizon (direct sun radiance went
    //from full to zero in a single sim tick). Smoothstep over ~1 deg of
    //sun altitude — wide enough to absorb the real sun disc (0.5 deg
    //diameter) plus atmospheric refraction blur, narrow enough to still
    //read as "sunset / sunrise" instead of an artificial dimmer.
    //sin() converts the angular fade width to the cos space the inputs
    //live in (small angle approx near horizon: d(cos)/dα ≈ -sin(α) ≈ -1).
    const float kHorizonFadeDeg = 1.0f;
    const float visEdge         = sin(kHorizonFadeDeg * DEG2RAD);
    S.visible = smoothstep(-visEdge, +visEdge, sunCosFromUp - cosEffHorizon);

    float3 Tr = AtmosphericTransmittance(S.dirWS);
    S.tint    = SUN_COLOR_VAL * Tr;

    //PDF stays at 1/omega while the sun is even partly visible so MIS
    //weights stay sensible; radiance carries the smooth fade factor so
    //direct sun contribution dies gracefully instead of popping.
    S.pdf      = (S.visible > 0.001f) ? (1.0f / S.omega) : 0.0f;
    S.radiance = S.tint * SUN_INTENSITY_VAL * S.visible / S.omega;

    return S;
}

//====================================
//PUBLIC SUN API
//====================================
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

//====================================
//STARS / MILKY WAY (texture sourced)
//====================================
//Sample an equirectangular sky texture in the celestial RA fixed frame so
//stars and Milky Way rotate in lockstep with the sun (via WorldToCelestial).
//Gating by horizon/atmosphere happens at EvaluateSky scope via viewTr and
//the scatter shield, identical to the previous procedural path.
float3 EvaluateStars(float3 rayDir)
{
    float3 v     = SafeNormalize(rayDir);
    float3 vStar = WorldToCelestial(v);

    //Equirectangular projection: u = RA / 2π, v = (π/2 - Dec) / π.
    //vStar.y is on the celestial pole axis (set by WorldToCelestial), so
    //Dec = asin(vStar.y) and RA = atan2(vStar.z, vStar.x).
    float dec = asin(clamp(vStar.y, -1.0f, 1.0f));
    float ra  = atan2(vStar.z, vStar.x);
    float2 uv = float2(ra * (1.0f / TAU) + 0.5f,
                       0.5f - dec * (1.0f / PI));

#if SKY_STAR_TEXTURE_FLIP_U
    uv.x = 1.0f - uv.x;
#endif

    //Footprint based LOD selection: pick a mip where one texel covers about
    //one screen pixel, plus SKY_STAR_LOD_BIAS extra to make single texel
    //bright stars stable under sub pixel jitter. This is the fix for the
    //flicker that DLSS RR smears across frames; without it, jitter shifts
    //which texel each pixel hits and the denoiser sees the stars as
    //incoherent moving features.
    //  pixel angular size = 2 / (proj._m11 * H)        (proj._m11 = cot(fovV/2))
    //  texel angular size = π / texH                   (equirect latitude)
    float texW, texH, texMips;
    gSkyStars.GetDimensions(0, texW, texH, texMips);
    //gImageSize comes from the Push cbuffer and is available in both raygen
    //and compute; gImageHeight is a compute-only alias for gImageSize.y.
    float pixelAngular = 2.0f / (projection._m11 * float(gImageSize.y));
    float texelAngular = PI / texH;
    float lod = log2(pixelAngular / texelAngular) + SKY_STAR_LOD_BIAS;
    lod = clamp(lod, 0.0f, texMips - 1.0f);

    float3 c = gSkyStars.SampleLevel(g_sampler, uv, lod).rgb;
    c = max(c, 0.0f);

#if SKY_STAR_TEXTURE_SRGB
    //sRGB -> linear (fast polynomial approximation, max error ~0.003)
    c = c * (c * (c * 0.305306011f + 0.682171111f) + 0.012522878f);
#endif

    //Black level lift: subtract a fixed pedestal so the soft bilinear halo
    //around each star (the dim averaged texels) clamps to zero, shrinking
    //the visible star footprint to the bright centre. Applied before gamma
    //so the curve operates on a clean signal.
    c = max(c - SKY_STAR_THRESHOLD, 0.0f);

    //Luminance based power curve: turn the bilinear averaged "soft bloom"
    //into sparse sparkles. Applied on luminance only (not per channel) so
    //star colours don't shift toward whichever channel happens to be
    //brightest. factor = lum^(gamma - 1) implements lum -> lum^gamma while
    //preserving chroma.
    const float lum    = max(dot(c, float3(0.2126f, 0.7152f, 0.0722f)), 1e-6f);
    const float factor = pow(lum, SKY_STAR_GAMMA - 1.0f);
    c *= factor;

    return c * SKY_STAR_INTENSITY * SKY_STAR_SCALE;
}

//====================================
//VOLUMETRIC CLOUDS
//====================================
//Pulled in after the atmosphere helpers (SampleMedium, TransmittanceToSun,
//RaySphereIntersect, Hash12) and before EvaluateSky so the cloud
//integrator can reuse them when shading sky pixels.
#include "Clouds_v8.hlsli"

//====================================
//PUBLIC SKY API
//====================================
//Sky is split into THREE evaluators so different consumers can pick the
//right cost/quality point:
//
//  EvaluateSkyBackground(v)     — atmosphere + stars + planet body only,
//                                 NO clouds. Used by raygen's primary
//                                 miss; clouds for primary pixels are
//                                 supplied by a dedicated compute pass
//                                 (Pass_clouds_primary_v8) and composited
//                                 in the shading pass.
//
//  EvaluateSky(v, cloudTrOut)   — background + CHEAP clouds. Used by
//                                 bounce-miss / inline-RT-miss paths
//                                 where the path's contribution is
//                                 attenuated by NRC/ReSTIR weights and
//                                 we don't want to spend the full
//                                 Skybolt step-and-retract march on
//                                 every indirect ray. cloudTrOut is the
//                                 cheap cloud transmittance so the
//                                 caller can attenuate sun-disc / MIS-
//                                 sun radiance consistently.
//
//  EvaluateSky(v)               — convenience overload, discards cloudTr.
//
//The full-quality EvaluateClouds is still exported by Clouds_v8.hlsli
//but only called from Pass_clouds_primary_v8.

float3 EvaluateSkyBackground(float3 rayDir)
{
    SunState S = ComputeSunState();

    float3 v       = SafeNormalize(rayDir);
    float  elevDeg = S.elevRad * RAD2DEG;
    float3 O       = g_skyObserverPlanet;

    //physical scattering, viewTr is the real atmospheric transmittance along
    //the integrated path (from observer to atmosphere top OR planet surface),
    //hitPlanet says whether the ray terminated on the ground
    bool   hitPlanet;
    float3 viewTr;
    float3 scatter = IntegrateScattering(v, S.dirWS, viewTr, hitPlanet);

    float3 daySky = scatter * SKY_INTENSITY;

    //No observer-position cloud-shadow proxy on daySky. The previous
    //attempt dimmed the entire atmospheric scatter based on whether
    //the OBSERVER was in cloud shadow — which made the whole sky go
    //dark when the camera entered even a tiny cloud shadow. The
    //correct behaviour is for the cloud's line-of-sight transmittance
    //(cloudTr from EvaluateClouds / EvaluateCloudsCheap) to attenuate
    //the sky per-pixel, which the Pass_clouds_primary_v8 composite
    //and the bounce-ray EvaluateSky already do. Atmospheric scatter
    //is allowed to remain at its physical (sun-lit) value here.

    //planet body, attenuated by the outgoing atmospheric leg back to the
    //observer. Sun-side cloud shadow uses the hit point (not the observer)
    //because the planet ground may be far away and live under different
    //cloud cover than the camera — same lookup pattern as mesh surface NEE
    //in raygen.
    float3 planetBody = float3(0, 0, 0);
    if (hitPlanet)
    {
        planetBody = EvaluatePlanetBody(O, v, S.dirWS) * viewTr;
        if (cloud_cloudShadowOnSurfaces > 0.5f)
        {
            float tG0, tG1;
            if (RaySphereIntersect(O, v, ATMOS_BOTTOM_RADIUS, tG0, tG1) && tG0 > 0.0f)
            {
                float3 Pplanet    = O + v * tG0;
                float3 PworldHit  = float3(Pplanet.x,
                                           Pplanet.y - ATMOS_BOTTOM_RADIUS,
                                           Pplanet.z) * WORLD_UNITS_PER_KM;
                planetBody *= CloudSunVisibility(PworldHit, S.dirWS);
            }
        }
    }

    //night base, residual upper-atmosphere airglow. Fades out above the
    //atmosphere top and is suppressed entirely when the planet body covers
    //the ray (no "sky glow leaking through the ground"). Crucially does
    //NOT scale by SKY_INTENSITY (= sunSunIntensity * sunSkyIntensity):
    //airglow and integrated starlight are physically independent of the
    //sun's brightness, so cranking sunSunIntensity must not brighten the
    //night sky. (Pre-fix bug: night sky bled with sun intensity, making
    //"midnight" never fully dark even with the sun far below the horizon.)
    float observerR     = length(O);
    float atmosResidual = saturate((ATMOS_TOP_RADIUS - observerR)
                                   / max(1e-6f, ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float tw            = Smooth01(saturate((elevDeg + SKY_TWILIGHT_DEG) / SKY_TWILIGHT_DEG));
    float mu            = saturate(dot(v, WORLD_UP));
    float3 nightBase    = SKY_NIGHT_BASE * skyNightBaseIntensity
                                         * lerp(1.6f, 1.0f, pow(mu, 0.7f));
    nightBase          *= atmosResidual * (hitPlanet ? 0.0f : 1.0f);

    //stars fade where sky scatter overwhelms them and where atmospheric
    //extinction reduces throughput. NO sun-elevation gating — at orbital
    //altitudes the atmosphere is thin enough that scatter ≈ 0 even with the
    //sun above the horizon, so stars must remain visible from high up during
    //the day. Hiding stars at ground noon comes entirely from the scatter
    //shield (blue scatter ~0.02 + shield 1000 → exp(-20)) plus viewTr.
    float3 starShield = exp(-scatter * SKY_STAR_SCATTER_SHIELD);
    float3 stars      = hitPlanet ? float3(0, 0, 0)
                                  : (EvaluateStars(v) * viewTr * starShield);

    return lerp(nightBase, daySky, tw) + stars + planetBody;
}

//Background components "behind" the atmospheric scatter — planet body,
//stars, and airglow nightBase. Used by the unified atmosphere+clouds
//march in Pass_clouds_primary_v8 to produce the part of the sky pixel
//that the unified march does NOT integrate (the unified march handles
//atmospheric scatter and cloud in-scatter directly).
//
//The caller multiplies the returned value by the combined atmosphere+
//cloud transmittance from the unified march, so all three terms get the
//correct attenuation through both media.
//
//unifiedInscatter is passed in so the star shield can fade stars where
//atmospheric scatter is bright (matches the old shield behaviour using
//IntegrateScattering's raw scatter — we divide by SKY_INTENSITY to
//recover the unamplified scatter magnitude the shield was tuned for).
float3 EvaluateSkyBackgroundBehind(float3 rayDir, SunState S,
                                   bool hitPlanet, float3 unifiedInscatter)
{
    float3 v = SafeNormalize(rayDir);
    float3 O = g_skyObserverPlanet;
    float elevDeg = S.elevRad * RAD2DEG;

    //Night base airglow — fades out at dawn via (1 - tw), zeroed when
    //ray terminates on the planet (no airglow behind the ground).
    float observerR     = length(O);
    float atmosResidual = saturate((ATMOS_TOP_RADIUS - observerR)
                                   / max(1e-6f, ATMOS_TOP_RADIUS - ATMOS_BOTTOM_RADIUS));
    float tw            = Smooth01(saturate((elevDeg + SKY_TWILIGHT_DEG) / SKY_TWILIGHT_DEG));
    float mu            = saturate(dot(v, WORLD_UP));
    float3 nightBase    = SKY_NIGHT_BASE * skyNightBaseIntensity
                                         * lerp(1.6f, 1.0f, pow(mu, 0.7f));
    nightBase          *= atmosResidual * (hitPlanet ? 0.0f : 1.0f) * (1.0f - tw);

    //Planet body with per-pixel cloud shadow at the ground hit point.
    //The outgoing atmospheric transmittance from hit to observer is the
    //combined transmittance applied by the caller — not multiplied here.
    float3 planetBody = float3(0, 0, 0);
    if (hitPlanet)
    {
        planetBody = EvaluatePlanetBody(O, v, S.dirWS);
        if (cloud_cloudShadowOnSurfaces > 0.5f)
        {
            float tG0, tG1;
            if (RaySphereIntersect(O, v, ATMOS_BOTTOM_RADIUS, tG0, tG1) && tG0 > 0.0f)
            {
                float3 Pplanet   = O + v * tG0;
                float3 PworldHit = float3(Pplanet.x,
                                          Pplanet.y - ATMOS_BOTTOM_RADIUS,
                                          Pplanet.z) * WORLD_UNITS_PER_KM;
                planetBody *= CloudSunVisibility(PworldHit, S.dirWS);
            }
        }
    }

    //Stars shielded by atmospheric (+cloud) brightness so they hide
    //during daytime. Shield input is the unified inscatter divided by
    //SKY_INTENSITY so the SKY_STAR_SCATTER_SHIELD constant retains its
    //original tuning (operates on un-amplified scatter magnitude).
    //Cloud brightness contributes to the shield too — a sunlit cloud
    //behind a star will hide it.
    float3 stars = float3(0, 0, 0);
    if (!hitPlanet)
    {
        float3 shieldInput = unifiedInscatter / max(SKY_INTENSITY, 1e-6f);
        float3 starShield  = exp(-shieldInput * SKY_STAR_SCATTER_SHIELD);
        stars = EvaluateStars(v) * starShield;
    }

    return nightBase + planetBody + stars;
}

//Background + CHEAP volumetric clouds. cloudTrOut returns the [0,1]
//cheap cloud transmittance along v so callers can attenuate sun-disc /
//MIS-sun radiance that passes through clouds. The cheap variant uses
//a single straight march (no shadow rays, single Worley octave, no
//multi-scatter octaves) — fine for bounce paths whose contribution is
//attenuated by NRC/ReSTIR weights and ultimately denoised by DLSS RR.
float3 EvaluateSky(float3 rayDir, out float3 cloudTrOut)
{
    cloudTrOut = float3(1.0f, 1.0f, 1.0f);

    float3 v          = SafeNormalize(rayDir);
    //EvaluateSkyBackground already applies sun-side cloud shadow to its
    //daySky and planetBody terms (stars / nightBase intentionally skipped).
    //No additional background-wide multiply here, which would have wrongly
    //dimmed stars and airglow by sun visibility.
    float3 background = EvaluateSkyBackground(v);

    SunState S = ComputeSunState();

    float3 cloudL = EvaluateCloudsCheap(v, S.dirWS,
                                        ATMOS_SOLAR_IRRADIANCE * SKY_INTENSITY,
                                        cloudTrOut);
    return background * cloudTrOut + cloudL;
}

//Single-argument overload kept so legacy call sites that don't care
//about cloud transmittance compile unchanged. The discarded out param
//costs only one register.
float3 EvaluateSky(float3 rayDir)
{
    float3 cloudTrIgnored;
    return EvaluateSky(rayDir, cloudTrIgnored);
}
