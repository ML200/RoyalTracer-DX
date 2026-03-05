#ifndef SUN_LIGHT_HLSLI
#define SUN_LIGHT_HLSLI

// ============================================================================
// Sun + Sky (day/night cycle) — drop-in replacement
// Keeps public API signatures identical:
//
//   SunSampleResult SampleSun(float2 u)
//   float          GetSunPdf(float3 rayDir)
//   float3         EvaluateSun(float3 rayDir)
//
// Adds (new):
//   float3 EvaluateSky(float3 rayDir)
//
// Requirements satisfied:
// - Sun motion from real latitude/longitude (constants below)
// - Simulation speed from global frame counter `time`
// - Horizon cutoff (sun shows/illuminates = 0 below horizon)
// - Realistic-ish dusk/dawn color via simple atmospheric transmittance
// - Night speed-up (deterministic time warp, no state)
// ============================================================================

#ifndef PI
#define PI 3.14159265359
#endif

#define TAU     (2.0f * PI)
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

// ---------------------------------------------------------------------------
// External time input
// You said: "time" is a global frame count value.
// If your engine uses a different symbol, define SUN_FRAMECOUNT before include.
// ---------------------------------------------------------------------------
#ifndef SUN_FRAMECOUNT
#define SUN_FRAMECOUNT time
#endif

// ---------------------------------------------------------------------------
// Location / date (constants)
// ---------------------------------------------------------------------------
// Geographic coordinates (degrees)
#ifndef SUN_LATITUDE_DEG
#define SUN_LATITUDE_DEG   12.5200f   // e.g. Berlin latitude
#endif

#ifndef SUN_LONGITUDE_DEG
#define SUN_LONGITUDE_DEG  13.4050f   // east positive
#endif

// Day of year [1..365] (constant unless you want seasons)
#ifndef SUN_DAY_OF_YEAR
#define SUN_DAY_OF_YEAR    172.0f     // ~June 21
#endif

// ---------------------------------------------------------------------------
// Simulation control (constants)
// ---------------------------------------------------------------------------
#ifndef SUN_FPS
#define SUN_FPS            110.0f      // your renderer FPS
#endif

#ifndef SUN_SIM_SPEED
#define SUN_SIM_SPEED      400.0f     // sim-seconds per real-second (240 => 4 min sim / 1 sec real)
#endif

#ifndef SUN_START_UTC_HOURS
#define SUN_START_UTC_HOURS 6.0f     // solar clock at frame 0 (UTC-like)
#endif

#ifndef SUN_NIGHT_SPEEDUP
#define SUN_NIGHT_SPEEDUP  8.0f       // nights run faster by this factor (>=1)
#endif

// ---------------------------------------------------------------------------
// Sun appearance (artist knobs)
// ---------------------------------------------------------------------------
#ifndef SUN_ANGULAR_DEG
#define SUN_ANGULAR_DEG    0.53f
#endif

#ifndef SUN_INTENSITY_VAL
#define SUN_INTENSITY_VAL  3.0f       // "disk-integrated" intensity scale
#endif

#ifndef SUN_COLOR_VAL
#define SUN_COLOR_VAL      float3(1.0f, 0.95f, 0.9f)
#endif

#ifndef SUN_DIST_INF
#define SUN_DIST_INF       1e7f
#endif

// Horizon altitude in degrees:
// - 0.0: strict geometric horizon
// - -0.833: common "sunrise/sunset" threshold including refraction+sun radius
#ifndef SUN_HORIZON_DEG
#define SUN_HORIZON_DEG    -0.833f
#endif

// Atmosphere knob for dusk/dawn warmth: ~2 (clear) ... 6 (hazy)
#ifndef SUN_TURBIDITY
#define SUN_TURBIDITY      2.5f
#endif

// ---------------------------------------------------------------------------
// World orientation (Y-up). Provide a "north" vector in XZ plane for azimuth.
// If your world has Z-forward as north, leave default.
// If your north is +X, set WORLD_NORTH to float3(1,0,0), etc.
// ---------------------------------------------------------------------------
#ifndef WORLD_NORTH
#define WORLD_NORTH        normalize(float3(0, 0, 1))
#endif

static const float3 WORLD_UP = float3(0, 1, 0);

// ============================================================================
// Structs (public API)
// ============================================================================
struct SunSampleResult
{
    float3 direction;   // sampled direction to sun
    float3 radiance;    // Le per steradian (uniform over disk)
    float  pdf;         // 1/solidAngle for uniform-cone sampling
    float  dist;        // "infinite" distance
};

// ============================================================================
// Internal solar state
// ============================================================================
struct SunState
{
    float3 dirWS;       // world-space direction to sun
    float  elevRad;     // solar elevation angle (radians)
    float  cosThetaMax; // cos(halfAngle) for disk membership
    float  omega;       // solid angle of sun disk cone
    float  pdf;         // 1/omega (0 if below horizon)
    float3 radiance;    // Le per sr (0 if below horizon)
    float3 tint;        // SUN_COLOR_VAL * transmittance
    float  visible;     // 1 if above horizon, else 0
};

// ============================================================================
// Helpers
// ============================================================================
inline float3 SafeNormalize(float3 v)
{
    return (dot(v, v) > 0.0f) ? normalize(v) : float3(0, 1, 0);
}

// --- FIXED 2x BUG HERE ---
// Correct cone solid angle for half-angle thetaMax:
// Ω = 2π(1 - cos θmax)  (equivalently 4π sin^2(θmax/2))
float GetSunSolidAngle(float thetaMaxRad)
{
    return 2.0f * PI * (1.0f - cos(thetaMaxRad));
}

// Ortho basis around direction N (for cone sampling)
void GetOrthoBasis(float3 N, out float3 T, out float3 B)
{
    // Use WORLD_UP as preferred "up" to keep basis stable.
    float3 Up = abs(dot(N, WORLD_UP)) < 0.999f ? WORLD_UP : float3(1, 0, 0);
    T = SafeNormalize(cross(Up, N));
    B = cross(N, T);
}

// Kasten–Young air mass approximation (good near horizon)
inline float AirMass(float elevRad)
{
    float elevDeg = elevRad * RAD2DEG;
    float e = max(elevDeg, -5.0f); // avoid singularities far below horizon
    float s = sin(e * DEG2RAD);
    return 1.0f / max(1e-3f, (s + 0.15f * pow(max(e + 3.885f, 0.01f), -1.253f)));
}

// Simple RGB transmittance: Rayleigh + Mie extinction (visual, cheap)
inline float3 SunTransmittance(float elevRad, float turbidity)
{
    float m = AirMass(elevRad);

    // Wavelengths in micrometers for RGB
    float3 lambda = float3(0.650f, 0.570f, 0.475f);

    float3 tauR = 0.008735f * pow(lambda, -4.08f) * m;
    float3 tauM = 0.000944f * pow(lambda, -1.30f) * m * turbidity;

    return exp(-(tauR + tauM));
}

// NOAA-ish declination approximation (radians)
inline float SolarDeclinationRad(float dayOfYear, float timeHours)
{
    float gamma = TAU / 365.0f * (dayOfYear - 1.0f + (timeHours - 12.0f) / 24.0f);

    return 0.006918f
         - 0.399912f * cos(gamma) + 0.070257f * sin(gamma)
         - 0.006758f * cos(2.0f * gamma) + 0.000907f * sin(2.0f * gamma)
         - 0.002697f * cos(3.0f * gamma) + 0.001480f * sin(3.0f * gamma);
}

// Sunrise/sunset at altitude h0 (radians) in *solar time hours*
inline void SunriseSunsetHours(float latRad, float declRad, float h0Rad,
                               out float sunriseH, out float sunsetH, out float dayLenH)
{
    float sinLat = sin(latRad), cosLat = cos(latRad);
    float sinD   = sin(declRad), cosD  = cos(declRad);

    float denom = max(1e-6f, cosLat * cosD);
    float cosH0 = (sin(h0Rad) - sinLat * sinD) / denom;

    if (cosH0 >= 1.0f) { dayLenH = 0.0f;  sunriseH = 12.0f; sunsetH = 12.0f; return; } // polar night
    if (cosH0 <= -1.0f){ dayLenH = 24.0f; sunriseH = 0.0f;  sunsetH = 24.0f; return; } // polar day

    float H0 = acos(clamp(cosH0, -1.0f, 1.0f));
    dayLenH  = 2.0f * H0 * RAD2DEG / 15.0f; // 15° per hour
    sunriseH = 12.0f - 0.5f * dayLenH;
    sunsetH  = 12.0f + 0.5f * dayLenH;
}

// Deterministic night speed-up (no state): warp u∈[0,1) to solar time [0,24)
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

// ENU -> world (Y-up), using WORLD_NORTH (projected to XZ)
inline float3 ENU_ToWorld(float east, float north, float up)
{
    float3 N = SafeNormalize(float3(WORLD_NORTH.x, 0.0f, WORLD_NORTH.z));
    float3 E = SafeNormalize(cross(WORLD_UP, N));
    return SafeNormalize(east * E + north * N + up * WORLD_UP);
}

// Compute solar time hours from frame count, sim speed, longitude, with night warp
inline float GetSolarTimeHours()
{
    float tReal = (float)SUN_FRAMECOUNT / SUN_FPS;    // seconds
    float tSim  = tReal * SUN_SIM_SPEED;              // simulated seconds

    // Base day phase u in [0,1):
    // - start offset in hours
    // - longitude phase shift (360° -> 24h)
    float u = frac((SUN_START_UTC_HOURS / 24.0f) + (tSim / 86400.0f) + (SUN_LONGITUDE_DEG / 360.0f));

    float latRad  = SUN_LATITUDE_DEG * DEG2RAD;
    float h0Rad   = SUN_HORIZON_DEG * DEG2RAD;

    // Use declination at noon to stabilize sunrise/sunset for the warp
    float declNoon = SolarDeclinationRad(SUN_DAY_OF_YEAR, 12.0f);

    float sunriseH, sunsetH, dayLenH;
    SunriseSunsetHours(latRad, declNoon, h0Rad, sunriseH, sunsetH, dayLenH);

    return WarpToSolarHours(u, sunriseH, sunsetH, dayLenH, SUN_NIGHT_SPEEDUP);
}

// Compute sun direction + elevation
inline void GetSunDirAndElev(out float3 dirWS, out float elevRad)
{
    float latRad  = SUN_LATITUDE_DEG * DEG2RAD;
    float solarH  = GetSolarTimeHours();
    float declRad = SolarDeclinationRad(SUN_DAY_OF_YEAR, solarH);

    float H = (solarH - 12.0f) * 15.0f * DEG2RAD; // hour angle

    float sinLat = sin(latRad), cosLat = cos(latRad);
    float sinD   = sin(declRad), cosD  = cos(declRad);

    // Local ENU components of direction (observer -> sun)
    float east  =  cosD * sin(H);
    float north =  cosLat * sinD - sinLat * cosD * cos(H);
    float up    =  sinLat * sinD + cosLat * cosD * cos(H);

    dirWS   = ENU_ToWorld(east, north, up);
    elevRad = asin(clamp(up, -1.0f, 1.0f));
}

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

    // Hard horizon cutoff
    float horizonRad = SUN_HORIZON_DEG * DEG2RAD;
    S.visible = (S.elevRad > horizonRad) ? 1.0f : 0.0f;

    // Dusk/dawn tint via transmittance
    float3 Tr = SunTransmittance(S.elevRad, SUN_TURBIDITY);
    S.tint    = SUN_COLOR_VAL * Tr;

    S.pdf      = (S.visible > 0.0f) ? (1.0f / S.omega) : 0.0f;

    // Uniform disk radiance per steradian
    // (Le/pdf == SUN_INTENSITY_VAL * tint, so SUN_INTENSITY_VAL is stable across disk size)
    S.radiance = (S.visible > 0.0f) ? (S.tint * SUN_INTENSITY_VAL / S.omega) : float3(0, 0, 0);

    return S;
}

// ============================================================================
// Public API (same signatures as your previous file)
// ============================================================================

// 1) NEE sampling (uniform cone over sun disk)
SunSampleResult SampleSun(float2 u)
{
    SunState S = ComputeSunState();

    SunSampleResult r;
    r.dist = SUN_DIST_INF;

    // If sun not visible, return pdf=0 and radiance=0 (caller should ignore)
    if (S.pdf <= 0.0f)
    {
        r.direction = S.dirWS;
        r.radiance  = float3(0, 0, 0);
        r.pdf       = 0.0f;
        return r;
    }

    // Uniform cone sampling about S.dirWS
    float z = 1.0f - u.y * (1.0f - S.cosThetaMax);
    float sinTheta = sqrt(max(0.0f, 1.0f - z * z));
    float phi = TAU * u.x;

    float3 localDir = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, z);

    float3 T, B;
    GetOrthoBasis(S.dirWS, T, B);

    r.direction = SafeNormalize(localDir.x * T + localDir.y * B + localDir.z * S.dirWS);
    r.pdf       = S.pdf;
    r.radiance  = S.radiance;
    return r;
}

// 2) MIS PDF ("as if sampled the light")
float GetSunPdf(float3 rayDir)
{
    SunState S = ComputeSunState();

    if (S.pdf <= 0.0f) return 0.0f;

    float3 d = SafeNormalize(rayDir);
    return (dot(d, S.dirWS) >= S.cosThetaMax) ? S.pdf : 0.0f;
}

// 3) Evaluation (hit sky)
float3 EvaluateSun(float3 rayDir)
{
    SunState S = ComputeSunState();

    if (S.pdf <= 0.0f) return float3(0, 0, 0);

    float3 d = SafeNormalize(rayDir);
    return (dot(d, S.dirWS) >= S.cosThetaMax) ? S.radiance : float3(0, 0, 0);
}

// ============================================================================
// Improved sky: Preetham/Perez + long twilight + robust stars
// Drop-in replacement for EvaluateSky(float3 rayDir)
// ============================================================================

// Optional knobs
#ifndef SKY_INTENSITY
// Scale sky with sun intensity to keep relative exposure stable in your renderer.
#define SKY_INTENSITY 0.02
#endif

#ifndef SKY_TWILIGHT_DEG
// Astronomical twilight end (~ -18). Larger magnitude = longer dusk/dawn.
#define SKY_TWILIGHT_DEG 12.0f
#endif

#ifndef SKY_STAR_GRID
// Star cell grid resolution (higher => more, smaller stars)
#define SKY_STAR_GRID 900.0f
#endif

#ifndef SKY_STAR_DENSITY
// 0.0..1.0 : higher => fewer stars. 0.997 gives a good amount.
#define SKY_STAR_DENSITY 0.9989f
#endif

#ifndef SKY_STAR_INTENSITY
#define SKY_STAR_INTENSITY 0.0f
#endif

#ifndef SKY_STAR_SCALE
// SDR star brightness scale (NOT multiplied by SKY_INTENSITY)
#define SKY_STAR_SCALE 0.0f
#endif

#ifndef SKY_SIDEREAL_RATIO
// Stars rotate ~1.0027379x solar rate (sidereal day)
#define SKY_SIDEREAL_RATIO 1.00273790935f
#endif

#ifndef SKY_GROUND_DARKEN
// 0..1. 0.04 means ground ~4% of horizon brightness when looking straight down.
#define SKY_GROUND_DARKEN 0.04f
#endif

#ifndef SKY_NIGHT_BASE
// Base night airglow-ish level
#define SKY_NIGHT_BASE float3(0.00015f, 0.00020f, 0.00035f)
#endif


inline float Smooth01(float x)
{
    x = saturate(x);
    return x * x * (3.0f - 2.0f * x);
}

inline float Hash12(float2 p)
{
    // fast-ish hash for cell coords (stable)
    // (sin is fine in miss/env; if you want, swap with integer hash)
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453f);
}

inline float3 XYZTosRGB(float3 XYZ)
{
    // D65, sRGB primaries, linear RGB
    float3 rgb;
    rgb.x =  3.2406f * XYZ.x + -1.5372f * XYZ.y + -0.4986f * XYZ.z;
    rgb.y = -0.9689f * XYZ.x +  1.8758f * XYZ.y +  0.0415f * XYZ.z;
    rgb.z =  0.0557f * XYZ.x + -0.2040f * XYZ.y +  1.0570f * XYZ.z;
    return max(rgb, 0.0f);
}

inline float3 xyY_to_RGB(float x, float y, float Y)
{
    y = max(y, 1e-4f);
    float X = (x / y) * Y;
    float Z = ((1.0f - x - y) / y) * Y;
    return XYZTosRGB(float3(X, Y, Z));
}

inline float3 RotateAroundAxis(float3 v, float3 axis, float angle)
{
    // Rodrigues rotation
    float s = sin(angle), c = cos(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0f - c);
}

// Perez distribution function
inline float Perez(float A, float B, float C, float D, float E, float theta, float gamma)
{
    float cosTheta = max(0.01f, cos(theta));
    float expTerm  = exp(B / cosTheta);
    float cg       = cos(gamma);
    float cg2      = cg * cg;
    return (1.0f + A * expTerm) * (1.0f + C * exp(D * gamma) + E * cg2);
}

// Preetham coefficients for Y / x / y
inline void PerezCoeffs_Y(float T, out float A, out float B, out float C, out float D, out float E)
{
    A =  0.1787f * T - 1.4630f;
    B = -0.3554f * T + 0.4275f;
    C = -0.0227f * T + 5.3251f;
    D =  0.1206f * T - 2.5771f;
    E = -0.0670f * T + 0.3703f;
}
inline void PerezCoeffs_x(float T, out float A, out float B, out float C, out float D, out float E)
{
    A = -0.0193f * T - 0.2592f;
    B = -0.0665f * T + 0.0008f;
    C = -0.0004f * T + 0.2125f;
    D = -0.0641f * T - 0.8989f;
    E = -0.0033f * T + 0.0452f;
}
inline void PerezCoeffs_y(float T, out float A, out float B, out float C, out float D, out float E)
{
    A = -0.0167f * T - 0.2608f;
    B = -0.0950f * T + 0.0092f;
    C = -0.0079f * T + 0.2102f;
    D = -0.0441f * T - 1.6537f;
    E = -0.0109f * T + 0.0529f;
}

// Zenith chromaticity/luminance (Preetham 1999 fit)
inline void Zenith_xyY(float T, float thetaS, out float xZ, out float yZ, out float YZ)
{
    // thetaS = sun zenith angle in radians, clamped to [0, ~pi/2]
    float t  = thetaS;
    float t2 = t * t;
    float t3 = t2 * t;
    float T2 = T * T;

    // Zenith luminance (kcd/m^2-ish in the original model). We'll treat it as relative and scale later.
    float chi = (4.0f / 9.0f - T / 120.0f) * (PI - 2.0f * thetaS);
    YZ = (4.0453f * T - 4.9710f) * tan(chi) - 0.2155f * T + 2.4192f;

    // Zenith chromaticities
    xZ =
        ( 0.00165f * t3 - 0.00374f * t2 + 0.00208f * t + 0.00000f) * T2 +
        (-0.02902f * t3 + 0.06377f * t2 - 0.03202f * t + 0.00394f) * T  +
        ( 0.11693f * t3 - 0.21196f * t2 + 0.06052f * t + 0.25885f);

    yZ =
        ( 0.00275f * t3 - 0.00610f * t2 + 0.00316f * t + 0.00000f) * T2 +
        (-0.04214f * t3 + 0.08970f * t2 - 0.04153f * t + 0.00515f) * T  +
        ( 0.15346f * t3 - 0.26756f * t2 + 0.06670f * t + 0.26688f);

    // Clamp to sane range
    xZ = clamp(xZ, 0.0f, 1.0f);
    yZ = clamp(yZ, 0.0f, 1.0f);
    YZ = max(YZ, 0.0f);
}

// Main sky evaluation
// Main sky evaluation
float3 EvaluateSky(float3 rayDir)
{
    SunState S = ComputeSunState();

    float3 v = SafeNormalize(rayDir);
    float  upDot = dot(v, WORLD_UP);

    // --- Star motion (sidereal rotation) ---
    float solarH = GetSolarTimeHours();
    float sidFrac = frac((solarH / 24.0f) * SKY_SIDEREAL_RATIO);
    float starAngle = TAU * sidFrac;

    float3 vStar = RotateAroundAxis(v, WORLD_UP, -starAngle);

    // --- Ground handling ---
    bool isGround = (upDot <= 0.0f);
    float3 vEval  = isGround ? SafeNormalize(float3(v.x, 1e-4f, v.z)) : v;

    float upDotEval = dot(vEval, WORLD_UP);

    // Angles
    float theta   = acos(clamp(upDotEval, 0.0f, 1.0f));
    float sunUp   = dot(S.dirWS, WORLD_UP);
    float elevDeg = S.elevRad * RAD2DEG;

    float thetaS = acos(clamp(sunUp, 0.0f, 1.0f));
    thetaS = min(thetaS, (PI * 0.5f) - 1e-3f);

    float gamma = acos(clamp(dot(vEval, S.dirWS), -1.0f, 1.0f));

    // Turbidity & Zenith
    float T = max(1.0f, SUN_TURBIDITY);
    float xZ, yZ, YZ;
    Zenith_xyY(T, thetaS, xZ, yZ, YZ);

    // Perez coefficients
    float Ay, By, Cy, Dy, Ey;
    float Ax, Bx, Cx, Dx, Ex;
    float Ayy, Byy, Cyy, Dyy, Eyy;

    PerezCoeffs_Y(T, Ay,  By,  Cy,  Dy,  Ey);
    PerezCoeffs_x(T, Ax,  Bx,  Cx,  Dx,  Ex);
    PerezCoeffs_y(T, Ayy, Byy, Cyy, Dyy, Eyy);

    float Fy0  = Perez(Ay,  By,  Cy,  Dy,  Ey,  0.0f, thetaS);
    float Fx0  = Perez(Ax,  Bx,  Cx,  Dx,  Ex,  0.0f, thetaS);
    float Fyy0 = Perez(Ayy, Byy, Cyy, Dyy, Eyy, 0.0f, thetaS);

    float Fy  = Perez(Ay,  By,  Cy,  Dy,  Ey,  theta, gamma) / max(1e-4f, Fy0);
    float Fx  = Perez(Ax,  Bx,  Cx,  Dx,  Ex,  theta, gamma) / max(1e-4f, Fx0);
    float Fyy = Perez(Ayy, Byy, Cyy, Dyy, Eyy, theta, gamma) / max(1e-4f, Fyy0);

    float Y = YZ * Fy;
    float x = xZ * Fx;
    float y = yZ * Fyy;

    float3 daySky = xyY_to_RGB(x, y, Y);

    float3 tint = max(S.tint, 0.0f);
    daySky *= lerp(float3(1,1,1), normalize(tint + 1e-6f) * 1.15f, 0.35f);

    float tw = Smooth01(saturate((elevDeg + SKY_TWILIGHT_DEG) / SKY_TWILIGHT_DEG));
    daySky *= tw;

    float mu = saturate(upDotEval);
    float3 nightBase = SKY_NIGHT_BASE * lerp(1.6f, 1.0f, pow(mu, 0.7f));
    nightBase *= SKY_INTENSITY;

    float starW = Smooth01(saturate((-elevDeg - 6.0f) / max(1.0f, (SKY_TWILIGHT_DEG - 6.0f))));
    starW *= pow(saturate(dot(vStar, WORLD_UP)), 0.25f);

    // --- STABLE STARS (DLSS RR & TAA Friendly) ---
    float2 uv;
    uv.x = atan2(vStar.z, vStar.x) / TAU + 0.5f;
    uv.y = acos(clamp(vStar.y, -1.0f, 1.0f)) / PI;

    float2 g = uv * SKY_STAR_GRID;
    float2 cell = floor(g);
    float2 f = frac(g);

    float r0 = Hash12(cell);
    float r1 = Hash12(cell + 17.0f);
    float r2 = Hash12(cell + 41.0f);

    float present = step(SKY_STAR_DENSITY, r0);

    // Keep star center away from the absolute edge of the cell to prevent grid clipping on larger stars
    float2 c = float2(Hash12(cell + 3.0f), Hash12(cell + 7.0f));
    c = lerp(0.2f, 0.8f, c);

    float2 d = f - c;
    float dist = length(d);

    float tmag = saturate((r0 - SKY_STAR_DENSITY) / max(1e-6f, (1.0f - SKY_STAR_DENSITY)));
    float mag  = pow(tmag, 1.5f); // Softer curve, less aggressive HDR peaking

    // STABILITY FIX 1: Much larger radius. Spans 15% to 45% of a grid cell.
    // This ensures the star covers multiple pixels, proving to DLSS that it is "structure" and not "noise".
    float radius = lerp(0.15f, 0.45f, mag);

    // STABILITY FIX 2: Soft falloff. Replaced the harsh power curve with a smoothstep gradient.
    float core = saturate(1.0f - (dist / max(1e-4f, radius)));
    core = smoothstep(0.0f, 1.0f, core);

    float3 starCol = lerp(float3(0.85f, 0.90f, 1.00f), float3(1.00f, 0.92f, 0.80f), r1);

    // STABILITY FIX 3: Drastically slowed down the twinkle.
    // Rapid temporal changes trigger denoiser rejection. This creates a slow, imperceptible pulse instead.
    float twk = 0.96f + 0.04f * sin((float)SUN_FRAMECOUNT * 0.001f + r2 * 60.0f);

    float3 stars = present * core * mag * twk * starCol * SKY_STAR_INTENSITY * SKY_STAR_SCALE * starW;

    float3 nightSky = nightBase + stars;
    float3 sky = lerp(nightSky, daySky * SKY_INTENSITY, tw);

    if (isGround)
    {
        float t = Smooth01(saturate((-upDot) / 0.35f));
        float3 horizonCol = sky;
        return lerp(horizonCol, horizonCol * SKY_GROUND_DARKEN, t);
    }

    sky = min(sky, 2.0f);

    return sky;
}

#endif // SUN_LIGHT_HLSLI