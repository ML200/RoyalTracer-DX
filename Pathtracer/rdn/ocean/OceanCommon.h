#pragma once

#include "../Common.h"
#include "../../shaders/OceanLayout.h"
#include <array>
#include <cmath>

namespace ocean {

constexpr double kGravity = 9.80665;

// Optional test the quadtree asks about a square of the plane. The sea covers its whole extent,
// and in a world whose water covers only part of it - a block world's coast and lakes, with a city
// on the rest - every ray pays to traverse water tiles standing where there is no water.
//
// Three answers, not two, because a yes/no test is not enough: a tile the size of the map that
// happens to contain a harbour would answer yes and stay that size, and its bounding box would
// then enclose everything else in the scene. Partial means "split me": subdividing until each
// tile is either all water or none of it is what actually keeps the bounds tight, and tight
// bounds are the whole point - a box that overlaps the world is one every ray has to enter.
//
// Coordinates are absolute world XZ, matching the quadtree's own; `size` is the tile's edge.
// Answering Full where there is no water only costs what it cost before; answering None where
// there is water would leave a hole, so err towards Partial when unsure.
enum class Coverage : uint8_t { None, Partial, Full };

struct ICoverage {
    virtual ~ICoverage() = default;
    virtual Coverage Test(double minX, double minZ, double size) const = 0;
};

// User-facing description of a sea state. Wind speed and fetch drive the JONSWAP spectrum, so a
// realistic sea needs only those two plus the water's optical type.
struct Params {
    bool enabled = true;

    // Wind speed at 10 m, m/s. Beaufort 5 ~ 10 m/s, Beaufort 7 ~ 15 m/s.
    float windSpeed = 11.0f;
    // Distance over which the wind has been blowing, metres. Open ocean is effectively unlimited;
    // 200 km already produces a nearly fully developed sea.
    float fetch = 200000.0f;
    // Wind bearing in degrees, clockwise from +Z.
    float windDirectionDeg = 30.0f;
    // 0 keeps the full wind-sea spread, 1 narrows it toward long-crested swell. Pushing this up
    // quickly turns the sea into parallel corrugations, so the default stays low.
    float swell = 0.15f;
    // Exponent of the downwind lobe. 0 is symmetric, higher values suppress waves travelling into
    // the wind and make the crests more parallel.
    float windAlign = 0.8f;
    // Negative values retain the existing wind/fetch-derived spectrum. Otherwise these
    // physical controls take precedence; amplitudeScale remains a final artistic gain.
    float significantHeight = -1.0f;
    float peakPeriod = -1.0f;
    float swellHeight = 0.0f;
    float swellPeriod = 11.0f;
    float swellDirectionDeg = 80.0f;
    float swellSpreadDeg = 12.0f;
    bool paused = false;
    float fixedTimeStep = 0.0f; // deterministic capture/testing; 0 uses frame time
    float bodyWeight = 0.0f; // optional diffuse surface contribution; clear water uses transmission + SSS
    uint32_t debugMode = 0; // 0 beauty, 1 normal, 2 compression, 3 covariance, 4 roughness, 5 foam/freshness, 6 mip

    // How hard the wind whips the surface, on top of the height it already sets. Scales both the
    // horizontal displacement that sharpens crests and the gain on the short wind waves, so 0
    // leaves smooth rounded swell and higher values peak the small detail. 1 is the calibrated
    // sea; the wind speed carries part of this on its own (see WindChop).
    float turbulence = 1.0f;

    // Kilometre-scale variation in sea state. A real ocean is not one sea everywhere: currents
    // shear the surface, wind arrives in gusts and lulls, and the result is patches of steeper,
    // more broken water drifting between calmer lanes. A tiling fBm of the gain the whole wave
    // field is multiplied by reproduces that. 0 gives the uniform sea; 0.4 means the roughest
    // patches carry 40% more wave than the mean and the calmest 40% less.
    float turbulenceVariation = 0.35f;
    // Tiling period of that field in metres. The octaves inside it run from this down to a
    // sixteenth of it, so the patches themselves are a few hundred metres to a few kilometres.
    float turbulencePeriod = 9000.0f;

    // Second-order Stokes crest sharpening, as a multiple of the physical bound harmonic: 0 gives
    // the symmetric Gaussian sea a linear spectrum produces, 1 the real wave's narrow crest and
    // long shallow trough, and higher values push each band towards its steepness limit. Unlike
    // choppiness this costs nothing against the no-fold bound, so it is what actually buys a sharp
    // crest once the horizontal displacement has saturated.
    float crestSharpening = 1.0f;

    // Requested horizontal displacement gain, limited by the per-frame composite deformation bound.
    float choppiness = 1.0f;
    // Overall wave height multiplier; 1 is the physical JONSWAP height.
    float amplitudeScale = 1.0f;
    // Amplitude gain on wind waves shorter than 8 m, reaching full gain below 2 m.
    // Leaves the long wind sea and independent swell unchanged; 1 uses the unmodified spectrum.
    float shortWaveAmplitude = 1.5f;

    // Chlorophyll concentration, mg/m^3, feeding Morel's Case-1 model. 0.03 is clear open ocean
    // (deep indigo), 1-10 is coastal green.
    float chlorophyll = 0.06f;
    // Extra scattering multiplier for turbid or sediment-laden water.
    float turbidity = 1.0f;

    // Legacy foam controls retained for source compatibility; foam rendering is removed.
    float foamCoverage = 0.0f;
    float foamDecay = 0.55f; // fraction of foam surviving each second
    // Local bubble-raft reflectance. Coverage is accounted for separately, so this must not
    // contain a second coverage average (which would turn whitecaps grey).
    float foamAlbedo = 0.65f;

    bool legacySubsurface = false; // retained API field; solid-object SSS is never used for water
    // Water volume controls. Radius scales the scattering mean free path in metres.
    float subsurfaceStrength = 1.0f;
    float subsurfaceRadiusScale = 1.0f;
    float subsurfacePhaseG = 0.8f;

    // Follow the curve of the Earth. Without it the horizon sits at infinity and distant ships
    // never drop below it, which reads as wrong immediately in a wide ocean shot.
    bool curvature = true;

    // Tiles the quadtree may keep resident. This is the ocean's cost knob: every tile is an
    // acceleration structure rebuilt or refitted each frame, an extra instance in the scene's top
    // level, and a block of the shared vertex and index buffers. An open-ocean shot wants the full
    // budget; a sea capping a block world does not. Read once, when the buffers are sized.
    uint32_t maxTiles = OCEAN_MAX_TILES;

    // Half-width of the simulated ocean in metres. The quadtree root spans 2x this.
    float extent = 60000.0f;
    // Smallest tile edge in metres; sets the finest displaced geometry.
    float minTileSize = 8.0f;
    // Tile edge length as a fraction of the distance to the camera. Lower is finer and costs
    // proportionally more acceleration-structure builds.
    float lodFactor = 0.42f;
    // Tiles nearer than this are kept even when outside the frustum, so reflections and shadows
    // of nearby waves stay correct.
    float nearKeepRadius = 400.0f;

    // Ray footprint multiplier before it picks the geometry/BRDF split. Raise it if the horizon
    // shimmers, lower it if the near surface looks over-smoothed.
    float filterScale = 1.0f;

    // Roughness floor the sun sampler and next-event estimation widen the water surface to, and
    // nothing else: continuation rays, environment reflections and the reconstruction guides keep
    // the authored roughness and the full-resolution wave normal. Clear water is authored
    // mirror-flat, which leaves direct lighting a delta lobe whose glitter resolves as isolated
    // fireflies rather than as a sun track, so this trades highlight sharpness against that noise.
    float sunLobeRoughness = 0.1f;

    // Mean sea level. The waves swing symmetrically about it, so a level of zero puts every trough
    // below the origin.
    float seaLevelY = 0.0f;
    // Lift the mean level so the deepest trough still clears this height. The atmosphere model
    // treats anything below the ground plane as underground, so a sea centred on zero renders its
    // own troughs black; the offset is computed from the wave height once the spectrum is baked.
    bool keepAboveZero = true;
    float minClearance = 0.5f; // metres the lowest trough should stay above zero

    uint32_t seed = 1337;
};

// Reject invalid state at the public boundary; NaNs must never reach spectra or GPU indices.
inline void ValidateParams(const Params& p) {
    const float values[] = {p.windSpeed,p.fetch,p.windDirectionDeg,p.swell,p.windAlign,p.significantHeight,
        p.peakPeriod,p.swellHeight,p.swellPeriod,p.swellDirectionDeg,p.swellSpreadDeg,p.fixedTimeStep,
        p.bodyWeight,p.turbulence,p.turbulenceVariation,p.turbulencePeriod,p.crestSharpening,p.choppiness,p.amplitudeScale,p.shortWaveAmplitude,p.chlorophyll,p.turbidity,p.foamCoverage,p.foamDecay,
        p.foamAlbedo,p.subsurfaceStrength,p.subsurfaceRadiusScale,p.subsurfacePhaseG,p.extent,p.minTileSize,p.lodFactor,p.nearKeepRadius,
        p.filterScale,p.sunLobeRoughness,p.seaLevelY,p.minClearance};
    for (float v : values) if (!std::isfinite(v)) throw std::invalid_argument("Ocean parameters must be finite");
    if (p.windSpeed < 0 || p.fetch <= 0 || p.windAlign < 0 || p.swell < 0 || p.swell > 1 ||
        p.swellHeight < 0 || p.swellPeriod <= 0 || p.swellSpreadDeg < 2 || p.swellSpreadDeg > 90 ||
        p.peakPeriod == 0 || p.fixedTimeStep < 0 || p.fixedTimeStep > 0.1f ||
        p.turbulence < 0 || p.turbulence > 3 || p.crestSharpening < 0 || p.crestSharpening > 8 ||
        p.turbulenceVariation < 0 || p.turbulenceVariation > 0.9f || p.turbulencePeriod < 64 ||
        p.choppiness < 0 || p.amplitudeScale < 0 || p.shortWaveAmplitude < 0 || p.shortWaveAmplitude > 3 || p.chlorophyll < 0 || p.turbidity < 0 ||
        p.foamCoverage < 0 || p.foamDecay <= 0 || p.foamDecay > 1 || p.foamAlbedo < 0 || p.foamAlbedo > 1 ||
        p.bodyWeight < 0 || p.bodyWeight > 1 || p.subsurfaceStrength < 0 ||
        p.subsurfaceRadiusScale <= 0 || std::abs(p.subsurfacePhaseG) >= 1 ||
        p.sunLobeRoughness < 0 || p.sunLobeRoughness > 1 ||
        p.maxTiles == 0 || p.maxTiles > (uint32_t)OCEAN_MAX_TILES ||
        p.extent < 64 || p.minTileSize < 1 || p.lodFactor <= 0 || p.nearKeepRadius < 0 || p.filterScale <= 0 || p.debugMode > 6)
        throw std::invalid_argument("Ocean parameter outside its supported range");
}
// Cascade tile lengths in metres. The outermost one has to be large compared to the field of view
// or its period shows as repeated structure across the frame, and it also has to be much longer
// than the peak wavelength so the energy-carrying waves sit well inside its grid rather than on
// the first few modes. The ratios are deliberately not powers of two, so the combined surface
// repeats on their least common multiple rather than on the largest of them.
inline const std::array<double, OCEAN_CASCADES>& CascadeLengths() {
    static const std::array<double, OCEAN_CASCADES> kL = {8192.0, 977.0, 119.0, 17.0};
    return kL;
}

inline double CascadeFundamental(int c) {
    return 6.283185307179586 / CascadeLengths()[(size_t)c];
}

inline double CascadeNyquist(int c) {
    return 3.141592653589793 * OCEAN_FFT_SIZE / CascadeLengths()[(size_t)c];
}

// Wind speed the whipped look is calibrated at: the default sea state, Beaufort 6.
constexpr double kReferenceWind = 11.0;

// Chop the wind carries on its own. A breeze leaves rounded swell however long it has been
// blowing, while a gale shears every crest it raises, so the whipped look has to track wind speed
// and not only wave height. Written as a deviation from the reference wind so that it is exactly
// one there, leaving the calibrated sea untouched.
inline double WindChop(const Params& p) {
    const double t = ((double)p.windSpeed - kReferenceWind) / kReferenceWind;
    return std::clamp(1.0 + 0.55 * t, 0.45, 1.5);
}

// Combined whipping gain: the user's turbulence on top of what the wind already supplies.
inline double ChopGain(const Params& p) {
    return std::max(0.0, (double)p.turbulence) * WindChop(p);
}

// Horizontal displacement gain the simulation actually asks for. The GPU still bounds the
// composite deformation each frame, so this is a request rather than a guarantee.
inline double EffectiveChoppiness(const Params& p) {
    return (double)p.choppiness * ChopGain(p);
}

// Largest gain the kilometre-scale sea-state field reaches. Everything that has to hold over the
// whole surface - the crest sharpening's monotonic range, the composite deformation budget - is
// solved against the roughest patch rather than against the mean sea.
inline double TurbulenceMaxGain(const Params& p) {
    return 1.0 + std::clamp((double)p.turbulenceVariation, 0.0, 0.9);
}

// Largest skew coefficient a band may carry, as a steepness a*sigma. The warp turns around at
// -1/(2a), so this holds that point past four standard deviations of the band's own elevation:
// beyond it the deepest troughs would come back up as a second crest.
constexpr double kMaxSkewSteepness = 0.12;

// Second-order Stokes coefficient for a band of mean wavenumber kBar and elevation standard
// deviation sigma. A physical bound harmonic has a = k; the short cascades are already near their
// steepness limit and clamp there, which is exactly the band whose crests break in a real sea.
inline double CrestSkew(const Params& p, double kBar, double sigma) {
    const double requested = std::max(0.0, (double)p.crestSharpening) * ChopGain(p) * kBar;
    // The kilometre-scale field raises this band's elevation inside a rough patch, so the
    // monotonic bound is solved against the roughest patch rather than against the mean sea.
    const double peak = sigma * TurbulenceMaxGain(p);
    const double limit = peak > 1e-6 ? kMaxSkewSteepness / peak : 0.0;
    return std::min(requested, limit);
}

inline double ShortWaveAmplitude(const Params& p, double k) {
    constexpr double kStart = 6.283185307179586 / 8.0;
    constexpr double kFull = 6.283185307179586 / 2.0;
    double t = std::clamp((k - kStart) / (kFull - kStart), 0.0, 1.0);
    t = t * t * (3.0 - 2.0 * t);
    // The gain is squared into a power below, so it must not be allowed to swing negative -
    // that would turn a suppressed band back into a boosted one.
    return std::max(0.0, 1.0 + (double(p.shortWaveAmplitude) - 1.0) * ChopGain(p) * t);
}

// The range over which a cascade actually carries energy. The low end is held a few modes above
// the fundamental because a cascade has almost no resolution there, and the high end stops at its
// Nyquist limit.
inline void CascadeBand(int c, double& kMin, double& kMax) {
    kMin = CascadeFundamental(c);
    kMax = CascadeNyquist(c);
}

// Unnormalised share of the spectrum this cascade takes at wavenumber k: one inside the range it
// resolves well, rolling off smoothly at both ends.
inline double CascadeWindow(int c, double k) {
    const double lo1 = CascadeFundamental(c);
    const double lo2 = lo1 * 6.0;   // below six modes a cascade barely resolves the wave
    const double hi2 = CascadeNyquist(c);
    const double hi1 = hi2 * 0.4;
    if (k <= lo1 || k >= hi2)
        return 0.0;
    auto smooth = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };
    const double rise = smooth((std::log(k) - std::log(lo1)) / (std::log(lo2) - std::log(lo1)));
    const double fall = 1.0 - smooth((std::log(k) - std::log(hi1)) / (std::log(hi2) - std::log(hi1)));
    return rise * fall;
}

// Partition of unity across the cascades. Power is what has to sum to one, so an amplitude picks
// up the square root of this.
inline double CascadeWeight(int c, double k) {
    double total = 0.0;
    for (int j = 0; j < OCEAN_CASCADES; ++j)
        total += CascadeWindow(j, k);
    if (total <= 1e-9)
        return 0.0;
    return CascadeWindow(c, k) / total;
}

// Highest wavenumber any cascade can represent; everything above it lives in the BRDF.
inline double NyquistK() {
    double k = 0.0;
    for (int c = 0; c < OCEAN_CASCADES; ++c)
        k = std::max(k, CascadeNyquist(c));
    return k;
}

// --------------------------------------------------------------------------------------------
// JONSWAP with Donelan-Banner spreading
// --------------------------------------------------------------------------------------------

struct Spectrum {
    double alpha = 0.0;
    double omegaP = 0.0;
    double gamma = 3.3;
    double swell = 0.0;
    double windAlign = 0.0;
    double energyScale = 1.0;
    // Per-omega normalisation of the directional term, so that its integral over theta is 1.
    static constexpr int kNormBins = 256;
    std::array<double, kNormBins> dirNorm{};
    double omegaNormMin = 0.02;
    double omegaNormMax = 40.0;

    void Init(const Params& p);

    double S(double omega) const; // frequency spectrum, m^2 s
    double D(double omega, double theta) const; // normalised directional spreading, 1/rad
    // Two-sided wavenumber spectrum in m^4, already including the omega->k jacobian.
    double S2D(double kx, double kz) const;

  private:
    double DRaw(double omega, double theta) const;
    double DirNormAt(double omega) const;
};

inline void Spectrum::Init(const Params& p) {
    const double U = std::max(0.1, (double)p.windSpeed);
    const double F = std::max(1000.0, (double)p.fetch);
    alpha = 0.076 * std::pow(U * U / (F * kGravity), 0.22);
    omegaP = 22.0 * std::pow(kGravity * kGravity / (U * F), 1.0 / 3.0);
    gamma = 3.3;
    swell = std::clamp((double)p.swell, 0.0, 1.0);
    windAlign = std::max(0.0, (double)p.windAlign);
    if (p.peakPeriod > 0.0f) omegaP = 6.283185307179586 / std::max(0.5, (double)p.peakPeriod);
    energyScale = 1.0;
    if (p.significantHeight >= 0.0f) {
        double integral = 0.0;
        constexpr int steps = 8192;
        const double lo = omegaP * 0.05, hi = omegaP * 100.0;
        const double logStep = std::log(hi / lo) / steps;
        for (int i = 0; i < steps; ++i) {
            const double w = lo * std::exp((i + 0.5) * logStep);
            integral += S(w) * w * logStep;
        }
        energyScale = double(p.significantHeight) * p.significantHeight / (16.0 * std::max(integral, 1e-30));
    }

    // Tabulate the directional normalisation over a log-spaced omega range; D is expensive enough
    // that evaluating the integral per grid point would dominate the bake.
    for (int i = 0; i < kNormBins; ++i) {
        const double t = (double)i / (double)(kNormBins - 1);
        const double omega = omegaNormMin * std::pow(omegaNormMax / omegaNormMin, t);
        const int kSteps = 512;
        double sum = 0.0;
        const double dTheta = 6.283185307179586 / (double)kSteps;
        for (int s = 0; s < kSteps; ++s) {
            const double theta = -3.141592653589793 + ((double)s + 0.5) * dTheta;
            sum += DRaw(omega, theta) * dTheta;
        }
        dirNorm[(size_t)i] = (sum > 1e-12) ? 1.0 / sum : 0.0;
    }
}

inline double Spectrum::S(double omega) const {
    if (omega <= 1e-4)
        return 0.0;
    const double sigma = (omega <= omegaP) ? 0.07 : 0.09;
    const double d = omega - omegaP;
    const double r = std::exp(-(d * d) / (2.0 * sigma * sigma * omegaP * omegaP));
    const double wp_w = omegaP / omega;
    const double body = alpha * kGravity * kGravity / std::pow(omega, 5.0);
    return energyScale * body * std::exp(-1.25 * std::pow(wp_w, 4.0)) * std::pow(gamma, r);
}

inline double Spectrum::DRaw(double omega, double theta) const {
    // Donelan-Banner beta_s, with the three published frequency regimes.
    const double ratio = std::max(0.56, omega / std::max(omegaP, 1e-6));
    double beta;
    if (ratio < 0.95)
        beta = 2.61 * std::pow(ratio, 1.3);
    else if (ratio < 1.6)
        beta = 2.28 * std::pow(ratio, -1.3);
    else {
        const double eps = -0.4 + 0.8393 * std::exp(-0.567 * std::log(ratio * ratio));
        beta = std::pow(10.0, eps);
    }
    // Swell narrows the lobe: a long-travelled sea arrives far more directional than it left.
    beta *= 1.0 + 3.0 * swell;

    const double ch = std::cosh(std::min(beta * theta, 30.0));
    double d = 1.0 / (ch * ch);
    if (windAlign > 0.0)
        d *= std::pow(std::max(0.0, 0.5 + 0.5 * std::cos(theta)), windAlign);
    return d;
}

inline double Spectrum::DirNormAt(double omega) const {
    const double t = std::log(std::clamp(omega, omegaNormMin, omegaNormMax) / omegaNormMin) /
                     std::log(omegaNormMax / omegaNormMin);
    const double f = t * (double)(kNormBins - 1);
    const int i0 = std::clamp((int)f, 0, kNormBins - 1);
    const int i1 = std::min(i0 + 1, kNormBins - 1);
    const double frac = f - (double)i0;
    return dirNorm[(size_t)i0] * (1.0 - frac) + dirNorm[(size_t)i1] * frac;
}

inline double Spectrum::D(double omega, double theta) const {
    return DRaw(omega, theta) * DirNormAt(omega);
}

inline double Spectrum::S2D(double kx, double kz) const {
    const double k2 = kx * kx + kz * kz;
    if (k2 < 1e-12)
        return 0.0;
    const double k = std::sqrt(k2);
    // Deep water: omega = sqrt(g k), dOmega/dk = 0.5 sqrt(g/k).
    const double omega = std::sqrt(kGravity * k);
    const double dOmegaDk = 0.5 * std::sqrt(kGravity / k);
    const double theta = std::atan2(kz, kx);
    return S(omega) * D(omega, theta) * dOmegaDk / k;
}

// --------------------------------------------------------------------------------------------
// Sea state summary, from a one-dimensional integral of the frequency spectrum. The directional
// term integrates to one, so this needs no angular pass and can be evaluated before the wave field
// is baked - which is what lets a scene position a camera or a hull against the water line.
// --------------------------------------------------------------------------------------------

inline double PredictElevationVariance(const Params& p) {
    Spectrum s;
    s.Init(p);
    const double wMin = 0.05, wMax = 25.0;
    const int steps = 4000;
    const double dw = (wMax - wMin) / (double)steps;
    double var = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double omega = wMin + ((double)i + 0.5) * dw;
        const double detailGain = ShortWaveAmplitude(p, omega * omega / kGravity);
        var += s.S(omega) * detailGain * detailGain * dw;
    }
    const double amp = std::max(0.0, (double)p.amplitudeScale);
    return (var + double(p.swellHeight) * p.swellHeight / 16.0) * amp * amp;
}

// Independent incoming swell: log-normal angular-frequency density and normalized
// wrapped Gaussian direction. Both integrate to one; Hm0 supplies the component energy.
inline double SwellDensity(const Params& p, double kx, double kz) {
    const double k = std::hypot(kx, kz);
    if (k < 1e-8 || p.swellHeight <= 0.0f) return 0.0;
    const double w = std::sqrt(kGravity * k);
    const double wp = 6.283185307179586 / std::max(0.5, double(p.swellPeriod));
    constexpr double sigma = 0.12;
    const double spectral = std::exp(-0.5 * std::pow(std::log(w / wp) / sigma, 2.0)) /
        (w * sigma * std::sqrt(6.283185307179586));
    const double bearing = double(p.swellDirectionDeg) * 0.017453292519943295;
    const double theta = std::remainder(std::atan2(kx, kz) - bearing, 6.283185307179586);
    const double spread = std::clamp(double(p.swellSpreadDeg), 2.0, 90.0) * 0.017453292519943295;
    double direction = 0.0;
    for (int wrap = -2; wrap <= 2; ++wrap)
        direction += std::exp(-0.5 * std::pow((theta + wrap * 6.283185307179586) / spread, 2.0));
    direction /= spread * std::sqrt(6.283185307179586);
    return double(p.swellHeight) * p.swellHeight / 16.0 * spectral * direction * kGravity / (2.0 * w * k);
}

inline double PredictSignificantWaveHeight(const Params& p) {
    return 4.0 * std::sqrt(std::max(0.0, PredictElevationVariance(p)));
}

// How far the deepest trough reaches below the mean. Elevation is Gaussian, so four and a half
// standard deviations covers all but about a millionth of the surface; choppiness sharpens the
// troughs beyond that.
inline double PredictWaveDepth(const Params& p) {
    const double sigma = std::sqrt(std::max(0.0, PredictElevationVariance(p)));
    return 4.5 * sigma * (1.0 + 0.3 * EffectiveChoppiness(p));
}

// Mean sea level the ocean will actually use.
inline double PredictSurfaceLevel(const Params& p) {
    double y = (double)p.seaLevelY;
    if (p.keepAboveZero)
        y = std::max(y, PredictWaveDepth(p) + (double)p.minClearance);
    return y;
}

// --------------------------------------------------------------------------------------------
// Cox & Munk (1954) mean-square slope of a clean sea surface, wind speed in m/s at 12.5 m. These
// are the measured totals the synthesised spectrum is calibrated against.
// --------------------------------------------------------------------------------------------

// Monahan & O'Muircheartaigh's whitecap coverage: the fraction of the sea surface that is actively
// breaking, W = 3.84e-6 U^3.41 with U in m/s at 10 m. At Beaufort 6 that is barely one per cent,
// which is far less foam than most ocean shaders draw - and the difference is a large part of why
// they read as stylised.
inline double WhitecapCoverage(double windSpeed10m) {
    const double U = std::max(0.0, windSpeed10m);
    return std::clamp(3.84e-6 * std::pow(U, 3.41), 0.0, 0.35);
}

// Inverse standard normal CDF by bisection on erfc. Called a handful of times per spectrum bake,
// so there is no reason to reach for a rational approximation.
inline double InverseNormalCdf(double p) {
    p = std::clamp(p, 1e-9, 1.0 - 1e-9);
    double lo = -8.0, hi = 8.0;
    for (int i = 0; i < 60; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double cdf = 0.5 * std::erfc(-mid * 0.7071067811865476);
        if (cdf < p)
            lo = mid;
        else
            hi = mid;
    }
    return 0.5 * (lo + hi);
}

inline void CoxMunkSlopeVariance(double windSpeed10m, double& varAlong, double& varCross) {
    // The relations are quoted at mast height; the logarithmic profile makes 12.5 m about 4%
    // slower than the 10 m reference wind.
    const double U = std::max(0.0, windSpeed10m) * 1.04;
    varAlong = 3.16e-3 * U;
    varCross = 0.003 + 1.92e-3 * U;
}

// --------------------------------------------------------------------------------------------
// Water optics: Pope & Fry (1997) pure-water absorption plus Morel's Case-1 relations for the
// chlorophyll-bearing component, band-averaged into linear RGB.
// --------------------------------------------------------------------------------------------

// Absorption of pure water, 1/m, at 20 nm steps from 400 to 700 nm.
inline const std::array<double, 16>& PureWaterAbsorption() {
    static const std::array<double, 16> a = {0.00663, 0.00454, 0.00635, 0.00979, 0.0127, 0.0204,
                                             0.0409,  0.0474,  0.0619,  0.0896,  0.2224, 0.2755,
                                             0.3108,  0.4100,  0.4650,  0.6240};
    return a;
}

// Scattering of pure sea water, 1/m (Morel 1974): b(500 nm) = 0.0029 with a lambda^-4.32 slope.
inline double PureSeaWaterScattering(double lambdaNm) {
    return 0.0029 * std::pow(500.0 / lambdaNm, 4.32);
}

// Averages a spectral quantity over rectangular R, G and B bands. Rectangular bands are a coarse
// stand-in for the sensor response, but the absorption varies by more than an order of magnitude
// across the visible range, so the band choice matters far less than using a spectrum at all.
template <typename Fn> inline XMFLOAT3 BandAverage(Fn&& f) {
    const double bands[3][2] = {{600.0, 700.0}, {500.0, 600.0}, {400.0, 500.0}};
    XMFLOAT3 out{};
    float* dst[3] = {&out.x, &out.y, &out.z};
    for (int b = 0; b < 3; ++b) {
        double sum = 0.0;
        int n = 0;
        for (double lambda = bands[b][0]; lambda <= bands[b][1] + 1e-6; lambda += 10.0) {
            sum += f(lambda);
            ++n;
        }
        *dst[b] = (float)(sum / std::max(1, n));
    }
    return out;
}

inline double PureWaterAbsorptionAt(double lambdaNm) {
    const auto& tbl = PureWaterAbsorption();
    const double t = std::clamp((lambdaNm - 400.0) / 20.0, 0.0, 15.0);
    const int i0 = std::min((int)t, 15);
    const int i1 = std::min(i0 + 1, 15);
    const double f = t - (double)i0;
    return tbl[(size_t)i0] * (1.0 - f) + tbl[(size_t)i1] * f;
}

// Morel Case-1: a(lambda) = (a_w + 0.06 C^0.65)(1 + 0.02 exp(-0.014(lambda - 380))).
inline XMFLOAT3 WaterAbsorptionRGB(double chlorophyll) {
    const double C = std::max(0.0, chlorophyll);
    return BandAverage([&](double lambda) {
        const double aw = PureWaterAbsorptionAt(lambda);
        const double pigment = 0.06 * std::pow(std::max(C, 1e-6), 0.65);
        return (aw + pigment) * (1.0 + 0.02 * std::exp(-0.014 * (lambda - 380.0)));
    });
}

// Morel Case-1 particulate scattering: b_p(lambda) = (550/lambda) 0.30 C^0.62, over the molecular
// scattering of pure sea water.
inline XMFLOAT3 WaterScatteringRGB(double chlorophyll, double turbidity) {
    const double C = std::max(0.0, chlorophyll);
    return BandAverage([&](double lambda) {
        const double bp = (550.0 / lambda) * 0.30 * std::pow(std::max(C, 1e-6), 0.62);
        return PureSeaWaterScattering(lambda) + bp * std::max(0.0, turbidity);
    });
}

// Diffuse reflectance of a deep water column, from Morel & Prieur's R = 0.33 b_b / (a + b_b).
// Only backscattered light returns to the surface, and molecular scattering turns light around
// far more readily than the strongly forward-peaked particulate phase function does - which is
// why clear ocean water is blue rather than merely dark.
inline XMFLOAT3 WaterUpwellingRGB(double chlorophyll, double turbidity) {
    const double C = std::max(0.0, chlorophyll);
    return BandAverage([&](double lambda) {
        const double aw = PureWaterAbsorptionAt(lambda);
        const double a = (aw + 0.06 * std::pow(std::max(C, 1e-6), 0.65)) *
                         (1.0 + 0.02 * std::exp(-0.014 * (lambda - 380.0)));
        const double bw = PureSeaWaterScattering(lambda);
        const double bp = (550.0 / lambda) * 0.30 * std::pow(std::max(C, 1e-6), 0.62) * std::max(0.0, turbidity);
        // Backscatter fractions: one half for molecular scattering, about 1.8% for particulates.
        const double bb = 0.5 * bw + 0.018 * bp;
        return 0.33 * bb / std::max(a + bb, 1e-6);
    });
}

} // namespace ocean


