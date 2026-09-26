#pragma once

#include "../Common.h"
#include "../../shaders/OceanLayout.h"
#include <array>
#include <cmath>

namespace ocean {

constexpr double kGravity = 9.80665;

// Water coverage of a quadtree tile (world XZ). Partial splits it; when unsure, answer Partial.
enum class Coverage : uint8_t { None, Partial, Full };

struct ICoverage {
    virtual ~ICoverage() = default;
    virtual Coverage Test(double minX, double minZ, double size) const = 0;
};

struct Params {
    bool enabled = true;

    float windSpeed = 11.0f;        // m/s at 10 m
    float fetch = 200000.0f;        // m
    float windDirectionDeg = 30.0f; // clockwise from +Z
    float swell = 0.15f;            // 0 wind sea, 1 long-crested swell
    float windAlign = 0.8f;         // downwind lobe exponent
    // < 0: derived from wind and fetch
    float significantHeight = -1.0f;
    float peakPeriod = -1.0f;
    float swellHeight = 0.0f;
    float swellPeriod = 11.0f;
    float swellDirectionDeg = 80.0f;
    float swellSpreadDeg = 12.0f;
    bool paused = false;
    float fixedTimeStep = 0.0f; // s; 0 uses frame time
    float bodyWeight = 0.0f; // diffuse surface share
    uint32_t debugMode = 0; // OCEAN_DEBUG_*

    float turbulence = 1.0f; // short-wave chop and spread: 0 calm, 1 calibrated, 3 storm

    float turbulenceVariation = 0.35f; // km-scale sea-state gain variation
    float turbulencePeriod = 9000.0f;  // m

    float crestSharpening = 1.0f; // x physical second-order harmonic

    float choppiness = 1.0f;         // horizontal displacement gain
    float amplitudeScale = 1.0f;
    float shortWaveAmplitude = 1.5f; // gain on wind waves under 8 m, full under 2 m

    float chlorophyll = 0.06f; // mg/m^3
    float turbidity = 1.0f;    // scattering gain

    float foamCoverage = 1.0f; // whitecap cover gain; 0 = off
    float foamDecay = 0.75f;   // lace surviving per second (Monahan & Lu 1990)
    float foamAlbedo = 0.65f;  // excludes coverage

    // Water volume; radius scales the mean free path.
    float subsurfaceStrength = 1.0f;
    float subsurfaceRadiusScale = 1.0f;
    float subsurfacePhaseG = 0.9f; // Petzold 1972: ~0.92

    bool curvature = true; // Earth curvature

    uint32_t maxTiles = OCEAN_MAX_TILES; // read once at init

    float extent = 60000.0f;        // m, half-width
    float minTileSize = 8.0f;       // m
    float lodFactor = 0.5f;         // tile edge / camera distance
    float offscreenLodScale = 4.0f; // lodFactor gain outside the view
    float nearKeepRadius = 24.0f;   // m, full detail in every direction

    float filterScale = 0.0f; // wave normal filtering to the ray footprint; 0 = off

    float sunLobeRoughness = 0.1f; // roughness floor for sun/NEE sampling only

    float seaLevelY = 0.0f;

    uint32_t seed = 1337;
};

inline void ValidateParams(const Params& p) {
    const float values[] = {p.windSpeed,p.fetch,p.windDirectionDeg,p.swell,p.windAlign,p.significantHeight,
        p.peakPeriod,p.swellHeight,p.swellPeriod,p.swellDirectionDeg,p.swellSpreadDeg,p.fixedTimeStep,
        p.bodyWeight,p.turbulence,p.turbulenceVariation,p.turbulencePeriod,p.crestSharpening,p.choppiness,p.amplitudeScale,p.shortWaveAmplitude,p.chlorophyll,p.turbidity,p.foamCoverage,p.foamDecay,
        p.foamAlbedo,p.subsurfaceStrength,p.subsurfaceRadiusScale,p.subsurfacePhaseG,p.extent,p.minTileSize,p.lodFactor,
        p.offscreenLodScale,p.nearKeepRadius,
        p.filterScale,p.sunLobeRoughness,p.seaLevelY};
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
        p.extent < 64 || p.minTileSize < 1 || p.lodFactor <= 0 || p.offscreenLodScale < 1 || p.offscreenLodScale > 16 ||
        p.nearKeepRadius < 0 || p.filterScale < 0 || p.filterScale > 8 || p.debugMode >= OCEAN_DEBUG_COUNT)
        throw std::invalid_argument("Ocean parameter outside its supported range");
}
// m; non-power-of-two ratios so the sum tiles on their LCM.
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

constexpr double kReferenceWind = 11.0; // m/s, calibration sea state (Beaufort 6)

// Short-wave forcing relative to kReferenceWind.
inline double WindChop(const Params& p) {
    const double t = ((double)p.windSpeed - kReferenceWind) / kReferenceWind;
    return std::clamp(1.0 + 0.55 * t, 0.45, 1.5);
}

// Extra horizontal chop on the short waves only; above the cap the surface folds.
constexpr double kMaxShortChop = 2.5;
inline double ShortWaveChop(const Params& p) {
    return std::min(std::max(0.0, (double)p.turbulence) * WindChop(p), kMaxShortChop);
}

// Typical short-wave gain, for bounds.
inline double EffectiveChoppiness(const Params& p) {
    return (double)p.choppiness * (1.0 + ShortWaveChop(p));
}

// Short-wave chop band, log2 k: rises from 2x to 8x peak k, fades below half-metre waves.
struct ChopBand {
    double lo = 0.0, hi = 0.0, cut = 0.0;
};
inline ChopBand ShortChopBand(double peakOmega) {
    const double kp = std::max(peakOmega * peakOmega / kGravity, 1e-6);
    const double cut = std::log2(12.566370614359172); // half-metre waves
    ChopBand b;
    b.lo = std::min(std::log2(2.0 * kp), cut - 1.0);
    b.hi = std::max(std::min(std::log2(8.0 * kp), cut), b.lo + 0.25);
    b.cut = cut;
    return b;
}
// Must match OceanChopGain (OceanMath.hlsli).
inline double ChopGainAt(double k, const ChopBand& b, double extra) {
    auto smooth = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };
    const double l = std::log2(std::max(k, 1e-9));
    return 1.0 + extra * smooth((l - b.lo) / (b.hi - b.lo)) * (1.0 - smooth(l - b.cut));
}

inline double TurbulenceMaxGain(const Params& p) {
    return 1.0 + std::clamp((double)p.turbulenceVariation, 0.0, 0.9);
}

// Max a*sigma: keeps the turn point -1/(2a) beyond 4 sigma.
constexpr double kMaxSkewSteepness = 0.12;

// Crest sharpening coefficient; physically a = k (Tayfun 1980).
inline double CrestSkew(const Params& p, double kBar, double sigma, double chopAtK) {
    const double requested = std::max(0.0, (double)p.crestSharpening) * kBar * chopAtK;
    const double peak = sigma * TurbulenceMaxGain(p);
    const double limit = peak > 1e-6 ? kMaxSkewSteepness / peak : 0.0;
    return std::min(requested, limit);
}

// Jacobian a crest must be squeezed below before it may break.
constexpr double kBreakingJacobian = 0.9;

inline double BreakingThreshold(const Params& p) {
    return std::clamp(kBreakingJacobian + 0.25 * ((double)p.foamCoverage - 1.0), 0.02, 0.95);
}

// Whitecap fraction, U at 10 m (Monahan & O'Muircheartaigh 1980).
inline double WhitecapCoverage(double windSpeed10m) {
    const double U = std::max(0.0, windSpeed10m);
    return std::clamp(3.84e-6 * std::pow(U, 3.41), 0.0, 0.35);
}

// Target foam cover: 30% of the measured W(U), at most 3%.
constexpr double kWhitecapShown = 0.3;
constexpr double kWhitecapMax = 0.03;
inline double WhitecapCover(const Params& p) {
    return std::clamp(kWhitecapShown * WhitecapCoverage(p.windSpeed) * std::max(0.0, (double)p.foamCoverage), 0.0,
                      kWhitecapMax);
}

// Spreading exponent scale; 1 at turbulence 1.
inline double DirectionalFocus(const Params& p) {
    return std::exp(-0.47 * (std::max(0.0, (double)p.turbulence) - 1.0));
}

inline double ShortWaveAmplitude(const Params& p, double k) {
    constexpr double kStart = 6.283185307179586 / 8.0;
    constexpr double kFull = 6.283185307179586 / 2.0;
    double t = std::clamp((k - kStart) / (kFull - kStart), 0.0, 1.0);
    t = t * t * (3.0 - 2.0 * t);
    // Squared later; must not go negative.
    return std::max(0.0, 1.0 + (double(p.shortWaveAmplitude) - 1.0) * t);
}

inline double CascadeWindow(int c, double k) {
    const double lo1 = CascadeFundamental(c);
    const double lo2 = lo1 * 6.0;   // resolved from six modes up
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

// Power partition of unity; amplitudes take the square root.
inline double CascadeWeight(int c, double k) {
    double total = 0.0;
    for (int j = 0; j < OCEAN_CASCADES; ++j)
        total += CascadeWindow(j, k);
    if (total <= 1e-9)
        return 0.0;
    return CascadeWindow(c, k) / total;
}

// JONSWAP (Hasselmann et al. 1973), Donelan-Banner spreading (Donelan et al. 1985, Banner 1990).
struct Spectrum {
    double alpha = 0.0;
    double omegaP = 0.0;
    double gamma = 3.3;
    double swell = 0.0;
    double windAlign = 0.0;
    double focus = 1.0; // DirectionalFocus
    double energyScale = 1.0;
    double equilibriumGain = 1.0; // above-peak energy gain
    // Per-omega normalisation of D.
    static constexpr int kNormBins = 256;
    std::array<double, kNormBins> dirNorm{};
    double omegaNormMin = 0.02;
    double omegaNormMax = 40.0;

    void Init(const Params& p);
    // S(omega) only: no equilibrium gain or D table.
    void InitShape(const Params& p);

    double S(double omega) const; // frequency spectrum, m^2 s
    // 0 below 1.5x peak k, 1 from 3x.
    double EquilibriumShare(double omega) const;
    double D(double omega, double theta) const; // normalised directional spreading, 1/rad
    double S2D(double kx, double kz) const;     // two-sided wavenumber spectrum, m^4

  private:
    double DRaw(double omega, double theta) const;
    double DirNormAt(double omega) const;
};

// Dimensionless fetch cap: fully developed sea (Pierson & Moskowitz 1964).
constexpr double kFullyDevelopedFetch = 1.7e4;
// JONSWAP gamma from inverse wave age U/c_p (Donelan et al. 1985).
inline double PeakEnhancement(double windSpeed, double omegaP) {
    const double inverseAge = windSpeed * omegaP / kGravity;
    if (inverseAge <= 0.83)
        return 1.0;
    if (inverseAge < 1.0)
        return 1.0 + 0.7 * (inverseAge - 0.83) / 0.17;
    return std::min(1.7 + 6.0 * std::log10(inverseAge), 3.3);
}

inline void CoxMunkSlopeVariance(double windSpeed10m, double& varAlong, double& varCross);

// Slope variance the cascades resolve; `ramped` is the equilibrium-gain part.
inline void ResolvedSlopeMoments(const Params& p, const Spectrum& s, double& all, double& ramped) {
    const double w0 = std::sqrt(kGravity * CascadeFundamental(0));
    const double w1 = std::sqrt(kGravity * CascadeNyquist(OCEAN_CASCADES - 1));
    constexpr int steps = 2048;
    const double logStep = std::log(w1 / w0) / steps;
    all = ramped = 0.0;
    for (int i = 0; i < steps; ++i) {
        const double w = w0 * std::exp((i + 0.5) * logStep);
        const double k = w * w / kGravity;
        const double gain = ShortWaveAmplitude(p, k);
        const double m = k * k * s.S(w) * gain * gain * w * logStep;
        all += m;
        ramped += m * s.EquilibriumShare(w);
    }
}

// Equilibrium gain makes resolved slope follow Cox & Munk 1954 with wind.
inline void Spectrum::Init(const Params& p) {
    InitShape(p);
    Spectrum reference;
    Params rp = p;
    rp.windSpeed = (float)kReferenceWind;
    reference.InitShape(rp);
    double refAll = 0.0, refRamped = 0.0, all = 0.0, ramped = 0.0;
    ResolvedSlopeMoments(rp, reference, refAll, refRamped);
    ResolvedSlopeMoments(p, *this, all, ramped);
    auto coxMunk = [](double U) {
        double along = 0.0, cross = 0.0;
        CoxMunkSlopeVariance(U, along, cross);
        return along + cross;
    };
    const double target = refAll / coxMunk(kReferenceWind) * coxMunk(std::max(0.0, (double)p.windSpeed));
    equilibriumGain = ramped > 1e-12 ? std::clamp(1.0 + (target - all) / ramped, 0.2, 4.0) : 1.0;
    if (p.significantHeight >= 0.0f && equilibriumGain != 1.0) {
        double integral = 0.0;
        constexpr int steps = 8192;
        const double lo = omegaP * 0.05, hi = omegaP * 100.0;
        const double logStep = std::log(hi / lo) / steps;
        for (int i = 0; i < steps; ++i) {
            const double w = lo * std::exp((i + 0.5) * logStep);
            integral += S(w) * w * logStep;
        }
        energyScale *= double(p.significantHeight) * p.significantHeight / (16.0 * std::max(integral, 1e-30));
    }

    // Tabulated: too costly per grid point.
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

inline void Spectrum::InitShape(const Params& p) {
    const double U = std::max(0.1, (double)p.windSpeed);
    const double chi = std::min(std::max(1000.0, (double)p.fetch) * kGravity / (U * U), kFullyDevelopedFetch);
    const double F = chi * U * U / kGravity;
    alpha = 0.076 * std::pow(U * U / (F * kGravity), 0.22);
    omegaP = 22.0 * std::pow(kGravity * kGravity / (U * F), 1.0 / 3.0);
    swell = std::clamp((double)p.swell, 0.0, 1.0);
    focus = DirectionalFocus(p);
    windAlign =std::max(0.0, (double)p.windAlign) * focus;
    if (p.peakPeriod > 0.0f) omegaP = 6.283185307179586 / std::max(0.5, (double)p.peakPeriod);
    gamma = PeakEnhancement(U, omegaP);
    equilibriumGain = 1.0;
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
}

inline double Spectrum::EquilibriumShare(double omega) const {
    const double r = omega / std::max(omegaP, 1e-6);
    double t = std::clamp(std::log2(r * r / 1.5), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double Spectrum::S(double omega) const {
    if (omega <= 1e-4)
        return 0.0;
    const double sigma = (omega <= omegaP) ? 0.07 : 0.09;
    const double d = omega - omegaP;
    const double r = std::exp(-(d * d) / (2.0 * sigma * sigma * omegaP * omegaP));
    const double wp_w = omegaP / omega;
    const double body = alpha * kGravity * kGravity / std::pow(omega, 5.0);
    const double equilibrium = 1.0 + (equilibriumGain - 1.0) * EquilibriumShare(omega);
    return energyScale * equilibrium * body * std::exp(-1.25 * std::pow(wp_w, 4.0)) * std::pow(gamma, r);
}

inline double Spectrum::DRaw(double omega, double theta) const {
    // Donelan-Banner beta_s
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
    // Swell narrows, turbulence widens.
    beta *=(1.0 + 3.0 * swell) * focus;

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

// Sea-state predictions from S(omega) alone, usable before the bake.
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

// Swell: log-normal in omega, wrapped Gaussian in direction, scaled to Hm0.
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

// Deepest trough below the mean: 4.5 sigma, deepened by chop.
inline double PredictWaveDepth(const Params& p) {
    const double sigma = std::sqrt(std::max(0.0, PredictElevationVariance(p)));
    return 4.5 * sigma * (1.0 + 0.3 * EffectiveChoppiness(p));
}

inline double PredictSurfaceLevel(const Params& p) {
    return (double)p.seaLevelY;
}

// Material block slot -> foam level and bubble density; the debug slot has neither.
inline float MaterialSlotFoam(uint32_t slot) {
    return slot < OCEAN_MATERIAL_LEVELS ? float(slot % OCEAN_FOAM_STEPS) / float(OCEAN_FOAM_STEPS - 1) : 0.0f;
}
inline float MaterialSlotBubbles(uint32_t slot) {
    return slot < OCEAN_MATERIAL_LEVELS ? float(slot / OCEAN_FOAM_STEPS) / float(OCEAN_BUBBLE_STEPS - 1) : 0.0f;
}

// Breaker bubble cloud (Deane & Stokes 2002) as a dipole half-space (Jensen et al. 2001).
static constexpr double kBubbleScatter = 0.8; // 1/m, reduced scattering of a fresh cloud
inline void BubbleCloud(double density, const XMFLOAT3& absorption, XMFLOAT3& albedo, float& cover) {
    const double boundary = 2.8; // (1 + Fdr) / (1 - Fdr), water to air
    const double scatter = std::max(density, 0.0) * kBubbleScatter;
    const double a[3] = {absorption.x, absorption.y, absorption.z};
    float r[3];
    for (int c = 0; c < 3; ++c) {
        const double alpha = scatter / std::max(scatter + std::max(a[c], 0.0), 1e-9);
        const double e = std::sqrt(3.0 * (1.0 - alpha));
        r[c] = (float)(0.5 * alpha * (1.0 + std::exp(-4.0 / 3.0 * boundary * e)) * std::exp(-e));
    }
    albedo = {r[0], r[1], r[2]};
    cover = (float)(1.0 - std::exp(-0.8 * std::max(density, 0.0)));
}

// Single writer of the ocean material block. `waterAbsorption` is the water's Tf.
template <typename Enable>
inline void WriteMaterialSlot(uint32_t slot, const XMFLOAT4& waterKd, const XMFLOAT3& waterAbsorption,
                              float waterSssWeight, uint32_t waterSssEnable, float foamAlbedo, XMFLOAT4& kd,
                              float& sssWeight, Enable& sssEnable) {
    if (slot == OCEAN_MATERIAL_DEBUG) {
        kd = XMFLOAT4{waterKd.x, waterKd.y, waterKd.z, 1.0f};
        sssWeight = 0.0f;
        sssEnable = 0u;
        return;
    }
    const float foam = MaterialSlotFoam(slot);
    const float bubbles = MaterialSlotBubbles(slot);
    if (foam <= 0.0f && bubbles <= 0.0f) {
        kd = waterKd;
        sssWeight = waterSssWeight;
        sssEnable = (Enable)waterSssEnable;
        return;
    }
    const float albedo = std::clamp(foamAlbedo, 0.0f, 1.0f);
    XMFLOAT3 cloud;
    float caught;
    BubbleCloud(bubbles, waterAbsorption, cloud, caught);
    const float below = (1.0f - foam) * caught;
    const float opaque = foam + below;
    kd = XMFLOAT4{(foam * albedo + below * cloud.x) / opaque, (foam * albedo + below * cloud.y) / opaque,
                  (foam * albedo + below * cloud.z) / opaque, waterKd.w + (1.0f - waterKd.w) * opaque};
    sssWeight = waterSssWeight * (1.0f - opaque);
    sssEnable = opaque < 1.0f ? (Enable)waterSssEnable : (Enable)0u;
}

// Clean-sea slope variance (Cox & Munk 1954).
inline void CoxMunkSlopeVariance(double windSpeed10m, double& varAlong, double& varCross) {
    const double U = std::max(0.0, windSpeed10m) * 1.04; // U10 -> U12.5
    varAlong = 3.16e-3 * U;
    varCross = 0.003 + 1.92e-3 * U;
}

// Pure water absorption, 1/m, 400-700 nm in 20 nm steps (Pope & Fry 1997).
inline const std::array<double, 16>& PureWaterAbsorption() {
    static const std::array<double, 16> a = {0.00663, 0.00454, 0.00635, 0.00979, 0.0127, 0.0204,
                                             0.0409,  0.0474,  0.0619,  0.0896,  0.2224, 0.2755,
                                             0.3108,  0.4100,  0.4650,  0.6240};
    return a;
}

// Pure seawater scattering, 1/m (Morel 1974).
inline double PureSeaWaterScattering(double lambdaNm) {
    return 0.0029 * std::pow(500.0 / lambdaNm, 4.32);
}

// Mean over rectangular R, G, B bands.
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

// Morel Case-1 absorption.
inline XMFLOAT3 WaterAbsorptionRGB(double chlorophyll) {
    const double C = std::max(0.0, chlorophyll);
    return BandAverage([&](double lambda) {
        const double aw = PureWaterAbsorptionAt(lambda);
        const double pigment = 0.06 * std::pow(std::max(C, 1e-6), 0.65);
        return (aw + pigment) * (1.0 + 0.02 * std::exp(-0.014 * (lambda - 380.0)));
    });
}

// Case-1 particulate scattering (Morel 1988) plus pure seawater.
inline XMFLOAT3 WaterScatteringRGB(double chlorophyll, double turbidity) {
    const double C = std::max(0.0, chlorophyll);
    return BandAverage([&](double lambda) {
        const double bp = (550.0 / lambda) * 0.30 * std::pow(std::max(C, 1e-6), 0.62);
        return PureSeaWaterScattering(lambda) + bp * std::max(0.0, turbidity);
    });
}

} // namespace ocean


