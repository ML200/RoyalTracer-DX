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
    uint32_t debugMode = 0; // OCEAN_DEBUG_*: 0 beauty, 1 normal, 2 compression, 3 geometry LOD, 4 filter mip

    // How hard a gusty, veering wind whips the short waves riding the big ones. It pulls their
    // crests into points (extra horizontal compression and Stokes sharpening on the short waves
    // only, see ShortWaveChop) and scatters the wind sea off the wind's heading into a confused,
    // short-crested one (DirectionalFocus). Wave height and length stay the wind's, and so does how
    // much of the sea is white (WhitecapCover). 0 leaves every crest rounded and the waves marching
    // one way; 1 is the calibrated sea; 3 a hard, confused one. The wind's own forcing scales the
    // whipping (see WindChop).
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
    // long shallow trough, and higher values push each band towards its steepness limit. It never
    // moves the surface sideways, so it lifts a crest without folding it.
    float crestSharpening = 1.0f;

    // Horizontal displacement gain on every wave: 1 is the physical first-order (Lagrangian) sea,
    // 0 a purely vertical, rounded one. The short waves' extra comes on top (ShortWaveChop).
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

    // Whitecaps are simulated, not painted: foam gathers wherever the surface is squeezed past
    // breaking and fades after it, so it sits on the crests the waves point, and a rounded sea
    // carries none. As many crests break as it takes for the foam to cover a third of what is
    // measured at sea, and at most three per cent (WhitecapCover) - a few small caps at Beaufort 6,
    // a crest breaking here and there in a gale. This scales that cover and moves the breaking
    // point with it (BreakingThreshold): 1 is the calibrated sea, higher lets gentler crests break
    // too, 0 turns foam off.
    float foamCoverage = 1.0f;
    // Fraction of the lace a whitecap leaves surviving each second. The cap and the sheet of foam
    // it leaves fade five times as fast (OCEAN_FOAM_FRESH_DECAY): a whitecap's white area decays
    // within a second or two (Monahan & Lu 1990, stage B), the thin lace of monolayer foam after it
    // lingers a few seconds more.
    float foamDecay = 0.75f;
    // Local bubble-raft reflectance. Coverage is accounted for separately, so this must not
    // contain a second coverage average (which would turn whitecaps grey).
    float foamAlbedo = 0.65f;

    bool legacySubsurface = false; // retained API field; solid-object SSS is never used for water
    // Water volume controls. Radius scales the scattering mean free path in metres.
    float subsurfaceStrength = 1.0f;
    float subsurfaceRadiusScale = 1.0f;
    float subsurfacePhaseG = 0.9f; // ocean particles are strongly forward scattering (Petzold: ~0.92)

    // Follow the curve of the Earth. Without it the horizon sits at infinity and distant ships
    // never drop below it, which reads as wrong immediately in a wide ocean shot.
    bool curvature = true;

    // Tiles the quadtree may keep resident. Every tile is an acceleration structure refitted each
    // frame, an instance in the scene's top level, and a block of the shared vertex and index
    // buffers. Selection coarsens its detail until the sea fits, so this is a ceiling rather than
    // what the default sea uses (about 300 from deck height). Read once, when the buffers are sized.
    uint32_t maxTiles = OCEAN_MAX_TILES;

    // Half-width of the simulated ocean in metres. The quadtree root spans 2x this.
    float extent = 60000.0f;
    // Smallest tile edge in metres; sets the finest displaced geometry (tile / OCEAN_TILE_GRID).
    float minTileSize = 8.0f;
    // Tile edge length as a fraction of the distance to the camera, for tiles the camera can see.
    // Quads come out at this / OCEAN_TILE_GRID of their distance, 1.6% at the default: a couple of
    // dozen pixels, which is as fine as geometry needs to be when the normal carries the detail.
    // Lower is finer; the triangle count goes with the inverse square.
    float lodFactor = 0.5f;
    // The same ratio is multiplied by this for tiles outside the view. The whole sea stays in the
    // scene for reflections, shadows and refraction, but a secondary ray sees it through a
    // blurred, noisy estimate and never along a silhouette, so it can be built far coarser. 1
    // makes the detail independent of where the camera looks.
    float offscreenLodScale = 4.0f;
    // Tiles nearer than this keep the in-view detail wherever the camera looks, so reflections and
    // shadows of the waves right beside it stay as sharp as the waves in front of it.
    float nearKeepRadius = 24.0f;

    // How far a hit averages the wave normal over its ray footprint, turning the ripples it
    // averages away into roughness. 0, the default, samples every ripple at full resolution at
    // every distance and keeps the surface a mirror: each pixel sees one sharp facet, and the
    // denoiser's accumulation over many of them is what builds the distant sheen and glitter
    // path, with the real sparkle of the waves in it. Filtering to the footprint (1) is steadier
    // but reads as a rough, plastic-looking sea far away.
    float filterScale = 0.0f;

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
        p.foamAlbedo,p.subsurfaceStrength,p.subsurfaceRadiusScale,p.subsurfacePhaseG,p.extent,p.minTileSize,p.lodFactor,
        p.offscreenLodScale,p.nearKeepRadius,
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
        p.extent < 64 || p.minTileSize < 1 || p.lodFactor <= 0 || p.offscreenLodScale < 1 || p.offscreenLodScale > 16 ||
        p.nearKeepRadius < 0 || p.filterScale < 0 || p.filterScale > 8 || p.debugMode >= OCEAN_DEBUG_COUNT)
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

// How hard the wind forces the short waves, relative to the reference wind. A fully developed
// spectrum is equally steep at every wind speed - linear theory alone would break a gale as rarely
// as a breeze - but a stronger wind drives its short waves harder, and that is where the measured
// rise of whitecapping with wind comes from. Exactly one at the reference wind.
inline double WindChop(const Params& p) {
    const double t = ((double)p.windSpeed - kReferenceWind) / kReferenceWind;
    return std::clamp(1.0 + 0.55 * t, 0.45, 1.5);
}

// Extra horizontal displacement the short waves carry on top of the physical first-order one.
// Every wave keeps its own full displacement - a gain of one, the Lagrangian sea - so the long waves
// that carry the swell and the dominant sea move exactly as far sideways as they rise. The short
// waves riding them get this much more on top: it narrows their crests into the points the
// whitecaps break from. Turbulence sets it, scaled by the wind's forcing; past the cap the surface
// folds over in sheets rather than pointing. How much of the sea breaks stays the wind's
// (WhitecapCover).
constexpr double kMaxShortChop = 2.5;
inline double ShortWaveChop(const Params& p) {
    return std::min(std::max(0.0, (double)p.turbulence) * WindChop(p), kMaxShortChop);
}

// Horizontal gain a typical short wave gets, for bounds and readouts.
inline double EffectiveChoppiness(const Params& p) {
    return (double)p.choppiness * (1.0 + ShortWaveChop(p));
}

// Which waves are "short", as log2 wavenumbers: the extra chop rises from twice the spectral peak's
// wavenumber to its full value at eight times it - the waves riding the dominant ones rather than
// the dominant ones themselves - and falls away again over the octave below half-metre ripples,
// whose own steepness would otherwise crumple the surface at centimetre scale. The band follows the
// peak, so a gale's pointed waves are metres long where a breeze's are palm-sized.
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
// Horizontal gain of a wave of wavenumber k. Must match OceanChopGain in shaders/OceanMath.hlsli.
inline double ChopGainAt(double k, const ChopBand& b, double extra) {
    auto smooth = [](double t) {
        t = std::clamp(t, 0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    };
    const double l = std::log2(std::max(k, 1e-9));
    return 1.0 + extra * smooth((l - b.lo) / (b.hi - b.lo)) * (1.0 - smooth(l - b.cut));
}

// Largest gain the kilometre-scale sea-state field reaches. The crest sharpening's monotonic range
// is solved against the roughest patch rather than against the mean sea.
inline double TurbulenceMaxGain(const Params& p) {
    return 1.0 + std::clamp((double)p.turbulenceVariation, 0.0, 0.9);
}

// Largest skew coefficient a band may carry, as a steepness a*sigma. The warp turns around at
// -1/(2a), so this holds that point past four standard deviations of the band's own elevation:
// beyond it the deepest troughs would come back up as a second crest.
constexpr double kMaxSkewSteepness = 0.12;

// Second-order Stokes coefficient for a band of mean wavenumber kBar and elevation standard
// deviation sigma. A physical bound harmonic has a = k; the short bands take the same whipping as
// their horizontal chop (chopAtK), so turbulence peaks them vertically too. The short cascades are
// already near their steepness limit and clamp there.
inline double CrestSkew(const Params& p, double kBar, double sigma, double chopAtK) {
    const double requested = std::max(0.0, (double)p.crestSharpening) * kBar * chopAtK;
    const double peak = sigma * TurbulenceMaxGain(p);
    const double limit = peak > 1e-6 ? kMaxSkewSteepness / peak : 0.0;
    return std::min(requested, limit);
}

// A crest has to be squeezed to under this share of its rest area before it may break at all: it
// keeps foam off water that is not being compressed, and a sea with no horizontal displacement
// carries none. Which of the squeezed crests do break - the hardest-squeezed, as many as it takes
// for WhitecapCover - is the wind's, not this: set at 0.55, where a crest has plainly lost
// cohesion, it held back every sea whose crests turbulence had left rounded, so the foam still
// rose with turbulence. Measured on the composite surface filtered to a quarter metre, where the
// metre-scale breakers resolve.
constexpr double kBreakingJacobian = 0.9;

// That limit after the whitecap control.
inline double BreakingThreshold(const Params& p) {
    return std::clamp(kBreakingJacobian + 0.25 * ((double)p.foamCoverage - 1.0), 0.02, 0.95);
}

// Monahan & O'Muircheartaigh's whitecap coverage: the fraction of the sea surface white with
// whitecaps, the breaking crests and the foam they leave, W = 3.84e-6 U^3.41 with U in m/s at 10 m.
// At Beaufort 6 that is barely one per cent, which is far less foam than most ocean shaders draw -
// and the difference is a large part of why they read as stylised.
inline double WhitecapCoverage(double windSpeed10m) {
    const double U = std::max(0.0, windSpeed10m);
    return std::clamp(3.84e-6 * std::pow(U, 3.41), 0.0, 0.35);
}

// Share of the sea the whitecaps cover. The foam simulation breaks the crests squeezed hardest,
// and only those past BreakingThreshold, and it breaks as many of them as it takes for the foam
// they leave to be plainly white over this much (OceanFoamStats). Left to the surface alone, a
// breeze's few breakers left next to nothing and anything that pointed the crests harder turned the
// sea white. The whitecap control multiplies it.
//
// It follows Monahan's W(U), but at a third of it and never past three per cent. The census counts
// the caps and the sheet they leave; the lace this sea draws after them is foam too, and at the full
// measured cover it doubled what the eye takes in - a gale's worth of white at Beaufort 7, most of
// it bright. However hard it blows or the control is set, the sea stays a sea with foam on it.
constexpr double kWhitecapShown = 0.3;
constexpr double kWhitecapMax = 0.03;
inline double WhitecapCover(const Params& p) {
    return std::clamp(kWhitecapShown * WhitecapCoverage(p.windSpeed) * std::max(0.0, (double)p.foamCoverage), 0.0,
                      kWhitecapMax);
}

// How much turbulence scatters the waves off the wind's heading, as a factor on the directional
// spreading exponent. A steady wind raises long crests marching one way; a gusty, veering one
// leaves a confused, short-crested sea whose waves run in every direction. Exactly one at the
// calibrated turbulence, so the measured Donelan-Banner spread is what the default sea gets; a
// storm at 3 spreads the wind sea about 2.5 times as wide.
inline double DirectionalFocus(const Params& p) {
    return std::exp(-0.47 * (std::max(0.0, (double)p.turbulence) - 1.0));
}

inline double ShortWaveAmplitude(const Params& p, double k) {
    constexpr double kStart = 6.283185307179586 / 8.0;
    constexpr double kFull = 6.283185307179586 / 2.0;
    double t = std::clamp((k - kStart) / (kFull - kStart), 0.0, 1.0);
    t = t * t * (3.0 - 2.0 * t);
    // The gain is squared into a power below, so it must not be allowed to swing negative - that
    // would turn a suppressed band back into a boosted one. Turbulence leaves it alone: short wind
    // waves sit in the spectrum's saturation range, where a harder wind makes them break rather than
    // grow, so a whipped sea gets its look from pointed crests and whitecaps, not taller ripples.
    return std::max(0.0, 1.0 + (double(p.shortWaveAmplitude) - 1.0) * t);
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
    double focus = 1.0; // DirectionalFocus
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

// Dimensionless fetch gF/U^2 past which the sea stops growing. The JONSWAP fetch laws keep
// lengthening the waves for as long as the fetch does, but a real sea saturates: once it has
// travelled far enough for the wind, it is fully developed (Pierson-Moskowitz) and its peak sits at
// omega_p U/g ~ 0.855 whatever the fetch. That is where 22 chi^(-1/3) reaches it. Without the cap
// a light breeze over an open-ocean fetch raised the long, slow waves of a gale, which is why the
// wind barely changed the wave period.
constexpr double kFullyDevelopedFetch = 1.7e4;
// Peak enhancement from the inverse wave age U/c_p (Donelan, Hamilton & Hui 1985): a young sea,
// its waves still slower than the wind, is sharply peaked; as they catch up the peak broadens to
// the Pierson-Moskowitz shape. Taken from the peak actually used, so an explicit period gets the
// shape that period has in this wind.
inline double PeakEnhancement(double windSpeed, double omegaP) {
    const double inverseAge = windSpeed * omegaP / kGravity;
    if (inverseAge <= 0.83)
        return 1.0;
    if (inverseAge < 1.0)
        return 1.0 + 0.7 * (inverseAge - 0.83) / 0.17;
    return std::min(1.7 + 6.0 * std::log10(inverseAge), 3.3);
}

inline void Spectrum::Init(const Params& p) {
    const double U = std::max(0.1, (double)p.windSpeed);
    const double chi = std::min(std::max(1000.0, (double)p.fetch) * kGravity / (U * U), kFullyDevelopedFetch);
    const double F = chi * U * U / kGravity;
    alpha = 0.076 * std::pow(U * U / (F * kGravity), 0.22);
    omegaP = 22.0 * std::pow(kGravity * kGravity / (U * F), 1.0 / 3.0);
    swell = std::clamp((double)p.swell, 0.0, 1.0);
    focus = DirectionalFocus(p);
    // A confused sea also sends more of its waves back against the wind.
    windAlign = std::max(0.0, (double)p.windAlign) * focus;
    if (p.peakPeriod > 0.0f) omegaP = 6.283185307179586 / std::max(0.5, (double)p.peakPeriod);
    gamma = PeakEnhancement(U, omegaP);
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
    // Turbulence widens it.
    beta *= (1.0 + 3.0 * swell) * focus;

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

// Foam coverage and bubble-cloud density of each slot in the ocean's material block:
// OCEAN_FOAM_STEPS steps from clear water to solid foam, repeated for each of OCEAN_BUBBLE_STEPS
// densities of the cloud under it. The diagnostic slot after them carries neither.
inline float MaterialSlotFoam(uint32_t slot) {
    return slot < OCEAN_MATERIAL_LEVELS ? float(slot % OCEAN_FOAM_STEPS) / float(OCEAN_FOAM_STEPS - 1) : 0.0f;
}
inline float MaterialSlotBubbles(uint32_t slot) {
    return slot < OCEAN_MATERIAL_LEVELS ? float(slot / OCEAN_FOAM_STEPS) / float(OCEAN_BUBBLE_STEPS - 1) : 0.0f;
}

// The bubble cloud a breaker drives under its cap: void fractions of a per cent or more in the top
// half metre, clearing within seconds as the bubbles rise (Deane & Stokes 2002). It is a strong
// scatterer in absorbing water, so what it sends back up is the diffuse reflectance of a scattering
// half-space - Jensen et al.'s (2001) dipole, behind sea water's refractive boundary - and the light
// diffusing through it is tinted by the water it travels, red lost first: the pale turquoise under
// a breaking crest, deepening as the cloud thins out. `absorption` is the water's, per metre;
// `cover` is the share of the light the surface lets through that the cloud catches before the
// water below would. A density of 1 is a fresh cloud.
static constexpr double kBubbleScatter = 0.8; // reduced scattering coefficient of a fresh cloud, 1/m
inline void BubbleCloud(double density, const XMFLOAT3& absorption, XMFLOAT3& albedo, float& cover) {
    const double boundary = 2.8; // (1 + Fdr) / (1 - Fdr) for light inside water meeting air
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

// Writes one slot of the ocean's material block from the water material. Foam replaces
// transmission with a diffuse raft of `foamAlbedo` on the surface; the bubble cloud under it
// replaces the part of the rest that it catches with its own diffuse return, beneath the water's
// own clear, mirror-smooth surface. What neither catches still enters the sea, and both take the
// volume scattering with them. The diagnostic slot is opaque. `waterAbsorption` is the water
// material's Tf. Every place that generates or edits the block goes through here.
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


