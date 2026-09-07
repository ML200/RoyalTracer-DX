#ifndef SHARC_GUIDE_V8_HLSLI
#define SHARC_GUIDE_V8_HLSLI
#include "Sharc_v8.hlsli"

// Cache-driven path guiding for the diffuse lobe, built on the radiance cache
// that docs/SHARC.md describes.
//
// Receiver entries are keyed by a coarse cell (the cache level plus
// GUIDE_LEVEL_OFFSET) and the dominant signed axis of the geometric normal.
// Each holds up to GUIDE_SLOTS bright patches that training paths have seen
// from that cell, taken from the radiance cache at the hit, plus a running
// mean of the irradiance the BSDF-sampled continuation collects there. A
// rendered or trained vertex builds a set of bounding cones toward those
// patches, weighted by their estimated irradiance contribution at the exact
// shading point, and replaces a fraction q of its cosine samples by cone
// samples. q is the capped fraction of the cell mean irradiance that the
// patches explain, so it collapses where no patch dominates. Only the pdf of
// the diffuse lobe changes; every consumer of a strategy pdf (scatter weight,
// NEE partner, emitter and sun MIS) uses the same mixture, so the estimator
// stays unbiased. Sampling and pdf evaluation decode the SAME packed cones,
// so the snapshot a lane took is self-consistent even while training lanes
// write.
//
// Receiver keys are drawn stochastically with the tent / LOD / normal-axis
// reconstruction weights (GuideKeyOf), so neighbouring receivers blend in
// expectation and cell edges leave no seam in the noise. Each patch also
// carries a learned visibility: guided training rays report whether they
// reached their patch, so a patch occluded from the cell loses its weight.
//
// Storage lives in the cache allocation behind the cache entries
// (SharcLayout.h). Rendering only reads it; the sparse update pass creates
// entries, registers patches, adds irradiance observations and visibility
// reports, and the prepare pass ages, windows and evicts.

static const uint GUIDE_INVALID = 0xffffffffu;
static const uint GUIDE_MIN_SAMPLES = 8u;    // irradiance observations before q can be positive
static const uint GUIDE_WINDOW = 1024u;      // observations kept in the irradiance mean
static const uint GUIDE_STREAM = 0x47554944u; // per-bounce RNG stream id
static const float GUIDE_PI = 3.14159265358979f;
static const float GUIDE_RANK_MIN_COS = 0.1f; // lenient receiver cosine when ranking at the cell centre

// Entry layout (byte offsets). Key [0,16), header [16,32), slots [32,160).
static const uint GUIDE_CELL = 0u;         // int3 receiver cell at the guide level
static const uint GUIDE_META = 12u;        // level | normal bin << 5 | worth flag (prepare pass)
static const uint GUIDE_META_KEY_MASK = 0xffu; // the part of meta that identifies the receiver
// Set by the prepare pass when the cell centre would guide with q above the
// floor: rendering skips the slot block of every other receiver.
static const uint GUIDE_META_WORTH = 0x100u;
// Below this guided fraction the cone set is not worth building (5% of the
// diffuse samples cannot change the noise much, the evaluation costs the same).
static const float GUIDE_Q_FLOOR = 0.05f;
static const uint GUIDE_IRRADIANCE = 16u;  // uint64, 2^-24 units: sum of BSDF-sampled irradiance observations
static const uint GUIDE_SAMPLES = 24u;     // observation count
static const uint GUIDE_LAST_TOUCH = 28u;  // frame of the last training visit
static const uint GUIDE_SLOT0 = 32u;       // GUIDE_SLOTS x 16 bytes
// Slot words: x = f16 offset.x | f16 offset.y << 16   (patch centre relative to the receiver cell centre)
//             y = f16 offset.z | level << 16 | valid << 31
//             z = f16 luminance | oct16 normal << 16
//             w = lastSeen & 0xffff | hits << 16 | visibility << 24
// visibility: 0..255 running estimate of the fraction of guided rays from this
// receiver that reach the patch; a patch occluded from the cell fades out.
static const uint GUIDE_SLOT_VALID = 0x80000000u;
// Normal bins 0..5 are receivers; bin 6 marks a TARGET patch for the debug
// view (only written while that view is active).
static const uint GUIDE_BIN_TARGET = 6u;

#define GUIDE_ENABLED ((guide_params & GUIDE_PARAM_ENABLED) != 0u)
#define GUIDE_QMAX ((float)((guide_params >> GUIDE_PARAM_QMAX_SHIFT) & 255u) * (1.0f / 255.0f))
#define GUIDE_LEVEL_OFFSET ((guide_params >> GUIDE_PARAM_LEVEL_SHIFT) & 7u)
#define GUIDE_LIFETIME ((float)(((guide_params >> GUIDE_PARAM_LIFETIME_SHIFT) & 255u) + 1u) * 8.0f)
#define GUIDE_TRAIN ((guide_params & GUIDE_PARAM_TRAIN) != 0u)
#define GUIDE_RADIUS ((float)((guide_params >> GUIDE_PARAM_RADIUS_SHIFT) & 127u) * (1.0f / 32.0f))
// Deepest path vertex (1 = primary) that looks up, samples and trains the
// guide. The first vertices carry the image variance; deeper ones mostly end
// in the cache, and the sparse training dispatch is bound by its longest
// lanes, so guide work there would cost far more than it returns.
#define GUIDE_MAX_DEPTH ((int)((guide_params >> GUIDE_PARAM_DEPTH_SHIFT) & 7u))

uint GuideStateAddress(uint slot) { return SHARC_CACHE_BYTES + slot * 4u; }
uint GuideEntryAddress(uint slot) { return SHARC_CACHE_BYTES + GUIDE_STATE_BYTES + slot * GUIDE_ENTRY_BYTES; }
uint GuideBucketOf(uint hash) { return hash & (GUIDE_CAPACITY / SHARC_BUCKET_SIZE - 1u); }
uint GuideSlotAddress(uint e, uint slot) { return e + GUIDE_SLOT0 + slot * 16u; }

uint GuideLevel(float3 position)
{
    return min((uint)SharcLevel(position) + GUIDE_LEVEL_OFFSET, SHARC_MAX_LEVEL - 1u);
}

struct GuideKey
{
    int3 cell;
    uint meta;
};

uint GuideNormalBin(float3 n)
{
    float3 a = abs(n);
    uint axis = (a.x >= a.y && a.x >= a.z) ? 0u : (a.y >= a.z ? 1u : 2u);
    return axis * 2u + (n[axis] < 0.0f ? 1u : 0u);
}

float3 GuideBinAxis(uint bin)
{
    float3 axis = 0.0f;
    axis[bin >> 1u] = (bin & 1u) != 0u ? -1.0f : 1.0f;
    return axis;
}

// Deterministic key of the cell containing the position (inspector, tests).
GuideKey GuideKeyCentre(float3 position, float3 geometricNormal)
{
    GuideKey k;
    uint level = GuideLevel(position);
    int3 base; float3 f;
    SharcGrid(position, level, base, f);
    k.cell = base;
    k.meta = level | (GuideNormalBin(geometricNormal) << 5u);
    return k;
}

// Stochastic receiver key for a path vertex: LOD, cell and signed normal axis
// are drawn with the tent / n^4 reconstruction weights, exactly as the cache
// query does. A vertex near a cell edge therefore blends the neighbouring
// receivers in expectation, so q and the patch set vary smoothly and cell
// boundaries leave no seam in the noise. One key serves the lookup and the
// training writes of the vertex.
GuideKey GuideKeyOf(float3 position, float3 geometricNormal, inout uint seed)
{
    GuideKey k;
    float lod = SharcLevel(position);
    uint level = min((uint)lod + (RandomFloatSingle(seed) < frac(lod) ? 1u : 0u) + GUIDE_LEVEL_OFFSET,
        SHARC_MAX_LEVEL - 1u);
    int3 base; float3 f;
    SharcGrid(position, level, base, f);
    float3 c = f - 0.5f; // offset from the cell centre in cells, [-0.5, 0.5)
    float3 u = float3(RandomFloatSingle(seed), RandomFloatSingle(seed), RandomFloatSingle(seed));
    k.cell = base + int3(sign(c)) * int3(u < abs(c));
    float3 w = SharcNormalWeights(geometricNormal);
    float r = RandomFloatSingle(seed);
    uint axis = r < w.x ? 0u : (r < w.x + w.y ? 1u : 2u);
    k.meta = level | ((axis * 2u + (geometricNormal[axis] < 0.0f ? 1u : 0u)) << 5u);
    return k;
}

// Key of the coarse patch around a surface point, as the debug view marks it.
GuideKey GuideTargetKey(float3 position)
{
    GuideKey k;
    uint level = GuideLevel(position);
    int3 base; float3 f;
    SharcGrid(position, level, base, f);
    k.cell = base;
    k.meta = level | (GUIDE_BIN_TARGET << 5u);
    return k;
}

uint GuideHash(GuideKey k)
{
    uint h = Hash32(asuint(k.cell.x) ^ 0x2545f491u);
    h = Hash32(h ^ asuint(k.cell.y));
    h = Hash32(h ^ asuint(k.cell.z));
    h = Hash32(h ^ k.meta ^ 0x9e3779b9u);
    return 1u + h % 0xfffffffeu;
}

float3 GuideCellCenter(GuideKey k)
{
    uint level = k.meta & 31u;
    return SharcNodeLocal(k.cell, level) + 0.5f * SharcCellSize(level);
}

GuideKey GuideLoadKey(uint e)
{
    uint4 a = g_sharc.Load4(e + GUIDE_CELL);
    GuideKey k;
    k.cell = asint(a.xyz);
    k.meta = a.w;
    return k;
}

bool GuideKeyMatches(GuideKey a, GuideKey b)
{
    return all(a.cell == b.cell) && ((a.meta ^ b.meta) & GUIDE_META_KEY_MASK) == 0u;
}

// Guiding only pays off toward patches brighter than the mean radiance the
// cosine sampler already sees (irradiance / pi): 0 at or below that mean, 1
// from four times it. Uniformly lit surroundings therefore keep q at zero.
float GuideContrast(float luminance, float irradiance)
{
    return saturate(0.5f * log2(max(luminance, 1e-8f) * GUIDE_PI / max(irradiance, 1e-8f)));
}

void GuideLoadBucket(uint bucket, out uint states[SHARC_BUCKET_SIZE])
{
    uint base = GuideStateAddress(bucket * SHARC_BUCKET_SIZE);
    [unroll] for (uint q = 0u; q < SHARC_BUCKET_SIZE / 4u; ++q)
    {
        uint4 v = g_sharc.Load4(base + q * 16u);
        states[q * 4u + 0u] = v.x;
        states[q * 4u + 1u] = v.y;
        states[q * 4u + 2u] = v.z;
        states[q * 4u + 3u] = v.w;
    }
}

// Read-only lookup: the entry address of the receiver cell, or GUIDE_INVALID.
uint GuideFind(GuideKey k)
{
    uint hash = GuideHash(k);
    uint bucket = GuideBucketOf(hash);
    uint states[SHARC_BUCKET_SIZE];
    GuideLoadBucket(bucket, states);
#if SHARC_COMPACT_QUERY
    // Same compaction as SharcQueryNode: one key-load/compare body instead of
    // sixteen, visited lowest slot first so the same entry is returned.
    uint matchingSlots = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        if (states[p] == hash) matchingSlots |= 1u << p;
    [loop] while (matchingSlots != 0u)
    {
        const uint p = (uint)firstbitlow(matchingSlots);
        matchingSlots &= matchingSlots - 1u;
        uint e = GuideEntryAddress(bucket * SHARC_BUCKET_SIZE + p);
        if (GuideKeyMatches(GuideLoadKey(e), k)) return e;
    }
#else
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
    {
        if (states[p] != hash) continue;
        uint e = GuideEntryAddress(bucket * SHARC_BUCKET_SIZE + p);
        if (GuideKeyMatches(GuideLoadKey(e), k)) return e;
    }
#endif
    return GUIDE_INVALID;
}

//------------------------------------------------------------------
// Candidate slots
//------------------------------------------------------------------
struct GuideSlot
{
    float3 offset;
    uint level;
    float luminance;
    float3 normal;
    uint lastSeen;
    uint hits;
    uint visibility; // 0..255
    bool valid;
};

// Move a 0..255 estimate a quarter of the way toward 255 or toward 0.
uint GuideVisibilityUp(uint v) { return min(v + (255u - v + 3u) / 4u, 255u); }
uint GuideVisibilityDown(uint v) { return v - (v + 3u) / 4u; }

uint GuidePackOct16(float3 n)
{
    if (!(dot(n, n) > 1e-6f)) return 0u;
    n = normalize(n);
    float2 p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0f) p = (1.0f - abs(p.yx)) * signNotZero(p);
    uint2 q = uint2(round(saturate(p * 0.5f + 0.5f) * 255.0f));
    return q.x | (q.y << 8u);
}

float3 GuideUnpackOct16(uint bits)
{
    float2 f = float2(bits & 255u, (bits >> 8u) & 255u) * (2.0f / 255.0f) - 1.0f;
    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.xy += -t * signNotZero(n.xy);
    return normalize(n);
}

GuideSlot GuideDecodeSlot(uint4 w)
{
    GuideSlot s;
    s.valid = (w.y & GUIDE_SLOT_VALID) != 0u;
    s.offset = float3(f16tof32(w.x & 0xffffu), f16tof32(w.x >> 16u), f16tof32(w.y & 0xffffu));
    s.level = (w.y >> 16u) & 31u;
    s.luminance = f16tof32(w.z & 0xffffu);
    s.normal = GuideUnpackOct16(w.z >> 16u);
    s.lastSeen = w.w & 0xffffu;
    s.hits = (w.w >> 16u) & 255u;
    s.visibility = w.w >> 24u;
    return s;
}

uint4 GuideEncodeSlot(GuideSlot s)
{
    uint4 w;
    w.x = f32tof16(s.offset.x) | (f32tof16(s.offset.y) << 16u);
    w.y = f32tof16(s.offset.z) | ((s.level & 31u) << 16u) | (s.valid ? GUIDE_SLOT_VALID : 0u);
    w.z = f32tof16(max(s.luminance, 0.0f)) | (GuidePackOct16(s.normal) << 16u);
    w.w = (s.lastSeen & 0xffffu) | (min(s.hits, 255u) << 16u) | (min(s.visibility, 255u) << 24u);
    return w;
}

// Bounding cone of a patch seen from x, with its estimated irradiance
// contribution: luminance x patch area x receiver cosine x patch cosine /
// distance^2, scaled by the learned visibility and faded by the time since
// the patch was last seen. Weight 0 means the patch is unusable from x:
// invalid, behind the receiver (below minCos) or enclosing it.
struct GuideCone
{
    float3 axis;
    float oneMinusCos;
    float weight;
};

GuideCone GuideConeOf(GuideSlot s, float3 receiverCenter, float3 x, float3 n, float minCos)
{
    GuideCone c;
    c.axis = 0.0f;
    c.oneMinusCos = 0.0f;
    c.weight = 0.0f;
    if (!s.valid) return c;
    float r = GUIDE_RADIUS * SharcCellSize(s.level);
    float3 d = receiverCenter + s.offset - x;
    float dist2 = dot(d, d);
    if (!(dist2 > r * r)) return c; // also rejects non-finite offsets
    float dist = sqrt(dist2);
    float3 a = d / dist;
    float cosX = dot(n, a);
    if (!(cosX > 0.0f)) return c;
    cosX = max(cosX, minCos);
    float cosC = max(-dot(s.normal, a), 0.1f);
    float age = (float)((sharc_frame - s.lastSeen) & 0xffffu);
    float decay = exp2(-age / GUIDE_LIFETIME);
    float sinA = r / dist;
    c.axis = a;
    c.oneMinusCos = max(1.0f - sqrt(max(1.0f - sinA * sinA, 0.0f)), 1e-4f);
    c.weight = s.luminance * GUIDE_PI * r * r * cosX * cosC / dist2 * decay * ((float)s.visibility / 255.0f);
    if (!isfinite(c.weight) || c.weight <= 0.0f) c.weight = 0.0f;
    return c;
}

//------------------------------------------------------------------
// Per-vertex cone set: packed so it stays in registers across NEE, and so
// sampling and pdf evaluation see identical cones.
//------------------------------------------------------------------
struct GuideSet
{
    uint2 cone[GUIDE_SLOTS]; // x: PackNormal(axis); y: f16 (1 - cos alpha) | f16 weight << 16
    uint2 bound;             // x: PackNormal(axis) of a cone enclosing every cone; y: f16 cos of its aperture
    float weightSum;         // sum of the DECODED f16 weights
    float q;                 // fraction of diffuse-lobe samples drawn from the cones; 0 = inactive
};

GuideSet GuideEmpty()
{
    GuideSet g;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c) g.cone[c] = uint2(0u, 0u);
    g.bound = uint2(0u, f32tof16(-1.0f));
    g.weightSum = 0.0f;
    g.q = 0.0f;
    return g;
}

// ignoreWorth: the prepare pass evaluates every receiver to set the flag.
GuideSet GuideBuild(uint e, float3 x, float3 n, bool ignoreWorth = false)
{
    GuideSet g = GuideEmpty();
    if (e == GUIDE_INVALID || !(GUIDE_QMAX > 0.0f)) return g;
    // Key and header first (one cache line): receivers the prepare pass did
    // not flag, or with too few observations, never touch the slot block.
    GuideKey k = GuideLoadKey(e);
    if (!ignoreWorth && (k.meta & GUIDE_META_WORTH) == 0u) return g;
    uint4 header = g_sharc.Load4(e + GUIDE_IRRADIANCE); // sum lo, sum hi, samples, lastTouch
    if (header.z < GUIDE_MIN_SAMPLES) return g;
    float irradiance = SharcFloat(header.xy, rcp(SHARC_RADIANCE_SCALE)) / (float)header.z;
    float3 center = GuideCellCenter(k);
    GuideCone cones[GUIDE_SLOTS];
    float total = 0.0f, largest = 0.0f;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
    {
        GuideSlot s = GuideDecodeSlot(g_sharc.Load4(GuideSlotAddress(e, c)));
        cones[c] = GuideConeOf(s, center, x, n, 0.0f);
        cones[c].weight *= GuideContrast(s.luminance, irradiance);
        total += cones[c].weight;
        largest = max(largest, cones[c].weight);
    }
    if (!(total > 0.0f)) return g;
    float q = GUIDE_QMAX * saturate(total / max(irradiance, 1e-8f));
    if (!(q > GUIDE_Q_FLOOR)) return g;
    float scale = rcp(largest);
    float3 axisSum = 0.0f;
    [unroll] for (uint i = 0u; i < GUIDE_SLOTS; ++i)
    {
        uint packedWeight = f32tof16(cones[i].weight * scale);
        g.cone[i] = uint2(PackNormal(cones[i].axis), f32tof16(cones[i].oneMinusCos) | (packedWeight << 16u));
        g.weightSum += f16tof32(packedWeight);
        axisSum += cones[i].axis * cones[i].weight;
    }
    // Enclosing cone, rounded OUTWARD so the packed cones stay inside it: a
    // direction outside it is outside every cone and GuidePdf returns at once.
    float cosBound = -1.0f;
    float3 axisBound = 0.0f;
    if (dot(axisSum, axisSum) > 1e-12f)
    {
        axisBound = normalize(axisSum);
        cosBound = 1.0f;
        [unroll] for (uint b = 0u; b < GUIDE_SLOTS; ++b)
        {
            if (!(cones[b].weight > 0.0f)) continue;
            float cosT = clamp(dot(cones[b].axis, axisBound), -1.0f, 1.0f);
            float sinT = sqrt(1.0f - cosT * cosT);
            float cosA = 1.0f - cones[b].oneMinusCos;
            float sinA = sqrt(max(1.0f - cosA * cosA, 0.0f));
            cosBound = min(cosBound, cosT * cosA - sinT * sinA); // cos(theta + alpha)
        }
        cosBound = cosBound > 0.0f ? cosBound - 4e-3f : -1.0f;
    }
    g.bound = uint2(PackNormal(axisBound), f32tof16(cosBound));
    g.q = g.weightSum > 0.0f ? q : 0.0f;
    return g;
}

void GuideBasis(float3 n, out float3 t, out float3 b)
{
    float sg = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sg + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0f + sg * n.x * n.x * a, sg * c, -sg * n.x);
    b = float3(c, sg + n.y * n.y * a, -n.y);
}

// Uniform direction in the cone: pdf 1 / (2 pi (1 - cos alpha)).
float3 GuideConeDirection(float3 axis, float oneMinusCos, float2 u)
{
    float z = 1.0f - u.y * oneMinusCos;
    float s = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * GUIDE_PI * u.x;
    float3 t, b;
    GuideBasis(axis, t, b);
    return normalize(t * (cos(phi) * s) + b * (sin(phi) * s) + axis * z);
}

// pick receives the slot index of the cone the direction was drawn from, so a
// training path can report whether the ray reached that patch.
float3 GuideSample(GuideSet g, inout uint seed, out uint pick)
{
    float u = RandomFloatSingle(seed) * g.weightSum;
    pick = 0u;
    uint last = 0u;
    float acc = 0.0f;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
    {
        float w = f16tof32(g.cone[c].y >> 16u);
        acc += w;
        if (w > 0.0f) last = c;
        if (u >= acc) pick = c + 1u;
    }
    pick = min(pick, last);
    float3 axis = 0.0f;
    float oneMinusCos = 0.0f;
    [unroll] for (uint i = 0u; i < GUIDE_SLOTS; ++i)
    {
        if (i != pick) continue;
        axis = UnpackNormal(g.cone[i].x);
        oneMinusCos = f16tof32(g.cone[i].y & 0xffffu);
    }
    float2 u2 = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
    return GuideConeDirection(axis, oneMinusCos, u2);
}

float GuidePdf(GuideSet g, float3 dir)
{
    if (dot(dir, UnpackNormal(g.bound.x)) < f16tof32(g.bound.y)) return 0.0f;
    float sum = 0.0f;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
    {
        float w = f16tof32(g.cone[c].y >> 16u);
        if (!(w > 0.0f)) continue;
        float oneMinusCos = f16tof32(g.cone[c].y & 0xffffu);
        float3 axis = UnpackNormal(g.cone[c].x);
        // Small tolerance: a direction generated at the cone rim must still
        // count as inside after the basis rotation rounding.
        if (dot(dir, axis) >= 1.0f - oneMinusCos - 1e-6f)
            sum += w / (2.0f * GUIDE_PI * oneMinusCos);
    }
    return sum / g.weightSum;
}

// The cosine pdf the guided share replaces (mirrors BRDF_PDF_Lambertian with
// n_s == n_g, as Pass_pt calls it).
float GuideLambertPdf(float3 n, float3 dir)
{
    if (dot(dir, n) <= 0.0f) return 0.0f;
    return max(dot(n, dir), 0.0f) / GUIDE_PI;
}

// Strategy pdf of a direction with the guided diffuse share:
//   p(w) = p_bsdf(w) + Pdiff * q * (p_guide(w) - p_cos(w)).
float GuideMixPdf(GuideSet g, float pdiff, float3 n, float3 dir, float bsdfPdf)
{
    if (!(g.q > 0.0f)) return bsdfPdf;
    return max(bsdfPdf + pdiff * g.q * (GuidePdf(g, dir) - GuideLambertPdf(n, dir)), 0.0f);
}

//------------------------------------------------------------------
// Training side
//------------------------------------------------------------------
#if SHARC_UPDATE_PASS
void GuideInitializeEntry(uint e, GuideKey k)
{
    g_sharc.Store4(e + GUIDE_CELL, uint4(asuint(k.cell), k.meta));
    [unroll] for (uint b = 16u; b < GUIDE_ENTRY_BYTES; b += 16u)
        g_sharc.Store4(e + b, 0u);
    g_sharc.Store(e + GUIDE_LAST_TOUCH, sharc_frame);
}

// Entry address of the receiver cell, creating one if needed. Full buckets
// replace their least recently visited entry after SHARC_REPLACE_AGE idle
// frames; otherwise this visit is a miss and retries later.
uint GuideFindOrInsert(GuideKey k)
{
    uint hash = GuideHash(k);
    uint bucket = GuideBucketOf(hash);
    uint states[SHARC_BUCKET_SIZE];
    GuideLoadBucket(bucket, states);
    uint slot = GUIDE_INVALID;
    bool contended = false;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
    {
        uint candidate = bucket * SHARC_BUCKET_SIZE + p;
        if (states[p] == hash)
        {
            uint e = GuideEntryAddress(candidate);
            DeviceMemoryBarrier(); // acquire the immutable key from its publisher
            if (GuideKeyMatches(GuideLoadKey(e), k))
            {
                g_sharc.Store(e + GUIDE_LAST_TOUCH, sharc_frame);
                return e;
            }
        }
        else if (states[p] == SHARC_LOCKED) contended = true;
        else if (states[p] == 0u && slot == GUIDE_INVALID) slot = candidate;
    }
    // A locked slot may be another lane publishing this very key. Claiming a
    // second slot would duplicate the receiver, so miss and retry next visit.
    if (contended) return GUIDE_INVALID;
    uint expected = 0u;
    if (slot == GUIDE_INVALID)
    {
        uint victimAge = 0u;
        [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        {
            uint candidate = bucket * SHARC_BUCKET_SIZE + p;
            uint age = sharc_frame - g_sharc.Load(GuideEntryAddress(candidate) + GUIDE_LAST_TOUCH);
            if (states[p] != SHARC_LOCKED && age > victimAge)
            {
                victimAge = age;
                slot = candidate;
                expected = states[p];
            }
        }
        if (slot == GUIDE_INVALID || victimAge < SHARC_REPLACE_AGE) return GUIDE_INVALID;
    }
    uint stateAddress = GuideStateAddress(slot), old;
    g_sharc.InterlockedCompareExchange(stateAddress, expected, SHARC_LOCKED, old);
    if (old != expected) return GUIDE_INVALID;
    uint e = GuideEntryAddress(slot);
    GuideInitializeEntry(e, k);
    DeviceMemoryBarrier();
    g_sharc.InterlockedExchange(stateAddress, hash, old);
    return e;
}

// One BSDF-sampled irradiance observation at the receiver: L_i cos / pdf.
void GuideObserve(uint e, float value)
{
    if (e == GUIDE_INVALID || !isfinite(value) || value < 0.0f) return;
    g_sharc.InterlockedAdd64(e + GUIDE_IRRADIANCE, SharcFixed(value, SHARC_RADIANCE_SCALE));
    g_sharc.InterlockedAdd(e + GUIDE_SAMPLES, 1u);
}

// Debug view support: remember that the coarse patch around y is held by some
// receiver, with the luminance it was stored with. Bin GUIDE_BIN_TARGET keys
// never collide with receivers; the entries age out like any other.
void GuideMarkTarget(float3 y, float luminance)
{
    uint e = GuideFindOrInsert(GuideTargetKey(y));
    if (e == GUIDE_INVALID) return;
    g_sharc.Store(e + GUIDE_IRRADIANCE, asuint(luminance));
    g_sharc.InterlockedAdd(e + GUIDE_SAMPLES, 1u);
}

// Register the coarse patch around a hit y, with the outgoing luminance the
// cache stores there, in the receiver entry e. A slot holding the same patch
// is refreshed (and its visibility raised: a ray just reached it); otherwise
// the weakest slot (ranked at the cell centre so every vertex of the cell
// agrees) is replaced when the new patch outweighs it. Concurrent writers may
// tear a slot; readers decode consistently and a torn patch only wastes
// samples until it is replaced. mark: also record the patch for the debug view.
void GuideDiscover(uint e, float3 y, float3 normalY, float luminanceY, bool mark)
{
    if (e == GUIDE_INVALID || !(luminanceY > 0.0f) || !isfinite(luminanceY) || !all(isfinite(y))) return;
    GuideKey k = GuideLoadKey(e);
    float3 center = GuideCellCenter(k);
    float3 axis = GuideBinAxis((k.meta >> 5u) & 7u);
    // Rank by contrast against the cell's mean incident radiance once that is
    // known, so the slots hold the patches guiding can actually exploit.
    uint4 header = g_sharc.Load4(e + GUIDE_IRRADIANCE);
    float irradiance = header.z >= GUIDE_MIN_SAMPLES
        ? SharcFloat(header.xy, rcp(SHARC_RADIANCE_SCALE)) / (float)header.z : 0.0f;
    uint level = GuideLevel(y);
    int3 base; float3 f;
    SharcGrid(y, level, base, f);
    float size = SharcCellSize(level);
    GuideSlot fresh;
    fresh.valid = true;
    fresh.offset = SharcNodeLocal(base, level) + 0.5f * size - center;
    fresh.level = level;
    fresh.luminance = luminanceY;
    fresh.normal = normalY;
    fresh.lastSeen = sharc_frame & 0xffffu;
    fresh.hits = 1u;
    fresh.visibility = 255u;
    GuideCone freshCone = GuideConeOf(fresh, center, center, axis, GUIDE_RANK_MIN_COS);
    freshCone.weight *= GuideContrast(luminanceY, irradiance);
    float weakest = 1e30f;
    uint victim = 0u;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
    {
        GuideSlot s = GuideDecodeSlot(g_sharc.Load4(GuideSlotAddress(e, c)));
        if (!s.valid)
        {
            if (weakest > -1.0f) { weakest = -1.0f; victim = c; }
            continue;
        }
        if (s.level == level && all(abs(s.offset - fresh.offset) < 0.25f * size))
        {
            s.luminance = lerp(s.luminance, luminanceY, rcp((float)min(s.hits + 1u, 16u)));
            s.normal = normalY;
            s.hits = min(s.hits + 1u, 255u);
            s.lastSeen = fresh.lastSeen;
            s.visibility = GuideVisibilityUp(s.visibility);
            g_sharc.Store4(GuideSlotAddress(e, c), GuideEncodeSlot(s));
            if (mark) GuideMarkTarget(y, s.luminance);
            return;
        }
        float w = GuideConeOf(s, center, center, axis, GUIDE_RANK_MIN_COS).weight *
            GuideContrast(s.luminance, irradiance);
        if (w < weakest) { weakest = w; victim = c; }
    }
    if (freshCone.weight > 0.0f && freshCone.weight > weakest)
    {
        g_sharc.Store4(GuideSlotAddress(e, victim), GuideEncodeSlot(fresh));
        if (mark) GuideMarkTarget(y, luminanceY);
    }
}

// Visibility feedback from a guided training ray drawn from slot `slot` of
// receiver e: it reached the patch when the hit lies inside the patch's
// bounding sphere; a miss (sky, an occluder, an emitter in front) lowers the
// estimate. Fully occluded patches therefore lose their weight within a few
// guided samples, and a patch seen again recovers just as fast.
void GuideReport(uint e, uint slot, bool hit, float3 hitPosition)
{
    if (e == GUIDE_INVALID || slot == GUIDE_INVALID) return;
    GuideKey k = GuideLoadKey(e);
    uint address = GuideSlotAddress(e, slot);
    GuideSlot s = GuideDecodeSlot(g_sharc.Load4(address));
    if (!s.valid) return;
    float3 d = hitPosition - (GuideCellCenter(k) + s.offset);
    float r = GUIDE_RADIUS * SharcCellSize(s.level);
    bool reached = hit && dot(d, d) <= r * r;
    if (reached)
    {
        s.visibility = GuideVisibilityUp(s.visibility);
        s.lastSeen = sharc_frame & 0xffffu;
    }
    else s.visibility = GuideVisibilityDown(s.visibility);
    g_sharc.Store4(address, GuideEncodeSlot(s));
}
#endif

//------------------------------------------------------------------
// Prepare-pass maintenance, one thread per entry (called from the cache
// prepare dispatch). Evicts stale or obsolete-LOD entries, windows the
// irradiance mean and retires patches unseen for four lifetimes.
//------------------------------------------------------------------
void GuidePrepareEntry(uint index)
{
    uint stateAddress = GuideStateAddress(index);
    uint e = GuideEntryAddress(index);
    bool evict = sharc_reset != 0u;
    if (!evict && g_sharc.Load(stateAddress) != 0u)
    {
        uint4 header = g_sharc.Load4(e + GUIDE_IRRADIANCE);
        GuideKey k = GuideLoadKey(e);
        float desired = SharcLevel(GuideCellCenter(k)) + (float)GUIDE_LEVEL_OFFSET;
        evict = (float)(sharc_frame - header.w) > (float)sharc_maxAge ||
            abs((float)(k.meta & 31u) - desired) > 3.0f;
        if (!evict)
        {
            if (header.z > GUIDE_WINDOW)
            {
                uint2 half = uint2((header.x >> 1u) | (header.y << 31u), header.y >> 1u);
                g_sharc.Store4(e + GUIDE_IRRADIANCE, uint4(half, header.z >> 1u, header.w));
            }
            if (((index + sharc_frame) & 15u) == 0u)
            {
                [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
                {
                    uint4 w = g_sharc.Load4(GuideSlotAddress(e, c));
                    if ((w.y & GUIDE_SLOT_VALID) == 0u) continue;
                    float age = (float)((sharc_frame - (w.w & 0xffffu)) & 0xffffu);
                    if (age > 4.0f * GUIDE_LIFETIME) g_sharc.Store4(GuideSlotAddress(e, c), 0u);
                }
            }
            // Worth flag: would the cell centre guide at all? Rendering skips
            // the slot block of every receiver without it. Evaluated after the
            // retirement above so a dropped patch stops attracting lookups.
            // Target markers (debug view) carry no slots and stay unflagged.
            uint bin = (k.meta >> 5u) & 7u;
            float q = bin < GUIDE_BIN_TARGET ? GuideBuild(e, GuideCellCenter(k), GuideBinAxis(bin), true).q : 0.0f;
            g_sharc.Store(e + GUIDE_META, (k.meta & GUIDE_META_KEY_MASK) | (q > 0.0f ? GUIDE_META_WORTH : 0u));
        }
    }
    if (evict)
    {
        // GuideInitializeEntry clears the payload before publishing a new key.
        g_sharc.Store(stateAddress, 0u);
    }
}

// Inspector color for the primary vertex.
// Targets (default): the surface's coarse patch is currently held by some
// receiver's slots. Shown as scene radiance (w = 1: normal exposure and
// tonemap) at the luminance it was stored with, fading with the time since a
// receiver last referenced it. Dark grey: not a target.
// Receivers (receiverView): magenta = no receiver entry, amber = too few
// irradiance observations, otherwise green rises with the guided fraction
// q / q_max and blue with the number of stored patches.
float4 GuideDebugColor(float3 position, float3 geometricNormal, float3 normal, bool receiverView)
{
    if (!receiverView)
    {
        uint t = GuideFind(GuideTargetKey(position));
        if (t == GUIDE_INVALID) return float4(0.10f, 0.10f, 0.10f, 0.0f);
        uint4 header = g_sharc.Load4(t + GUIDE_IRRADIANCE); // luminance bits, -, references, last touch
        float age = (float)(sharc_frame - header.w);
        float fade = 1.0f - smoothstep(0.5f * GUIDE_LIFETIME, GUIDE_LIFETIME, age);
        float luminance = asfloat(header.x);
        if (!isfinite(luminance) || luminance < 0.0f) return float4(1, 0, 0, 0);
        return float4(luminance.xxx * fade, 1.0f);
    }
    uint e = GuideFind(GuideKeyCentre(position, geometricNormal));
    if (e == GUIDE_INVALID) return float4(0.45f, 0.025f, 0.35f, 0.0f);
    uint4 header = g_sharc.Load4(e + GUIDE_IRRADIANCE);
    if (header.z < GUIDE_MIN_SAMPLES) return float4(0.65f, 0.32f, 0.025f, 0.0f);
    uint patches = 0u;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
        if ((g_sharc.Load(GuideSlotAddress(e, c) + 4u) & GUIDE_SLOT_VALID) != 0u) ++patches;
    GuideSet g = GuideBuild(e, position, normal, true);
    float q = GUIDE_QMAX > 0.0f ? saturate(g.q / GUIDE_QMAX) : 0.0f;
    return float4(0.06f, 0.06f + 0.9f * q, 0.06f + 0.4f * (float)patches / (float)GUIDE_SLOTS, 0.0f);
}
#endif
