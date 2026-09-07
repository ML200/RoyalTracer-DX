#ifndef SHARC_V8_HLSLI
#define SHARC_V8_HLSLI
#include "SharcLayout.h"
#ifndef SHARC_COMPACT_QUERY
#define SHARC_COMPACT_QUERY 0
#endif
#if SHARC_COMPACT_QUERY && SHARC_BUCKET_SIZE > 32
#error SHARC_COMPACT_QUERY requires a bucket that fits in a uint mask
#endif

// Original SHaRC implementation. See docs/SHARC.md for the estimators and sources.
// Update and query are separate dispatches. Metadata is immutable between prepare
// passes. Per-frame sums are unsigned fixed point added with single-shot atomics.
// Only concurrent update publication needs device-coherent loads. Rendering
// reads an immutable snapshot behind the resolve UAV barrier and can use caches.
#if SHARC_UPDATE_PASS && !defined(SHARC_READ_ONLY)
globallycoherent RWByteAddressBuffer g_sharc : register(u27);
#else
RWByteAddressBuffer g_sharc : register(u27);
#endif
static const uint SHARC_LOCKED = 0xffffffffu;
static const uint SHARC_INVALID = 0xffffffffu;

// Entry layout (byte offsets relative to the entry). Sectors: key + surface
// descriptor in [0,64), frame sums in [64,112), history in [112,160). A probe
// reads the descriptor and the [128,160) history sector with independent
// 16-byte loads, so after the bucket's state words it costs one round trip.
static const uint SHARC_NODE = 0u;              // int3 grid node
static const uint SHARC_META = 12u;             // meta, instance, material
static const uint SHARC_GEOMETRIC_NORMAL = 24u; // packed
static const uint SHARC_SHADING_NORMAL = 28u;   // packed
static const uint SHARC_REPRESENTATIVE = 32u;   // float3 position relative to the node
static const uint SHARC_DEMODULATOR = 44u;      // float3
static const uint SHARC_ROUGHNESS = 56u;
static const uint SHARC_FRAME_RGB = 64u;        // 3 x uint64, 2^-24 units
static const uint SHARC_FRAME_L2 = 88u;         // uint64, 2^-24 units
static const uint SHARC_FRAME_W_W2 = 96u;       // uint64: low word W, high word W^2, 2^-16 units
static const uint SHARC_FRAME_POSITIVE = 104u;  // uint32, 2^-16 units
static const uint SHARC_HISTORY_W = 112u;       // float
static const uint SHARC_HISTORY_L2 = 116u;
static const uint SHARC_HISTORY_W2 = 120u;
static const uint SHARC_HISTORY_POSITIVE = 124u;
static const uint SHARC_MEAN = 128u;            // float3
static const uint SHARC_CONFIDENCE = 140u;
static const uint SHARC_LAST_UPDATE = 144u;     // frame of the last resolved observation
static const uint SHARC_INTERVAL = 148u;        // measured revisit interval (frames)
static const uint SHARC_FRAMES = 152u;          // resolved observation frames
static const uint SHARC_LAST_TOUCH = 156u;      // frame of the last registration (replacement guard)

static const float SHARC_RADIANCE_SCALE = 16777216.0f; // 2^24: 6e-8 resolution, 2^40 range
static const float SHARC_WEIGHT_SCALE = 65536.0f;      // 2^16: splat weights are <= 1
// A full bucket replaces its least recently observed record only after this
// many idle frames, so contended buckets do not thrash between live nodes.
static const uint SHARC_REPLACE_AGE = 16u;
// Separate surface records per key. Geometry below the cell size (foliage,
// railings) would otherwise need one record per leaf, all in the SAME bucket:
// it fills with young single-observation records and nothing inserts again.
// From this many records on, a node is "merged": deposits and queries weight
// its records by geometric-normal agreement only, so they act as orientation
// clusters and the cell caches the detail's average, like a position-only grid.
static const uint SHARC_MERGE_RECORDS = 4u;

struct SharcSurface
{
    float3 position;
    float3 geometricNormal;
    float3 normal;
    float3 demodulator;
    uint instance;
    uint material;
    uint variant;
    float roughness;
};

// Immutable key + surface descriptor, sectors [0,64).
struct SharcDescriptor
{
    int3 node;
    uint meta;
    uint instance;
    uint material;
    float3 geometricNormal;
    float3 normal;
    float3 representative;
    float3 demodulator;
    float roughness;
};

// Resolved state a query needs, sector [128,160).
struct SharcHistory
{
    float3 mean;
    float confidence;
    uint lastUpdate;
    float interval;
    uint frames;
    uint lastTouch;
};

uint SharcStateAddress(uint slot) { return slot * 4u; }
uint SharcEntryAddress(uint slot) { return SHARC_STATE_BYTES + slot * SHARC_ENTRY_BYTES; }
uint SharcDirtyAddress(uint word) { return SHARC_DIRTY_OFFSET + word * 4u; }
uint SharcBucketOf(uint hash) { return hash & (SHARC_CAPACITY / SHARC_BUCKET_SIZE - 1u); }

SharcDescriptor SharcLoadDescriptor(uint e)
{
    uint4 a = g_sharc.Load4(e + SHARC_NODE);
    uint4 b = g_sharc.Load4(e + SHARC_NODE + 16u);
    uint4 c = g_sharc.Load4(e + SHARC_REPRESENTATIVE);
    uint4 d = g_sharc.Load4(e + SHARC_REPRESENTATIVE + 16u);
    SharcDescriptor r;
    r.node = asint(a.xyz);
    r.meta = a.w;
    r.instance = b.x;
    r.material = b.y;
    r.geometricNormal = UnpackNormal(b.z);
    r.normal = UnpackNormal(b.w);
    r.representative = asfloat(c.xyz);
    r.demodulator = asfloat(uint3(c.w, d.x, d.y));
    r.roughness = asfloat(d.z);
    return r;
}

SharcHistory SharcLoadHistory(uint e)
{
    uint4 a = g_sharc.Load4(e + SHARC_MEAN);
    uint4 b = g_sharc.Load4(e + SHARC_LAST_UPDATE);
    SharcHistory h;
    h.mean = asfloat(a.xyz);
    h.confidence = asfloat(a.w);
    h.lastUpdate = b.x;
    h.interval = asfloat(b.y);
    h.frames = b.z;
    h.lastTouch = b.w;
    return h;
}

float SharcCellSize(uint level) { return sharc_cellSize * exp2((float)level); }

float SharcLevel(float3 position)
{
    float originMagnitude = max(abs(sceneOriginWorld.x), max(abs(sceneOriginWorld.y), abs(sceneOriginWorld.z)));
    float coordinateLevel = max(0.0f, ceil(log2(max(originMagnitude /
        (sharc_cellSize * 536870912.0f), 1.0f)))); // signed int32 coordinate headroom
    return clamp(log2(max(length(position - InitOrigin()) * sharc_lodScale /
        sharc_cellSize, 1.0f)), coordinateLevel, (float)(SHARC_MAX_LEVEL - 1u));
}

// Split the snapped floating origin from the small local coordinates BEFORE
// quantization. Powers-of-two cell sizes keep this exact through km rebases;
// forming position + sceneOriginWorld first would erase centimetre detail.
void SharcGrid(float3 position, uint level, out int3 base, out float3 fraction)
{
    float size = SharcCellSize(level);
    float3 originGrid = floor(sceneOriginWorld / size);
    float3 localGrid = (position + (sceneOriginWorld - originGrid * size)) / size;
    float3 localBase = floor(localGrid);
    base = int3(originGrid) + int3(localBase);
    fraction = localGrid - localBase;
}

float3 SharcNodeLocal(int3 node, uint level)
{
    float size = SharcCellSize(level);
    float3 originGrid = floor(sceneOriginWorld / size);
    return float3(node - int3(originGrid)) * size -
        (sceneOriginWorld - originGrid * size);
}

// Coarse geometric normal direction is part of the key; exact oriented normals
// additionally reject opposite faces and steep folds within a coarse bucket.
float3 SharcNormalWeights(float3 n)
{
    float3 w = n * n;
    w *= w;
    return w / max(w.x + w.y + w.z, 1e-20f);
}

uint SharcMeta(SharcSurface s, uint level, uint axis)
{
    uint normalBin = axis * 2u + (s.geometricNormal[axis] < 0.0f ? 1u : 0u);
    return level | (normalBin << 5u) | (s.variant << 8u);
}

uint SharcHash(int3 node, uint meta, uint instance, uint material)
{
    uint h = Hash32(asuint(node.x) ^ 0x9e3779b9u);
    h = Hash32(h ^ asuint(node.y));
    h = Hash32(h ^ asuint(node.z));
    h = Hash32(h ^ meta);
    h = Hash32(h ^ instance);
    return 1u + Hash32(h ^ material) % 0xfffffffeu;
}

bool SharcKeyMatches(SharcDescriptor d, int3 node, uint meta, SharcSurface s)
{
    return all(d.node == node) && d.meta == meta && d.instance == s.instance && d.material == s.material;
}

// One 64-byte read yields every state word of a bucket. All later loops over
// the slots are unrolled, so the array stays in registers.
void SharcLoadBucket(uint bucket, out uint states[SHARC_BUCKET_SIZE])
{
    uint base = SharcStateAddress(bucket * SHARC_BUCKET_SIZE);
    [unroll] for (uint q = 0u; q < SHARC_BUCKET_SIZE / 4u; ++q)
    {
        uint4 v = g_sharc.Load4(base + q * 16u);
        states[q * 4u + 0u] = v.x;
        states[q * 4u + 1u] = v.y;
        states[q * 4u + 2u] = v.z;
        states[q * 4u + 3u] = v.w;
    }
}

// Records of this key already in the bucket (fingerprint collisions within one
// bucket are negligible, so the state words suffice and no memory is touched).
bool SharcMergedNode(uint states[SHARC_BUCKET_SIZE], uint hash)
{
    uint sameKey = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        sameKey += states[p] == hash ? 1u : 0u;
    return sameKey >= SHARC_MERGE_RECORDS;
}

// Merged-node support: orientation agreement only. Plane, shading normal,
// roughness and reflectance are what sub-cell detail cannot satisfy; albedo is
// demodulated anyway and the normal bin already fixes the dominant axis sign.
float SharcMergedWeight(SharcDescriptor d, SharcSurface s)
{
    return smoothstep(0.3f, 0.8f, dot(d.geometricNormal, s.geometricNormal));
}

// Smooth bilateral support over geometric/shading normals, BOTH tangent-plane
// distances, roughness and reflectance. Neither thin opposite faces nor nearby
// parallel layers inherit each other's history. The arrival direction is NOT
// part of the record: eligible surfaces are diffuse by construction, and
// splitting records per direction multiplied the table load without changing
// the cached quantity.
float SharcSurfaceWeight(SharcDescriptor d, SharcSurface s, float3 relative, float size)
{
    float3 delta = relative - d.representative;
    float plane = max(abs(dot(delta, d.geometricNormal)), abs(dot(delta, s.geometricNormal)));
    float w = smoothstep(0.90f, 0.99f, dot(d.geometricNormal, s.geometricNormal));
    w *= smoothstep(0.90f, 0.99f, dot(d.normal, s.normal));
    w *= 1.0f - smoothstep(0.015f * size, 0.05f * size, plane);
    w *= 1.0f - smoothstep(0.05f, 0.15f, abs(s.roughness - d.roughness));
    // Albedo demodulation removes texture detail from the cache. Reject large
    // reflectance changes anyway: the layered specular remainder is not Kd-linear.
    float3 ratio = s.demodulator / d.demodulator;
    // Positive reflectances: max(abs(log2(r))) = log2(max(max(r), 1/min(r))).
    // One logarithm instead of three for every compatible cache record.
    float ratioMax = max(ratio.x, max(ratio.y, ratio.z));
    float ratioMin = min(ratio.x, min(ratio.y, ratio.z));
    w *= 1.0f - smoothstep(0.3f, 0.7f, log2(max(ratioMax, rcp(ratioMin))));
    return w;
}

void SharcInitializeEntry(uint e, int3 node, uint meta, SharcSurface s, float3 relative)
{
    g_sharc.Store4(e + SHARC_NODE, uint4(asuint(node), meta));
    g_sharc.Store4(e + SHARC_NODE + 16u, uint4(s.instance, s.material,
        PackNormal(s.geometricNormal), PackNormal(s.normal)));
    g_sharc.Store4(e + SHARC_REPRESENTATIVE, uint4(asuint(relative), asuint(s.demodulator.x)));
    g_sharc.Store4(e + SHARC_REPRESENTATIVE + 16u, uint4(asuint(s.demodulator.y),
        asuint(s.demodulator.z), asuint(s.roughness), 0u));
    // Sums and history start from zero whether the slot was empty or replaced.
    [unroll] for (uint b = SHARC_FRAME_RGB; b < SHARC_ENTRY_BYTES; b += 16u)
        g_sharc.Store4(e + b, 0u);
    g_sharc.Store(e + SHARC_LAST_UPDATE, sharc_frame);
    g_sharc.Store(e + SHARC_LAST_TOUCH, sharc_frame);
}

// Entry address of a compatible record for the node, creating one if needed.
// A full bucket replaces its least recently observed record when that record
// has been idle for SHARC_REPLACE_AGE frames and nobody registered into it this
// frame; otherwise the observation is a miss and is retried on a later frame.
uint SharcFindOrInsert(int3 node, uint level, uint axis, SharcSurface s, float3 relative, out float weight)
{
    uint meta = SharcMeta(s, level, axis);
    uint hash = SharcHash(node, meta, s.instance, s.material);
    uint bucket = SharcBucketOf(hash);
    float size = SharcCellSize(level);
    uint states[SHARC_BUCKET_SIZE];
    SharcLoadBucket(bucket, states);
    const bool merged = SharcMergedNode(states, hash);
    weight = 1.0f;
    uint slot = SHARC_INVALID;
    uint best = SHARC_INVALID;
    float bestWeight = 0.05f;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
    {
        uint candidate = bucket * SHARC_BUCKET_SIZE + p;
        if (states[p] == hash)
        {
            uint e = SharcEntryAddress(candidate);
            DeviceMemoryBarrier(); // acquire immutable metadata from its publisher
            SharcDescriptor d = SharcLoadDescriptor(e);
            if (SharcKeyMatches(d, node, meta, s))
            {
                if (merged)
                {
                    // Orientation cluster of a sub-cell-detail node.
                    float w = SharcMergedWeight(d, s);
                    if (w > bestWeight) { bestWeight = w; best = e; }
                }
                else
                {
                    float w = SharcSurfaceWeight(d, s, relative, size);
                    // Several separated sheets can occupy one spatial/normal cell.
                    // Give them distinct surface records instead of starving the second.
                    if (w > 0.25f)
                    {
                        g_sharc.Store(e + SHARC_LAST_TOUCH, sharc_frame);
                        weight = w;
                        return e;
                    }
                }
            }
        }
        else if (states[p] == 0u && slot == SHARC_INVALID) slot = candidate;
        // Locked slots are being initialized by another lane; keep scanning.
    }
    if (best != SHARC_INVALID)
    {
        g_sharc.Store(best + SHARC_LAST_TOUCH, sharc_frame);
        weight = bestWeight;
        return best;
    }
    // Search the WHOLE bucket before reusing a hole; an existing matching
    // surface may lie beyond it. Otherwise eviction would duplicate live keys.
    uint expected = 0u;
    if (slot == SHARC_INVALID)
    {
        uint victimAge = 0u;
        [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        {
            uint candidate = bucket * SHARC_BUCKET_SIZE + p;
            uint4 h = g_sharc.Load4(SharcEntryAddress(candidate) + SHARC_LAST_UPDATE);
            uint age = sharc_frame - h.x;
            if (states[p] != SHARC_LOCKED && h.w != sharc_frame && age > victimAge)
            {
                victimAge = age;
                slot = candidate;
                expected = states[p];
            }
        }
        if (slot == SHARC_INVALID || victimAge < SHARC_REPLACE_AGE) return SHARC_INVALID;
    }
    uint stateAddress = SharcStateAddress(slot), old;
    g_sharc.InterlockedCompareExchange(stateAddress, expected, SHARC_LOCKED, old);
    if (old != expected) return SHARC_INVALID; // claimed by another lane; retry next frame
    uint e = SharcEntryAddress(slot);
    SharcInitializeEntry(e, node, meta, s, relative);
    DeviceMemoryBarrier();
    g_sharc.InterlockedExchange(stateAddress, hash, old); // publish only fully initialized metadata
    return e;
}

uint64_t SharcFixed(float value, float scale)
{
    return (uint64_t)(min(value, 1e10f) * scale + 0.5f);
}

float SharcFloat(uint2 words, float inverseScale)
{
    return (float(words.y) * 4294967296.0f + float(words.x)) * inverseScale;
}

// Single-shot fixed-point atomics: no returned value, no retry loop, so the
// commit does not wait on memory. 2^-24 resolution over a 2^40 range needs no
// clipping of rare bright samples.
void SharcAccumulate(uint e, float3 radiance, float w)
{
    if (e == SHARC_INVALID || !all(isfinite(radiance)) || !isfinite(w) || w <= 0.0f) return;
    radiance = max(radiance, 0.0f);
    w = min(w, 1.0f);
    float lum = Luma(radiance);
    uint fixedW = (uint)(w * SHARC_WEIGHT_SCALE + 0.5f);
    uint fixedW2 = (uint)(w * w * SHARC_WEIGHT_SCALE + 0.5f);
    g_sharc.InterlockedAdd64(e + SHARC_FRAME_RGB, SharcFixed(radiance.x * w, SHARC_RADIANCE_SCALE));
    g_sharc.InterlockedAdd64(e + SHARC_FRAME_RGB + 8u, SharcFixed(radiance.y * w, SHARC_RADIANCE_SCALE));
    g_sharc.InterlockedAdd64(e + SHARC_FRAME_RGB + 16u, SharcFixed(radiance.z * w, SHARC_RADIANCE_SCALE));
    g_sharc.InterlockedAdd64(e + SHARC_FRAME_L2, SharcFixed(lum * lum * w, SHARC_RADIANCE_SCALE));
    g_sharc.InterlockedAdd64(e + SHARC_FRAME_W_W2, ((uint64_t)fixedW2 << 32) | (uint64_t)fixedW);
    if (lum > 1e-8f) g_sharc.InterlockedAdd(e + SHARC_FRAME_POSITIVE, fixedW);
    // Resolve runs after the update UAV barrier; a bit publishes all sums for
    // this slot, including zero-radiance observations and concurrent deposits.
    uint slot = (e - SHARC_STATE_BYTES) / SHARC_ENTRY_BYTES;
    g_sharc.InterlockedOr(SharcDirtyAddress(slot >> 5u), 1u << (slot & 31u));
}

bool SharcValidSurface(SharcSurface s)
{
    return all(isfinite(s.position)) && all(isfinite(s.geometricNormal)) &&
        all(isfinite(s.normal)) && all(isfinite(s.demodulator)) && all(s.demodulator > 0.0f) &&
        isfinite(s.roughness) && dot(s.geometricNormal, s.geometricNormal) > 0.5f &&
        dot(s.normal, s.normal) > 0.5f;
}

// One deposit per vertex: a stochastic tent corner at the level the rendering
// query samples with the same probability. Matching that distribution keeps the
// expected reconstruction kernel and halves the insertion/atomic traffic.
bool SharcAllocateDeposit(SharcSurface s, inout uint seed, out uint address, out float weight)
{
    address = SHARC_INVALID;
    weight = 0.0f;
    if (!SharcValidSurface(s)) return false;
    float lod = SharcLevel(s.position);
    uint level = (uint)lod + (RandomFloatSingle(seed) < frac(lod) ? 1u : 0u);
    float3 normalWeights = SharcNormalWeights(s.geometricNormal);
    float r = RandomFloatSingle(seed);
    uint axis = r < normalWeights.x ? 0u : (r < normalWeights.x + normalWeights.y ? 1u : 2u);
    int3 base; float3 f;
    SharcGrid(s.position, level, base, f);
    float size = SharcCellSize(level);
    uint3 corner = uint3(RandomFloatSingle(seed) < f.x,
        RandomFloatSingle(seed) < f.y, RandomFloatSingle(seed) < f.z);
    float3 relative = (f - float3(corner)) * size;
    float w;
    uint e = SharcFindOrInsert(base + int3(corner), level, axis, s, relative, w);
    if (e == SHARC_INVALID || !isfinite(w) || w <= 1e-5f) return false;
    address = e;
    weight = w;
    return true;
}

void SharcSplat(SharcSurface s, float3 radiance, inout uint seed)
{
    // Check BEFORE max/division so NaNs cannot silently turn into black samples.
    if (!all(isfinite(radiance)) || !SharcValidSurface(s)) return;
    radiance = max(radiance / s.demodulator, 0.0f);
    if (!all(isfinite(radiance))) return;
    uint address; float weight;
    if (SharcAllocateDeposit(s, seed, address, weight)) SharcAccumulate(address, radiance, weight);
}

float SharcStatisticalConfidence(uint e)
{
    float w = asfloat(g_sharc.Load(e + SHARC_HISTORY_W));
    float w2 = asfloat(g_sharc.Load(e + SHARC_HISTORY_W2));
    float neff = w * w / max(w2, 1e-20f);
    float mean = Luma(asfloat(g_sharc.Load3(e + SHARC_MEAN)));
    float variance = max(asfloat(g_sharc.Load(e + SHARC_HISTORY_L2)) - mean * mean, 0.0f);
    float error = sqrt(variance / max(neff, 1.0f)) / max(mean, 1e-8f);
    uint frames = g_sharc.Load(e + SHARC_FRAMES);
    // Zeros alone cannot establish that a difficult multi-bounce path is black.
    // Positive evidence and variance both matter; splat neighbors are correlated
    // and their sample counts must never be added together at query time.
    float positive = asfloat(g_sharc.Load(e + SHARC_HISTORY_POSITIVE)) / max(w, 1e-20f) * neff;
    float confidence = smoothstep((float)sharc_minSamples, 2.0f * sharc_minSamples, neff);
    confidence *= smoothstep(2.0f, 4.0f, (float)frames);
    confidence *= smoothstep(3.0f, 8.0f, positive);
    confidence *= 1.0f - smoothstep(0.20f, 0.50f, error);
    return all(isfinite(float4(mean, variance, neff, confidence))) ? confidence : 0.0f;
}

float SharcAgeLimit(SharcHistory h)
{
    float interval = isfinite(h.interval) ? h.interval : 0.0f;
    // Rarely observed fine cells need time to collect evidence. Bound retention
    // so abandoned regions still refill; obsolete LODs are evicted independently.
    float limit = clamp(16.0f * interval, (float)sharc_maxAge, 4.0f * sharc_maxAge);
    // A record with a single resolved observation has no usable history yet.
    // Holding every stray deep-bounce record for the full lifetime saturated
    // the table; give it a quarter lifetime to receive a second observation.
    if (h.frames < 2u) limit = min(limit, max(0.25f * sharc_maxAge, 32.0f));
    return limit;
}

float SharcConfidence(SharcHistory h)
{
    float age = (float)(sharc_frame - h.lastUpdate);
    // Sparse cells retain evidence between observations. Fading them to zero
    // after 32 frames previously made small, rarely visited cells unusable.
    float ageLimit = SharcAgeLimit(h);
    float confidence = h.confidence * (1.0f - smoothstep(0.5f * ageLimit, ageLimit, age));
    return isfinite(confidence) ? saturate(confidence) : 0.0f;
}

// ignoreConfidence: any record with resolved history counts with its surface
// weight alone (training tail termination), instead of scaling by confidence.
void SharcQueryNode(SharcSurface s, uint level, uint axis, int3 node, float3 relative,
    bool ignoreConfidence, out float3 sum, out float support)
{
    uint meta = SharcMeta(s, level, axis);
    uint hash = SharcHash(node, meta, s.instance, s.material);
    uint bucket = SharcBucketOf(hash);
    uint states[SHARC_BUCKET_SIZE];
    SharcLoadBucket(bucket, states);
    const bool merged = SharcMergedNode(states, hash);
    sum = 0.0f; support = 0.0f;
    float geometricSupport = 0.0f;
#if SHARC_COMPACT_QUERY
    // Compact the immutable bucket snapshot into matching slots. Iterating
    // lowest bit first preserves the original floating-point accumulation order
    // while compiling just one descriptor/history reconstruction body.
    uint matchingSlots = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        if (states[p] == hash) matchingSlots |= 1u << p;
    [loop] while (matchingSlots != 0u)
    {
        const uint p = (uint)firstbitlow(matchingSlots);
        matchingSlots &= matchingSlots - 1u;
#else
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
    {
        if (states[p] != hash) continue;
#endif
        uint e = SharcEntryAddress(bucket * SHARC_BUCKET_SIZE + p);
        // Descriptor and history are fetched together, before any test, so a
        // matching record costs one memory round trip rather than four.
        SharcDescriptor d = SharcLoadDescriptor(e);
        SharcHistory h = SharcLoadHistory(e);
        if (!SharcKeyMatches(d, node, meta, s)) continue;
        float surfaceWeight = merged ? SharcMergedWeight(d, s)
            : SharcSurfaceWeight(d, s, relative, SharcCellSize(level));
        if (!isfinite(surfaceWeight) || surfaceWeight <= 0.0f) continue;
        geometricSupport += surfaceWeight;
        float confidence = ignoreConfidence ? (h.frames > 0u ? 1.0f : 0.0f) : SharcConfidence(h);
        // Do not multiply rejected history by zero: IEEE NaN * 0 is still NaN.
        if (confidence <= 0.0f || !all(isfinite(h.mean))) continue;
        sum += h.mean * (surfaceWeight * confidence);
        support += surfaceWeight * confidence;
    }
    float scale = rcp(max(geometricSupport, 1.0f));
    sum *= scale;
    support *= scale;
}

// Deterministic reconstruction retained for numerical comparisons. The live
// tracer samples the SAME kernel stochastically below, visiting only one node.
void SharcQueryLevel(SharcSurface s, uint level, out float3 sum, out float support)
{
    int3 base; float3 f;
    SharcGrid(s.position, level, base, f);
    sum = 0.0f; support = 0.0f;
    float3 normalWeights = SharcNormalWeights(s.geometricNormal);
    [loop] for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (normalWeights[axis] <= 1e-5f) continue;
        [loop] for (uint c = 0u; c < 8u; ++c)
        {
            uint3 corner = uint3(c & 1u, (c >> 1u) & 1u, c >> 2u);
            float3 t = lerp(1.0f - f, f, float3(corner));
            float w = t.x * t.y * t.z * normalWeights[axis];
            if (w <= 1e-5f) continue;
            float3 nodeSum; float nodeSupport;
            SharcQueryNode(s, level, axis, base + int3(corner),
                (f - float3(corner)) * SharcCellSize(level), false, nodeSum, nodeSupport);
            sum += nodeSum * w;
            support += nodeSupport * w;
        }
    }
}

// Footprint acceptance of a rendering query as a function of the path spread
// at a point of the given distance level. ReSTIR lite draws against it at
// its candidate vertex without the always-trace share (Pass_pt_v8.hlsl).
float SharcFootprintRamp(float pathSpread, uint level)
{
    return smoothstep(sharc_queryFootprint, 2.0f * sharc_queryFootprint,
        pathSpread / SharcCellSize(level + 1u));
}

// One node of the rendering reconstruction, every choice drawn from `seed`.
// Sample the partition of unity over LOD, normal and trilinear corners.
// E[accepted cached contribution] and E[fallback probability] match the
// deterministic 48-node reconstruction. This replaces grid discontinuities
// with ordinary sampling noise, without renormalizing missing cells bright.
bool SharcQueryDraws(SharcSurface s, inout uint seed, out float3 radiance)
{
    radiance = 0.0f;
    float lod = SharcLevel(s.position);
    uint level = (uint)lod;
    level += RandomFloatSingle(seed) < frac(lod) ? 1u : 0u;
    float3 normalWeights = SharcNormalWeights(s.geometricNormal);
    float r = RandomFloatSingle(seed);
    uint axis = r < normalWeights.x ? 0u : (r < normalWeights.x + normalWeights.y ? 1u : 2u);
    int3 base; float3 f;
    SharcGrid(s.position, level, base, f);
    uint3 corner = uint3(RandomFloatSingle(seed) < f.x,
        RandomFloatSingle(seed) < f.y, RandomFloatSingle(seed) < f.z);
    float3 sum; float support;
    SharcQueryNode(s, level, axis, base + int3(corner),
        (f - float3(corner)) * SharcCellSize(level), false, sum, support);
    if (support <= 1e-5f || RandomFloatSingle(seed) >= support) return false;
    radiance = sum / support * s.demodulator;
    return all(isfinite(radiance));
}

bool SharcQueryFootprintAccepted(float3 position, float pathSpread, inout uint seed)
{
    // Require a path footprint several times wider than even the coarser cell.
    float footprint = SharcFootprintRamp(pathSpread, (uint)SharcLevel(position));
    if (footprint <= 0.0f) return false;
    if (RandomFloatSingle(seed) >= footprint * (31.0f / 32.0f)) return false;
    return true;
}

bool SharcQuery(SharcSurface s, float pathSpread, inout uint seed, out float3 radiance)
{
    radiance = 0.0f;
    return SharcQueryFootprintAccepted(s.position, pathSpread, seed) &&
        SharcQueryDraws(s, seed, radiance);
}

// Tail termination for a training path at its depth cap: any resolved record
// for this vertex, without the footprint, confidence or always-trace gates.
// Roulette is compensated, the cap is not; an uncertain estimate of the
// vertex's outgoing radiance is unbiased where the dropped suffix would have
// been trained as darkness and propagated through resampling.
bool SharcQueryForced(SharcSurface s, inout uint seed, out float3 radiance)
{
    radiance = 0.0f;
    float lod = SharcLevel(s.position);
    uint level = (uint)lod + (RandomFloatSingle(seed) < frac(lod) ? 1u : 0u);
    float3 normalWeights = SharcNormalWeights(s.geometricNormal);
    float r = RandomFloatSingle(seed);
    uint axis = r < normalWeights.x ? 0u : (r < normalWeights.x + normalWeights.y ? 1u : 2u);
    int3 base; float3 f;
    SharcGrid(s.position, level, base, f);
    uint3 corner = uint3(RandomFloatSingle(seed) < f.x,
        RandomFloatSingle(seed) < f.y, RandomFloatSingle(seed) < f.z);
    float3 sum; float support;
    SharcQueryNode(s, level, axis, base + int3(corner),
        (f - float3(corner)) * SharcCellSize(level), true, sum, support);
    if (support <= 1e-5f) return false;
    radiance = sum / support * s.demodulator;
    return all(isfinite(radiance));
}

#endif
