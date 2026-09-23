#pragma once
#include "SharcLayout.h"

static const uint SHARC_LOCKED = 0xffffffffu;
static const uint SHARC_INVALID = 0xffffffffu;

static const uint SHARC_QUERY = 0u;
static const uint SHARC_GEOMETRIC_NORMAL = 4u;
static const uint SHARC_SHADING_NORMAL = 8u;
static const uint SHARC_NODE = 32u;
static const uint SHARC_META = 44u;
static const uint SHARC_INSTANCE = 48u;
static const uint SHARC_MATERIAL = 52u;
static const uint SHARC_FRAME_RGB = 64u;
static const uint SHARC_FRAME_L2 = 88u;
static const uint SHARC_FRAME_W_W2 = 96u;
static const uint SHARC_FRAME_POSITIVE = 104u;
static const uint SHARC_HISTORY_W = 112u;
static const uint SHARC_HISTORY_L2 = 116u;
static const uint SHARC_HISTORY_W2 = 120u;
static const uint SHARC_HISTORY_POSITIVE = 124u;
static const uint SHARC_MEAN = 128u;
static const uint SHARC_CONFIDENCE = 140u;
static const uint SHARC_LAST_UPDATE = 144u;
static const uint SHARC_INTERVAL = 148u;
static const uint SHARC_FRAMES = 152u;
static const uint SHARC_LAST_TOUCH = 156u;

static const float SHARC_RADIANCE_SCALE = 16777216.0f;
static const float SHARC_WEIGHT_SCALE = 65536.0f;

static const uint SHARC_REPLACE_AGE = 16u;

static const uint SHARC_MERGE_RECORDS = 4u;

// How far a query may differ from an entry and still read it. The geometric normal and the plane
// keep light from leaking between surfaces; the shading normal, the albedo (the entries store
// demodulated radiance) and the roughness only vary with texture detail, which a query through a
// wide cone averages anyway, so they are allowed to differ a lot before an entry is split.
static const float2 SHARC_SIMILAR_NORMAL    = float2(0.50f, 0.80f);   // shading-normal cosine
static const float2 SHARC_SIMILAR_ALBEDO    = float2(1.00f, 2.00f);   // log2 of the albedo ratio
static const float2 SHARC_SIMILAR_ROUGHNESS = float2(0.15f, 0.35f);   // roughness difference

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

struct SharcDescriptor
{
    float3 geometricNormal;
    float3 normal;
    float3 representative;
    float3 demodulator;
    float roughness;
};

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

// Hash spatial identity and surface identity into one cache key.
uint SharcCheckHash(int3 node, uint meta, uint instance, uint material)
{
    uint h = Hash32(asuint(node.z) ^ 0x7f4a7c15u);
    h = Hash32(h ^ asuint(node.y) ^ 0x2545f491u);
    h = Hash32(h ^ asuint(node.x));
    h = Hash32(h ^ material);
    h = Hash32(h ^ instance);
    return Hash32(h ^ meta);
}
uint SharcPackSnorm10x3(float3 v)
{
    uint3 q = (uint3)clamp(round(v * 511.0f) + 512.0f, 1.0f, 1023.0f);
    return q.x | (q.y << 10u) | (q.z << 20u);
}
float3 SharcUnpackSnorm10x3(uint w)
{
    return (float3(w & 1023u, (w >> 10u) & 1023u, (w >> 20u) & 1023u) - 512.0f) * (1.0f / 511.0f);
}
uint SharcPackLog10x3(float3 v)
{
    uint3 q = (uint3)clamp((log2(max(v, 1e-20f)) + 12.0f) * 64.0f + 0.5f, 0.0f, 1023.0f);
    return q.x | (q.y << 10u) | (q.z << 20u);
}
float3 SharcUnpackLog10x3(uint w)
{
    return exp2(float3(w & 1023u, (w >> 10u) & 1023u, (w >> 20u) & 1023u) * (1.0f / 64.0f) - 12.0f);
}

uint SharcPackMean(float v) { return f32tof16(min(v, 1048064.0f) * 0.0625f); }
float SharcUnpackMean(uint h) { return f16tof32(h) * 16.0f; }
uint SharcPackUnorm8(float v) { return (uint)(saturate(v) * 255.0f + 0.5f); }

// Decode the packed query record shared by all SHaRC passes.
void SharcLoadQueryRecord(uint e, float size, out uint check, out SharcDescriptor d, out SharcHistory h)
{
    uint4 a = g_sharc.Load4(e + SHARC_QUERY);
    uint4 b = g_sharc.Load4(e + SHARC_QUERY + 16u);
    check = a.x;
    d.geometricNormal = UnpackNormal(a.y);
    d.normal = UnpackNormal(a.z);
    d.representative = SharcUnpackSnorm10x3(a.w) * size;
    d.demodulator = SharcUnpackLog10x3(b.x);
    d.roughness = (float)((b.z >> 16u) & 255u) * (1.0f / 255.0f);
    h.mean = float3(SharcUnpackMean(b.y & 0xffffu), SharcUnpackMean(b.y >> 16u), SharcUnpackMean(b.z & 0xffffu));
    h.confidence = (float)(b.z >> 24u) * (1.0f / 255.0f);

    h.lastUpdate = sharc_frame - ((sharc_frame - (b.w & 0xffffu)) & 0xffffu);
    h.interval = f16tof32(b.w >> 16u);
    h.frames = a.w >> 30u;
    h.lastTouch = 0u;
}
float3 SharcLoadDemodulator(uint e) { return SharcUnpackLog10x3(g_sharc.Load(e + SHARC_QUERY + 16u)); }

void SharcPublishQueryHistory(uint e, float3 mean, float confidence, uint frames, float interval)
{
    uint w3 = g_sharc.Load(e + SHARC_QUERY + 12u);
    g_sharc.Store(e + SHARC_QUERY + 12u, (w3 & 0x3fffffffu) | (min(frames, 3u) << 30u));
    uint w6 = g_sharc.Load(e + SHARC_QUERY + 24u);
    g_sharc.Store3(e + SHARC_QUERY + 20u, uint3(
        SharcPackMean(mean.x) | (SharcPackMean(mean.y) << 16u),
        SharcPackMean(mean.z) | (w6 & 0x00ff0000u) | (SharcPackUnorm8(confidence) << 24u),
        (sharc_frame & 0xffffu) | (f32tof16(interval) << 16u)));
}

void SharcLoadKey(uint e, out uint4 key, out uint2 idm)
{
    key = g_sharc.Load4(e + SHARC_NODE);
    idm = g_sharc.Load2(e + SHARC_INSTANCE);
}
bool SharcKeyMatches(uint4 key, uint2 idm, int3 node, uint meta, SharcSurface s)
{
    return all(asint(key.xyz) == node) && key.w == meta && idm.x == s.instance && idm.y == s.material;
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
float3 SharcLoadRepresentative(uint e)
{
    uint level = g_sharc.Load(e + SHARC_META) & 31u;
    return SharcUnpackSnorm10x3(g_sharc.Load(e + SHARC_QUERY + 12u)) * SharcCellSize(level);
}

float SharcLevel(float3 position)
{
    float originMagnitude = max(abs(sceneOriginWorld.x), max(abs(sceneOriginWorld.y), abs(sceneOriginWorld.z)));
    float coordinateLevel = max(0.0f, ceil(log2(max(originMagnitude /
        (sharc_cellSize * 536870912.0f), 1.0f))));
    return clamp(log2(max(length(position - InitOrigin()) * sharc_lodScale /
        sharc_cellSize, 1.0f)), coordinateLevel, (float)(SHARC_MAX_LEVEL - 1u));
}

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

// A slot's state from a bucket read, with constant indices only: an array indexed at run time
// is kept in local memory for the whole insert (3% of the training pass on bistro).
uint SharcBucketState(uint states[SHARC_BUCKET_SIZE], uint p)
{
    uint state = 0u;
    [unroll] for (uint q = 0u; q < SHARC_BUCKET_SIZE; ++q)
        if (q == p) state = states[q];
    return state;
}

bool SharcMergedNode(uint states[SHARC_BUCKET_SIZE], uint hash)
{
    uint sameKey = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        sameKey += states[p] == hash ? 1u : 0u;
    return sameKey >= SHARC_MERGE_RECORDS;
}

float SharcMergedWeight(SharcDescriptor d, SharcSurface s)
{
    return smoothstep(0.3f, 0.8f, dot(d.geometricNormal, s.geometricNormal));
}

float SharcSurfaceWeight(SharcDescriptor d, SharcSurface s, float3 relative, float size)
{
    float3 delta = relative - d.representative;
    float plane = max(abs(dot(delta, d.geometricNormal)), abs(dot(delta, s.geometricNormal)));
    float w = smoothstep(0.90f, 0.99f, dot(d.geometricNormal, s.geometricNormal));
    w *= smoothstep(SHARC_SIMILAR_NORMAL.x, SHARC_SIMILAR_NORMAL.y, dot(d.normal, s.normal));
    w *= 1.0f - smoothstep(0.015f * size, 0.05f * size, plane);
    w *= 1.0f - smoothstep(SHARC_SIMILAR_ROUGHNESS.x, SHARC_SIMILAR_ROUGHNESS.y, abs(s.roughness - d.roughness));

    float3 ratio = s.demodulator / d.demodulator;

    float ratioMax = max(ratio.x, max(ratio.y, ratio.z));
    float ratioMin = min(ratio.x, min(ratio.y, ratio.z));
    w *= 1.0f - smoothstep(SHARC_SIMILAR_ALBEDO.x, SHARC_SIMILAR_ALBEDO.y, log2(max(ratioMax, rcp(ratioMin))));
    return w;
}

void SharcInitializeEntry(uint e, int3 node, uint meta, SharcSurface s, float3 relative)
{

    const float size = SharcCellSize(meta & 31u);
    g_sharc.Store4(e + SHARC_QUERY, uint4(SharcCheckHash(node, meta, s.instance, s.material),
        PackNormal(s.geometricNormal), PackNormal(s.normal), SharcPackSnorm10x3(relative / size)));
    g_sharc.Store4(e + SHARC_QUERY + 16u, uint4(SharcPackLog10x3(s.demodulator), 0u,
        SharcPackUnorm8(s.roughness) << 16u, sharc_frame & 0xffffu));

    g_sharc.Store4(e + SHARC_NODE, uint4(asuint(node), meta));
    g_sharc.Store4(e + SHARC_NODE + 16u, uint4(s.instance, s.material, 0u, 0u));

    [unroll] for (uint b = SHARC_FRAME_RGB; b < SHARC_ENTRY_BYTES; b += 16u)
        g_sharc.Store4(e + b, 0u);
    g_sharc.Store(e + SHARC_LAST_UPDATE, sharc_frame);
    g_sharc.Store(e + SHARC_LAST_TOUCH, sharc_frame);
}

// Claim or reuse a cache entry while publishing complete state.
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
    bool contended = false;

    uint matchingSlots = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
    {
        if (states[p] == hash) matchingSlots |= 1u << p;
        else if (states[p] == SHARC_LOCKED) contended = true;
        else if (states[p] == 0u && slot == SHARC_INVALID) slot = bucket * SHARC_BUCKET_SIZE + p;
    }
    [loop] while (matchingSlots != 0u)
    {
        const uint p = (uint)firstbitlow(matchingSlots);
        matchingSlots &= matchingSlots - 1u;
        uint e = SharcEntryAddress(bucket * SHARC_BUCKET_SIZE + p);
        DeviceMemoryBarrier();

        uint4 key; uint2 idm;
        SharcLoadKey(e, key, idm);
        uint check; SharcDescriptor d; SharcHistory h;
        SharcLoadQueryRecord(e, size, check, d, h);
        if (!SharcKeyMatches(key, idm, node, meta, s)) continue;
        if (merged)
        {

            float w = SharcMergedWeight(d, s);
            if (w > bestWeight) { bestWeight = w; best = e; }
        }
        else
        {
            float w = SharcSurfaceWeight(d, s, relative, size);

            if (w > 0.25f)
            {
                g_sharc.Store(e + SHARC_LAST_TOUCH, sharc_frame);
                weight = w;
                return e;
            }
        }
    }
    if (best != SHARC_INVALID)
    {
        g_sharc.Store(best + SHARC_LAST_TOUCH, sharc_frame);
        weight = bestWeight;
        return best;
    }

    if (contended) return SHARC_INVALID;

    uint expected = 0u;
    if (slot == SHARC_INVALID)
    {
        // No slot is locked here (that returned as contended above).
        uint victimAge = 0u;
        [loop] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        {
            uint candidate = bucket * SHARC_BUCKET_SIZE + p;
            uint4 h = g_sharc.Load4(SharcEntryAddress(candidate) + SHARC_LAST_UPDATE);
            uint age = sharc_frame - h.x;
            if (h.w != sharc_frame && age > victimAge)
            {
                victimAge = age;
                slot = candidate;
            }
        }
        if (slot == SHARC_INVALID || victimAge < SHARC_REPLACE_AGE) return SHARC_INVALID;
        expected = SharcBucketState(states, slot - bucket * SHARC_BUCKET_SIZE);
    }
    uint stateAddress = SharcStateAddress(slot), old;
    g_sharc.InterlockedCompareExchange(stateAddress, expected, SHARC_LOCKED, old);
    if (old != expected) return SHARC_INVALID;
    uint e = SharcEntryAddress(slot);
    SharcInitializeEntry(e, node, meta, s, relative);
    DeviceMemoryBarrier();
    g_sharc.InterlockedExchange(stateAddress, hash, old);
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

// Accumulate weighted radiance and moments with fixed-point atomics.
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

    // Trust follows the sample count. The relative error only rejects an entry that is plainly
    // garbage: light-tree NEE keeps the per-sample variance so high that a tight error bound left
    // most entries unused, and the paths that should have ended in them went on to take their
    // own light samples instead.
    float confidence = smoothstep(0.5f * (float)sharc_minSamples, (float)sharc_minSamples, neff);
    confidence *= smoothstep(1.0f, 2.0f, (float)frames);
    confidence *= 1.0f - smoothstep(1.0f, 2.0f, error);
    return all(isfinite(float4(mean, variance, neff, confidence))) ? confidence : 0.0f;
}

float SharcAgeLimit(SharcHistory h)
{
    float interval = isfinite(h.interval) ? h.interval : 0.0f;

    float limit = clamp(16.0f * interval, (float)sharc_maxAge, 4.0f * sharc_maxAge);

    if (h.frames < 2u) limit = min(limit, max(0.25f * sharc_maxAge, 32.0f));
    return limit;
}

float SharcConfidence(SharcHistory h)
{
    float age = (float)(sharc_frame - h.lastUpdate);

    float ageLimit = SharcAgeLimit(h);
    float confidence = h.confidence * (1.0f - smoothstep(0.5f * ageLimit, ageLimit, age));
    return isfinite(confidence) ? saturate(confidence) : 0.0f;
}

void SharcQueryNode(SharcSurface s, uint level, uint axis, int3 node, float3 relative,
    bool ignoreConfidence, out float3 sum, out float support)
{
    uint meta = SharcMeta(s, level, axis);
    uint hash = SharcHash(node, meta, s.instance, s.material);
    uint bucket = SharcBucketOf(hash);
    uint states[SHARC_BUCKET_SIZE];
    SharcLoadBucket(bucket, states);
    const bool merged = SharcMergedNode(states, hash);
    const uint check = SharcCheckHash(node, meta, s.instance, s.material);
    const float size = SharcCellSize(level);
    sum = 0.0f; support = 0.0f;
    float geometricSupport = 0.0f;

    uint matchingSlots = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        if (states[p] == hash) matchingSlots |= 1u << p;
    [loop] while (matchingSlots != 0u)
    {
        const uint p = (uint)firstbitlow(matchingSlots);
        matchingSlots &= matchingSlots - 1u;
        uint e = SharcEntryAddress(bucket * SHARC_BUCKET_SIZE + p);

        uint recordCheck; SharcDescriptor d; SharcHistory h;
        SharcLoadQueryRecord(e, size, recordCheck, d, h);
        if (recordCheck != check) continue;
        float surfaceWeight = merged ? SharcMergedWeight(d, s)
            : SharcSurfaceWeight(d, s, relative, size);
        if (!isfinite(surfaceWeight) || surfaceWeight <= 0.0f) continue;
        geometricSupport += surfaceWeight;
        float confidence = ignoreConfidence ? (h.frames > 0u ? 1.0f : 0.0f) : SharcConfidence(h);

        if (confidence <= 0.0f || !all(isfinite(h.mean))) continue;
        sum += h.mean * (surfaceWeight * confidence);
        support += surfaceWeight * confidence;
    }
    float scale = rcp(max(geometricSupport, 1.0f));
    sum *= scale;
    support *= scale;
}

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

// Full angle of the circular cone of a diffuse lobe (SharcLobeConeAngle): the cosine lobe covers
// 1.5 pi, since the integral of (cos/pi)^2 over the hemisphere is 2/(3 pi) (pi/4 angle^2 = 1.5 pi).
static const float SHARC_DIFFUSE_CONE = 2.44948974f;

// Share of queries the cache may answer where the path's cone meets a surface. The cone is the
// lobes that led here (SharcLobeConeAngle), as its full angle and its width at the hit, so its apex
// lies width/angle back along the ray. Seen from there, one cell may fill at most 1/q of the cone's
// solid angle (q = sharc_queryFootprint), and so hold about that share of the lobe's light: the
// lobe then reaches other cells as well, and no single cell can show. The cell is the one a query
// reads, the coarser level's with four times the area as often as the query picks that level. It
// counts as a disk of that area facing the apex, whose solid angle is exact at any distance and
// never exceeds the hemisphere, not as the smaller disk a slant shows: a cone meeting the surface
// at a slant is stretched along it but no wider across it, so a cell wider than the cone shows
// across the slant however many cells the stretch reaches. (Counted with the slant, the narrow
// lobe of a 0.03 glass pane let the cache answer on the ground seen through it at a glance, and its
// cells showed as splotches.) At the default q = 2 a diffuse cone reaches the limit
// where the surface it meets lies closer than about half a cell: in a crevice or where two objects
// touch, the cell also holds lit surface the query cannot see. A short band around the limit mixes
// both answers, so the switch leaves no edge.
float SharcConeRamp(float coneWidth, float coneAngle, float3 position)
{
    if (!(coneAngle > 0.0f)) return 0.0f;
    const float coneSolidAngle = min(0.25f * PI * coneAngle * coneAngle, 2.0f * PI);
    const float apex = coneWidth / coneAngle;
    const float lod  = SharcLevel(position);
    const float size = sharc_cellSize * exp2(floor(lod));
    const float area = size * size * (1.0f + 3.0f * frac(lod));
    const float s    = sqrt(apex * apex + area * INV_PI);
    const float cellSolidAngle = 2.0f * area / (s * (s + apex));   // 2 pi (1 - apex / s)
    return smoothstep(0.8f, 1.25f, coneSolidAngle / (sharc_queryFootprint * cellSolidAngle));
}

// Draw a cache estimate using confidence-aware stochastic rejection.
bool SharcQueryStochastic(SharcSurface s, bool ignoreConfidence, inout uint seed, out float3 radiance)
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
        (f - float3(corner)) * SharcCellSize(level), ignoreConfidence, sum, support);
    if (support <= 1e-5f) return false;
    if (!ignoreConfidence && RandomFloatSingle(seed) >= support) return false;
    radiance = sum / support * s.demodulator;
    return all(isfinite(radiance));
}

bool SharcQueryDraws(SharcSurface s, inout uint seed, out float3 radiance)
{
    return SharcQueryStochastic(s, false, seed, radiance);
}
// Whether a training path may end in the cache here (SharcConeRamp). A share of the paths goes on
// past the cache either way, so that it does not only learn from itself.
bool SharcQueryFootprintAccepted(float3 position, float coneWidth, float coneAngle, inout uint seed)
{
    float footprint = SharcConeRamp(coneWidth, coneAngle, position);
    if (footprint <= 0.0f) return false;
    if (RandomFloatSingle(seed) >= footprint * (31.0f / 32.0f)) return false;
    return true;
}

// Accept cache history only when its footprint and confidence agree, for a diffuse cone of the
// given width.
bool SharcQuery(SharcSurface s, float coneWidth, inout uint seed, out float3 radiance)
{
    radiance = 0.0f;
    return SharcQueryFootprintAccepted(s.position, coneWidth, SHARC_DIFFUSE_CONE, seed) &&
        SharcQueryDraws(s, seed, radiance);
}

bool SharcQueryForced(SharcSurface s, inout uint seed, out float3 radiance)
{
    return SharcQueryStochastic(s, true, seed, radiance);
}

// Fold the frame accumulation of an entry into its history (resolve pass and tests).
void SharcResolveEntry(uint slot)
{
    uint state = g_sharc.Load(SharcStateAddress(slot));
    if (state == 0u || state == SHARC_LOCKED) return;
    uint e = SharcEntryAddress(slot);
    uint4 sumRG = g_sharc.Load4(e + SHARC_FRAME_RGB);
    uint4 sumBL = g_sharc.Load4(e + SHARC_FRAME_RGB + 16u);
    uint4 sumW = g_sharc.Load4(e + SHARC_FRAME_W_W2);

    if (all(sumRG == 0u) && all(sumBL == 0u) && all(sumW.xyz == 0u)) return;
    const float radianceScale = rcp(SHARC_RADIANCE_SCALE);
    const float weightScale = rcp(SHARC_WEIGHT_SCALE);
    float4 frame = float4(SharcFloat(sumRG.xy, radianceScale), SharcFloat(sumRG.zw, radianceScale),
        SharcFloat(sumBL.xy, radianceScale), (float)sumW.x * weightScale);
    float3 moments = float3(SharcFloat(sumBL.zw, radianceScale),
        (float)sumW.y * weightScale, (float)sumW.z * weightScale);
    if (frame.w > 0.0f)
    {
        float4 previous = float4(asfloat(g_sharc.Load3(e + SHARC_MEAN)), asfloat(g_sharc.Load(e + SHARC_HISTORY_W)));
        float3 oldMoments = float3(asfloat(g_sharc.Load(e + SHARC_HISTORY_L2)),
            asfloat(g_sharc.Load(e + SHARC_HISTORY_W2)), asfloat(g_sharc.Load(e + SHARC_HISTORY_POSITIVE)));
        bool validHistory = all(isfinite(previous)) && all(isfinite(oldMoments)) &&
            previous.w > 0.0f && oldMoments.y > 0.0f;
        // Corrupt history must not contaminate the next frame.
        if (!validHistory)
        {
            previous = 0.0f;
            oldMoments = 0.0f;
            g_sharc.Store(e + SHARC_FRAMES, 0u);
        }

        float decay = 1.0f - rcp(max((float)sharc_historyFrames, 2.0f));
        float oldWeight = previous.w * decay;
        float weight = oldWeight + frame.w;
        float oldFraction = oldWeight / weight;
        float3 mean = previous.xyz * oldFraction + frame.xyz / weight;
        float second = oldMoments.x * oldFraction + moments.x / weight;
        float weight2 = oldMoments.y * decay * decay + moments.y;
        float positives = oldMoments.z * decay + moments.z;
        if (all(isfinite(float4(mean, weight))) && all(isfinite(float3(second, weight2, positives))))
        {
            g_sharc.Store3(e + SHARC_MEAN, asuint(mean));
            g_sharc.Store4(e + SHARC_HISTORY_W, asuint(float4(weight, second, weight2, positives)));
            const uint frames = min(g_sharc.Load(e + SHARC_FRAMES) + 1u, 65535u);
            g_sharc.Store(e + SHARC_FRAMES, frames);
            float gap = (float)(sharc_frame - g_sharc.Load(e + SHARC_LAST_UPDATE));
            float interval = validHistory ? asfloat(g_sharc.Load(e + SHARC_INTERVAL)) : 1.0f;
            if (!isfinite(interval)) interval = 1.0f;
            interval = validHistory ? max(lerp(interval, gap, 0.125f), 0.5f * gap) : 1.0f;
            g_sharc.Store(e + SHARC_INTERVAL, asuint(interval));
            g_sharc.Store(e + SHARC_LAST_UPDATE, sharc_frame);

            const float confidence = SharcStatisticalConfidence(e);
            g_sharc.Store(e + SHARC_CONFIDENCE, asuint(confidence));

            SharcPublishQueryHistory(e, mean, confidence, frames, interval);
        }
    }

    g_sharc.Store4(e + SHARC_FRAME_RGB, 0u);
    g_sharc.Store4(e + SHARC_FRAME_RGB + 16u, 0u);
    g_sharc.Store4(e + SHARC_FRAME_W_W2, 0u);
}
