#pragma once
#include "Sharc_v8.hlsli"

static const uint GUIDE_INVALID = 0xffffffffu;
static const uint GUIDE_STREAM = 0x47554944u;
static const float GUIDE_PI = 3.14159265358979f;
static const float GUIDE_MIN_APERTURE = 1.0e-4f;
static const float GUIDE_SEED_APERTURE = 0.1f;
static const float GUIDE_NARROW = 0.25f;
static const float GUIDE_GATE_MARGIN = 0.02f;
static const float GUIDE_PROBATION_MASS = 0.05f;
static const float GUIDE_CONFIRM_SUPPORT = 16.0f;
static const float GUIDE_CONFIRM_SUPPORT_SLOW = 8.0f;
static const uint GUIDE_CONFIRM_EPOCHS = 24u;
static const float GUIDE_FIT_SUPPORT = 4.0f;
static const float GUIDE_HISTORY_DECAY = 0.957603f;
static const float GUIDE_HISTORY_CAP = 1024.0f;
static const float GUIDE_CLIP = 1.0e6f;

static const float GUIDE_PEAK_EXCESS = 1.546f;
static const float GUIDE_Q_FLOOR = 0.05f;
static const uint GUIDE_PROBATION_EPOCHS = 64u;
static const float GUIDE_MASS_SCALE = 4096.0f;
static const uint GUIDE_SUPPORT_SHIFT = 44u;
static const float GUIDE_SUPPORT_SCALE = 16.0f;
static const uint GUIDE_OPPORTUNITY_SCALE = 256u;

static const uint GUIDE_CELL = 0u;
static const uint GUIDE_META = 12u;
static const uint GUIDE_META_KEY_MASK = 0xffu;

static const uint GUIDE_META_WORTH = 0x100u;
static const uint GUIDE_EVIDENCE = 16u;
static const uint GUIDE_MEAN = 20u;
static const uint GUIDE_COUNTS = 24u;
static const uint GUIDE_LAST_TOUCH = 28u;
static const uint GUIDE_LOBE0 = 32u;
static const uint GUIDE_OPPORTUNITIES = 160u;
static const uint GUIDE_UNEXPLAINED = 168u;
static const uint GUIDE_CHALLENGER = 176u;
static const uint GUIDE_MOMENT0 = 192u;
static const uint GUIDE_MOMENT_BYTES = 48u;
static const uint GUIDE_MOMENT_PHI = 40u;
static const uint GUIDE_MOMENT_PEAK = 44u;

static const uint GUIDE_LOBE_VALID = 0x80000000u;
static const uint GUIDE_LOBE_MEASURED = 0x40000000u;

#define GUIDE_ENABLED ((guide_params & GUIDE_PARAM_ENABLED) != 0u)
#define GUIDE_QMAX min((float)((guide_params >> GUIDE_PARAM_QMAX_SHIFT) & 255u) * (1.0f / 255.0f), 0.9f)
#define GUIDE_LEVEL_OFFSET ((guide_params >> GUIDE_PARAM_LEVEL_SHIFT) & 7u)
#define GUIDE_FRESHNESS ((float)(((guide_params >> GUIDE_PARAM_FRESHNESS_SHIFT) & 255u) + 1u) * 8.0f)
#define GUIDE_TRAIN ((guide_params & GUIDE_PARAM_TRAIN) != 0u)

#define GUIDE_MAX_DEPTH ((int)((guide_params >> GUIDE_PARAM_DEPTH_SHIFT) & 7u))

uint GuideStateAddress(uint slot) { return SHARC_CACHE_BYTES + slot * 4u; }
uint GuideEntryAddress(uint slot) { return SHARC_CACHE_BYTES + GUIDE_STATE_BYTES + slot * GUIDE_ENTRY_BYTES; }
uint GuideSlotOf(uint e) { return (e - GuideEntryAddress(0u)) / GUIDE_ENTRY_BYTES; }
uint GuideDirtyAddress(uint word) { return GUIDE_DIRTY_OFFSET + word * 4u; }
uint GuideBucketOf(uint hash) { return hash & (GUIDE_CAPACITY / SHARC_BUCKET_SIZE - 1u); }
uint GuideLobeAddress(uint e, uint j) { return e + GUIDE_LOBE0 + j * 16u; }
uint GuideMomentAddress(uint e, uint j) { return e + GUIDE_MOMENT0 + j * GUIDE_MOMENT_BYTES; }

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

// Jittered keys distribute guide updates across neighboring cells.
GuideKey GuideKeyOf(float3 position, float3 geometricNormal, inout uint seed)
{
    GuideKey k;
    const uint level = GuideLevel(position);
    int3 base; float3 f;
    SharcGrid(position, level, base, f);
    float3 c = f - 0.5f;
    float3 u = float3(RandomFloatSingle(seed), RandomFloatSingle(seed), RandomFloatSingle(seed));
    k.cell = base + int3(sign(c)) * int3(u < abs(c));
    float3 w = SharcNormalWeights(geometricNormal);
    float r = RandomFloatSingle(seed);
    uint axis = r < w.x ? 0u : (r < w.x + w.y ? 1u : 2u);
    k.meta = level | ((axis * 2u + (geometricNormal[axis] < 0.0f ? 1u : 0u)) << 5u);
    return k;
}

GuideKey GuideParentKey(GuideKey k)
{
    GuideKey p;
    p.cell = k.cell >> 1;
    p.meta = min((k.meta & 31u) + 1u, SHARC_MAX_LEVEL - 1u) | (k.meta & (GUIDE_META_KEY_MASK & ~31u));
    return p;
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

uint GuideFind(GuideKey k)
{
    uint hash = GuideHash(k);
    uint bucket = GuideBucketOf(hash);
    uint states[SHARC_BUCKET_SIZE];
    GuideLoadBucket(bucket, states);

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
    return GUIDE_INVALID;
}

uint64_t GuideWord(uint2 w) { return ((uint64_t)w.y << 32u) | (uint64_t)w.x; }
uint2 GuideWords(uint64_t v) { return uint2((uint)v, (uint)(v >> 32u)); }
uint64_t GuideFixed(float v) { return (uint64_t)(min(max(v, 0.0f), 1.0e8f) * GUIDE_MASS_SCALE + 0.5f); }
uint64_t GuideFixedSigned(float v) { return (uint64_t)(int64_t)(clamp(v, -1.0e8f, 1.0e8f) * GUIDE_MASS_SCALE); }
float GuideUnsignedOf(uint64_t v) { return (float)v / GUIDE_MASS_SCALE; }
float GuideSignedOf(uint64_t v) { return (float)(int64_t)v / GUIDE_MASS_SCALE; }

uint64_t GuideMassWord(float mass, float support)
{
    return GuideFixed(mass) |
        ((uint64_t)(uint)(min(max(support, 0.0f), 65535.0f) * GUIDE_SUPPORT_SCALE + 0.5f) << GUIDE_SUPPORT_SHIFT);
}
float GuideMassOf(uint64_t v, out float support)
{
    support = (float)(v >> GUIDE_SUPPORT_SHIFT) / GUIDE_SUPPORT_SCALE;
    return (float)(v & (((uint64_t)1u << GUIDE_SUPPORT_SHIFT) - (uint64_t)1u)) / GUIDE_MASS_SCALE;
}

void GuideBasis(float3 n, out float3 t, out float3 b)
{
    float sg = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sg + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0f + sg * n.x * n.x * a, sg * c, -sg * n.x);
    b = float3(c, sg + n.y * n.y * a, -n.y);
}

float GuideCapPdf(float3 axis, float aperture, float3 dir)
{

    float3 dv = dir - axis;
    float d = 0.5f * dot(dv, dv);
    return max(0.0f, 1.0f - d / aperture) / (GUIDE_PI * aperture);
}

float3 GuideCapSample(float3 axis, float aperture, float2 u)
{
    float delta = aperture * u.x / (1.0f + sqrt(max(1.0f - u.x, 0.0f)));
    float radial = sqrt(max(delta * (2.0f - delta), 0.0f));
    float s, c;
    sincos(2.0f * GUIDE_PI * u.y, s, c);
    float3 t, b;
    GuideBasis(axis, t, b);
    return normalize((1.0f - delta) * axis + radial * (c * t + s * b));
}

float GuideGate(float aperture)
{
    return min(4.0f * aperture - 2.0f * aperture * aperture + GUIDE_GATE_MARGIN, 1.0f);
}

struct GuideSet
{
    uint2 cone[GUIDE_SLOTS];
    float weightSum;
    float q;
};

GuideSet GuideEmpty()
{
    GuideSet g;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c) g.cone[c] = uint2(0u, 0u);
    g.weightSum = 0.0f;
    g.q = 0.0f;
    return g;
}

uint4 GuideSlotWords(uint4 w0, uint4 w1, uint4 w2, uint4 w3, uint4 w4, uint4 w5, uint4 w6, uint4 w7, uint c)
{
    return c == 0u ? w0 : (c == 1u ? w1 : (c == 2u ? w2 : (c == 3u ? w3 :
        (c == 4u ? w4 : (c == 5u ? w5 : (c == 6u ? w6 : w7))))));
}

// Reject stale lobes before assembling the active guide mixture.
GuideSet GuideBuild(uint e, float3 x, float3 n, bool training = false)
{
    GuideSet g = GuideEmpty();

    if (e < GuideEntryAddress(0u) || e >= GuideEntryAddress(GUIDE_CAPACITY) || !(GUIDE_QMAX > 0.0f)) return g;

    const GuideKey k = GuideLoadKey(e);
    if (!training && (k.meta & GUIDE_META_WORTH) == 0u) return g;
    const uint4 header = g_sharc.Load4(e + GUIDE_EVIDENCE);
    const uint count = training ? (header.z >> 4u) & 15u : header.z & 15u;
    if (count == 0u) return g;

    const float freshness = exp2(-(float)(sharc_frame - header.x) / GUIDE_FRESHNESS);
    if (!(freshness > 1e-3f)) return g;
    const float3 center = GuideCellCenter(k);

    const uint4 w0 = g_sharc.Load4(GuideLobeAddress(e, 0u));
    uint4 w1 = 0u, w2 = 0u, w3 = 0u, w4 = 0u, w5 = 0u, w6 = 0u, w7 = 0u;
    if (count > 1u) w1 = g_sharc.Load4(GuideLobeAddress(e, 1u));
    if (count > 2u) { w2 = g_sharc.Load4(GuideLobeAddress(e, 2u)); w3 = g_sharc.Load4(GuideLobeAddress(e, 3u)); }
    if (count > 4u) { w4 = g_sharc.Load4(GuideLobeAddress(e, 4u)); w5 = g_sharc.Load4(GuideLobeAddress(e, 5u)); }
    if (count > 6u) { w6 = g_sharc.Load4(GuideLobeAddress(e, 6u)); w7 = g_sharc.Load4(GuideLobeAddress(e, 7u)); }
    [loop] for (uint c = 0u; c < count; ++c)
    {
        const uint4 w = GuideSlotWords(w0, w1, w2, w3, w4, w5, w6, w7, c);
        const float branch = f16tof32(w.y >> 16u) * freshness;
        if (!(branch > 0.0f)) continue;
        float3 axis = UnpackNormal(w.x);
        float aperture = f16tof32(w.y & 0xffffu);
        const float invDist = asfloat(w.z);
        if (invDist > 0.0f)
        {

            const float3 v = center + axis / invDist - x;
            const float r2 = dot(v, v);
            if (!(r2 > 1e-8f)) continue;
            axis = v * rsqrt(r2);
            aperture = aperture / (invDist * invDist * r2);
        }
        aperture = clamp(aperture, GUIDE_MIN_APERTURE, 2.0f);

        if (dot(axis, n) < -sqrt(max(aperture * (2.0f - aperture), 0.0f))) continue;
        const uint packedWeight = f32tof16(branch);
        const uint2 packed = uint2(PackNormal(axis), f32tof16(aperture) | (packedWeight << 16u));
        if (c == 0u)      g.cone[0] = packed;
        else if (c == 1u) g.cone[1] = packed;
        else if (c == 2u) g.cone[2] = packed;
        else if (c == 3u) g.cone[3] = packed;
        else if (c == 4u) g.cone[4] = packed;
        else if (c == 5u) g.cone[5] = packed;
        else if (c == 6u) g.cone[6] = packed;
        else              g.cone[7] = packed;
        g.weightSum += f16tof32(packedWeight);
    }
    g.q = min(g.weightSum, GUIDE_QMAX);
    if (!(g.q > 0.0f)) g.weightSum = 0.0f;
    return g;
}

bool GuideUsable(uint e, bool training)
{
    if (e < GuideEntryAddress(0u) || e >= GuideEntryAddress(GUIDE_CAPACITY)) return false;
    const uint counts = g_sharc.Load(e + GUIDE_COUNTS);
    return ((training ? counts >> 4u : counts) & 15u) != 0u;
}

GuideSet GuideBuildFrom(uint e, uint parent, float3 x, float3 n, bool training = false)
{
    return GuideBuild(GuideUsable(e, training) ? e : parent, x, n, training);
}

GuideSet GuideBuildAt(GuideKey key, float3 x, float3 n, bool training = false)
{
    const uint e = GuideFind(key);
    if (GuideUsable(e, training)) return GuideBuild(e, x, n, training);
    return GuideBuild(GuideFind(GuideParentKey(key)), x, n, training);
}

uint2 GuideConeAt(GuideSet g, uint c)
{
    return c == 0u ? g.cone[0] : (c == 1u ? g.cone[1] : (c == 2u ? g.cone[2] : (c == 3u ? g.cone[3] :
        (c == 4u ? g.cone[4] : (c == 5u ? g.cone[5] : (c == 6u ? g.cone[6] : g.cone[7]))))));
}

// Sample packed guide cones using their normalized mixture weights.
float3 GuideSample(GuideSet g, float uPick, float2 uDir, out uint pick)
{
    float u = uPick * g.weightSum;
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
    const uint2 cone = GuideConeAt(g, pick);
    return GuideCapSample(UnpackNormal(cone.x), f16tof32(cone.y & 0xffffu), uDir);
}
float3 GuideSample(GuideSet g, inout uint seed, out uint pick)
{
    const float  uPick = RandomFloatSingle(seed);
    const float2 uDir  = float2(RandomFloatSingle(seed), RandomFloatSingle(seed));
    return GuideSample(g, uPick, uDir, pick);
}

float GuidePdf(GuideSet g, float3 dir)
{
    float sum = 0.0f;
    [loop] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
    {
        const uint2 cone = GuideConeAt(g, c);
        const float w = f16tof32(cone.y >> 16u);
        if (!(w > 0.0f)) continue;
        sum += w * GuideCapPdf(UnpackNormal(cone.x), f16tof32(cone.y & 0xffffu), dir);
    }
    return sum / g.weightSum;
}

float GuideLambertPdf(float3 n, float3 dir)
{
    if (dot(dir, n) <= 0.0f) return 0.0f;
    return max(dot(n, dir), 0.0f) / GUIDE_PI;
}

// Correct the BSDF PDF with the guide mixture contribution.
float GuideMixPdf(GuideSet g, float pShare, float sharePdf, float3 dir, float bsdfPdf)
{
    if (!(g.q > 0.0f)) return bsdfPdf;
    return max(bsdfPdf + pShare * g.q * (GuidePdf(g, dir) - sharePdf), 0.0f);
}


void GuidePrepareEntry(uint index)
{
    uint stateAddress = GuideStateAddress(index);
    bool evict = (sharc_reset & 1u) != 0u;
    if (!evict && g_sharc.Load(stateAddress) != 0u)
    {
        uint e = GuideEntryAddress(index);
        GuideKey k = GuideLoadKey(e);
        float desired = SharcLevel(GuideCellCenter(k)) + (float)GUIDE_LEVEL_OFFSET;
        evict = (float)(sharc_frame - g_sharc.Load(e + GUIDE_LAST_TOUCH)) > (float)sharc_maxAge ||
            abs((float)(k.meta & 31u) - desired) > 3.0f;
    }

    if (evict) g_sharc.Store(stateAddress, 0u);
}

void GuideResolveEntry(uint slot)
{
    const uint state = g_sharc.Load(GuideStateAddress(slot));
    if (state == 0u || state == SHARC_LOCKED) return;
    const uint e = GuideEntryAddress(slot);
    const GuideKey k = GuideLoadKey(e);
    const uint4 header = g_sharc.Load4(e + GUIDE_EVIDENCE);
    const uint2 counts = g_sharc.Load2(e + GUIDE_OPPORTUNITIES);
    const float opportunities = (float)counts.x / (float)GUIDE_OPPORTUNITY_SCALE;
    const float bsdfTotal = (float)counts.y / (float)GUIDE_OPPORTUNITY_SCALE;
    float unexplained = GuideUnsignedOf(GuideWord(g_sharc.Load2(e + GUIDE_UNEXPLAINED)));
    const uint4 challengers = g_sharc.Load4(e + GUIDE_CHALLENGER);

    float3 axis[GUIDE_SLOTS], dsum[GUIDE_SLOTS];
    float aperture[GUIDE_SLOTS], invDist[GUIDE_SLOTS], mass[GUIDE_SLOTS], support[GUIDE_SLOTS], anchor[GUIDE_SLOTS], phi[GUIDE_SLOTS], peak[GUIDE_SLOTS];
    uint age[GUIDE_SLOTS];
    bool active[GUIDE_SLOTS], measured[GUIDE_SLOTS];
    float total = unexplained;
    [unroll] for (uint j = 0u; j < GUIDE_SLOTS; ++j)
    {
        const uint4 h = g_sharc.Load4(GuideLobeAddress(e, j));
        const uint m = GuideMomentAddress(e, j);
        const uint4 m0 = g_sharc.Load4(m);
        const uint4 m1 = g_sharc.Load4(m + 16u);
        const uint4 m2 = g_sharc.Load4(m + 32u);
        active[j] = (h.w & GUIDE_LOBE_VALID) != 0u;
        measured[j] = (h.w & GUIDE_LOBE_MEASURED) != 0u;
        age[j] = h.w & 255u;
        axis[j] = UnpackNormal(h.x);
        aperture[j] = f16tof32(h.y & 0xffffu);
        invDist[j] = asfloat(h.z);
        float s;
        mass[j] = GuideMassOf(GuideWord(m0.xy), s);
        support[j] = s;
        dsum[j] = float3(GuideSignedOf(GuideWord(m0.zw)), GuideSignedOf(GuideWord(m1.xy)), GuideSignedOf(GuideWord(m1.zw)));
        anchor[j] = GuideUnsignedOf(GuideWord(m2.xy));
        phi[j] = (float)m2.z / (float)GUIDE_OPPORTUNITY_SCALE;
        peak[j] = asfloat(m2.w);
        if (!active[j] || !all(isfinite(dsum[j])) || !isfinite(mass[j]) || !isfinite(anchor[j]) || !isfinite(peak[j]))
        {
            mass[j] = 0.0f; support[j] = 0.0f; dsum[j] = 0.0f; anchor[j] = 0.0f; phi[j] = 0.0f; peak[j] = 0.0f;
        }
        total += mass[j];
    }
    const float mean = opportunities > 16.0f ? total / opportunities : 0.0f;

    [unroll] for (uint f = 0u; f < GUIDE_SLOTS; ++f)
    {
        if (!active[f] || support[f] < GUIDE_FIT_SUPPORT || !(mass[f] > 0.0f)) continue;
        const float len = length(dsum[f]);
        if (len > 1e-20f) axis[f] = dsum[f] / len;
        const float resultant = min(len / mass[f], 1.0f);
        const float floorA = max(GUIDE_MIN_APERTURE, GUIDE_SEED_APERTURE * GUIDE_FIT_SUPPORT / support[f]);
        aperture[f] = clamp(3.0f * (1.0f - resultant), floorA, 2.0f);
        invDist[f] = anchor[f] / mass[f];
    }

    [unroll] for (uint p = 0u; p < GUIDE_SLOTS; ++p)
    {
        if (!active[p]) continue;
        [unroll] for (uint q = p + 1u; q < GUIDE_SLOTS; ++q)
        {
            if (!active[q]) continue;
            const float3 dv = axis[p] - axis[q];
            if (0.5f * dot(dv, dv) > 0.25f * max(aperture[p], aperture[q])) continue;

            if (min(aperture[p], aperture[q]) < 0.25f * max(aperture[p], aperture[q])) continue;
            mass[p] += mass[q]; support[p] += support[q]; dsum[p] += dsum[q]; anchor[p] += anchor[q]; phi[p] += phi[q];
            peak[p] = max(peak[p], peak[q]);
            measured[p] = measured[p] || measured[q];
            age[p] = max(age[p], age[q]);
            active[q] = false;
            mass[q] = 0.0f; support[q] = 0.0f; dsum[q] = 0.0f; anchor[q] = 0.0f; phi[q] = 0.0f; peak[q] = 0.0f;
            if (support[p] >= GUIDE_FIT_SUPPORT && mass[p] > 0.0f)
            {
                const float len = length(dsum[p]);
                if (len > 1e-20f) axis[p] = dsum[p] / len;
                aperture[p] = clamp(3.0f * (1.0f - min(len / mass[p], 1.0f)), GUIDE_MIN_APERTURE, 2.0f);
                invDist[p] = anchor[p] / mass[p];
            }
        }
    }

    uint measuredCount = 0u, activeCount = 0u;
    [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
    {
        if (!active[c]) continue;

        if (mass[c] > peak[c] && (support[c] >= GUIDE_CONFIRM_SUPPORT ||
            (support[c] >= GUIDE_CONFIRM_SUPPORT_SLOW && age[c] >= GUIDE_CONFIRM_EPOCHS))) measured[c] = true;
        bool drop = !measured[c] && age[c] >= GUIDE_PROBATION_EPOCHS;
        drop = drop || (measured[c] && age[c] >= 16u && mass[c] < 0.001f * total);
        if (drop)
        {
            active[c] = false;
            mass[c] = 0.0f; support[c] = 0.0f; dsum[c] = 0.0f; anchor[c] = 0.0f; phi[c] = 0.0f; peak[c] = 0.0f;
        }
    }

    float massFit[GUIDE_SLOTS];
    float fitted = unexplained;
    [unroll] for (uint g = 0u; g < GUIDE_SLOTS; ++g)
    {
        massFit[g] = active[g] ? max(min(mass[g], (mass[g] - peak[g]) * GUIDE_PEAK_EXCESS), 0.0f) : 0.0f;
        fitted += massFit[g];
    }

    float branch[GUIDE_SLOTS];
    float S = 0.0f, guidedMass = 0.0f, guidedShare = 0.0f, guidedPhi = 0.0f;
    const float bsdfScale = bsdfTotal > 0.0f ? rcp(bsdfTotal) : 0.0f;
    [unroll] for (uint s0 = 0u; s0 < GUIDE_SLOTS; ++s0)
    {
        branch[s0] = 0.0f;
        if (!active[s0] || !measured[s0] || !(fitted > 0.0f) || !(phi[s0] > 1.0f)) continue;
        const float share = min(phi[s0] * bsdfScale, 1.0f);
        guidedMass += massFit[s0] / fitted;
        guidedShare += share;
        guidedPhi += phi[s0];
        S = max(S, 1.0f - (massFit[s0] / fitted) / share - 2.0f * rsqrt(phi[s0]));
    }
    const float restPhi = bsdfTotal - guidedPhi;
    if (restPhi >= 8.0f && guidedShare < 0.98f)
        S = max(S, 1.0f - (1.0f - guidedMass) / (1.0f - guidedShare) - 2.0f * rsqrt(restPhi));
    S = clamp(S, 0.0f, 1.0f);
    float sumB = 0.0f;
    [unroll] for (uint s1 = 0u; s1 < GUIDE_SLOTS; ++s1)
    {
        if (!active[s1] || !measured[s1] || !(fitted > 0.0f) || !(phi[s1] > 1.0f)) continue;
        branch[s1] = max(massFit[s1] / fitted - (1.0f - S) * min(phi[s1] * bsdfScale, 1.0f), 0.0f);
        sumB += branch[s1];
    }

    [unroll] for (uint r = 0u; r < 2u; ++r)
    {
        const uint keyBits = r == 0u ? challengers.y : challengers.w;
        const uint dirBits = r == 0u ? challengers.x : challengers.z;
        if (keyBits == 0xffffffffu) continue;
        const float3 dir = UnpackNormal(dirBits);
        if (!(dot(dir, dir) > 0.5f)) continue;
        bool duplicate = false;
        uint target = GUIDE_INVALID;
        float targetSupport = 1e30f;
        [unroll] for (uint d = 0u; d < GUIDE_SLOTS; ++d)
        {
            if (!active[d])
            {
                if (target == GUIDE_INVALID) target = d;
                continue;
            }
            const float3 dv = dir - axis[d];
            const float dist = 0.5f * dot(dv, dv);
            if (dist <= 0.25f * aperture[d] || (aperture[d] <= GUIDE_NARROW && dist <= GuideGate(aperture[d])))
                duplicate = true;
        }
        if (duplicate) continue;
        if (target == GUIDE_INVALID)
        {
            [unroll] for (uint u = 0u; u < GUIDE_SLOTS; ++u)
            {
                if (measured[u] || !(support[u] < targetSupport)) continue;
                targetSupport = support[u];
                target = u;
            }
        }
        if (target == GUIDE_INVALID)
        {
            float weakest = 0.02f * total;
            [unroll] for (uint v = 0u; v < GUIDE_SLOTS; ++v)
            {
                if (!(mass[v] < weakest)) continue;
                weakest = mass[v];
                target = v;
            }
        }
        if (target == GUIDE_INVALID) continue;
        active[target] = true;
        measured[target] = false;
        age[target] = 0u;
        axis[target] = dir;
        aperture[target] = GUIDE_SEED_APERTURE;
        invDist[target] = 0.0f;
        mass[target] = 0.0f; support[target] = 0.0f; dsum[target] = 0.0f; anchor[target] = 0.0f; phi[target] = 0.0f;
        peak[target] = 0.0f;
        branch[target] = 0.0f;
    }

    uint probation = 0u;
    [unroll] for (uint n0 = 0u; n0 < GUIDE_SLOTS; ++n0)
        if (active[n0] && !measured[n0]) ++probation;
    const float qMax = GUIDE_QMAX;
    const float exploration = probation > 0u ? min(GUIDE_PROBATION_MASS, qMax) : 0.0f;
    const float measuredBudget = max(qMax - exploration, 0.0f);
    const float measuredScale = sumB > measuredBudget && sumB > 0.0f ? measuredBudget / sumB : 1.0f;
    float guided = 0.0f;
    [unroll] for (uint n1 = 0u; n1 < GUIDE_SLOTS; ++n1)
    {
        if (!active[n1]) continue;
        if (measured[n1]) { branch[n1] *= measuredScale; guided += branch[n1]; ++measuredCount; }
        else branch[n1] = exploration / (float)probation;
        ++activeCount;
    }
    const bool worth = guided >= GUIDE_Q_FLOOR;

    float scale = GUIDE_HISTORY_DECAY;
    float keptOpportunities = opportunities * scale;
    if (keptOpportunities > GUIDE_HISTORY_CAP)
    {
        scale *= GUIDE_HISTORY_CAP / keptOpportunities;
        keptOpportunities = GUIDE_HISTORY_CAP;
    }
    unexplained *= scale;
    const float keptBsdf = bsdfTotal * scale;

    uint o = 0u;
    [unroll] for (uint pass = 0u; pass < 2u; ++pass)
    {
        [unroll] for (uint w = 0u; w < GUIDE_SLOTS; ++w)
        {
            if (!active[w] || measured[w] != (pass == 0u)) continue;
            const uint flags = GUIDE_LOBE_VALID | (measured[w] ? GUIDE_LOBE_MEASURED : 0u) | min(age[w] + 1u, 255u);
            g_sharc.Store4(GuideLobeAddress(e, o), uint4(PackNormal(axis[w]),
                f32tof16(aperture[w]) | (f32tof16(branch[w]) << 16u), asuint(invDist[w]), flags));
            const uint m = GuideMomentAddress(e, o);
            g_sharc.Store4(m, uint4(GuideWords(GuideMassWord(mass[w] * scale, support[w] * scale)),
                GuideWords(GuideFixedSigned(dsum[w].x * scale))));
            g_sharc.Store4(m + 16u, uint4(GuideWords(GuideFixedSigned(dsum[w].y * scale)),
                GuideWords(GuideFixedSigned(dsum[w].z * scale))));
            g_sharc.Store4(m + 32u, uint4(GuideWords(GuideFixed(anchor[w] * scale)),
                (uint)(phi[w] * scale * (float)GUIDE_OPPORTUNITY_SCALE + 0.5f), asuint(peak[w] * scale)));
            ++o;
        }
    }
    [loop] for (; o < GUIDE_SLOTS; ++o)
    {
        g_sharc.Store4(GuideLobeAddress(e, o), 0u);
        const uint m = GuideMomentAddress(e, o);
        g_sharc.Store4(m, 0u);
        g_sharc.Store4(m + 16u, 0u);
        g_sharc.Store4(m + 32u, 0u);
    }
    g_sharc.Store4(e + GUIDE_EVIDENCE, uint4(sharc_frame, asuint(mean), measuredCount | (activeCount << 4u), header.w));
    g_sharc.Store(e + GUIDE_META, (k.meta & GUIDE_META_KEY_MASK) | (worth ? GUIDE_META_WORTH : 0u));
    g_sharc.Store2(e + GUIDE_OPPORTUNITIES, uint2((uint)(keptOpportunities * (float)GUIDE_OPPORTUNITY_SCALE + 0.5f),
        (uint)(keptBsdf * (float)GUIDE_OPPORTUNITY_SCALE + 0.5f)));
    g_sharc.Store2(e + GUIDE_UNEXPLAINED, GuideWords(GuideFixed(unexplained)));
    g_sharc.Store4(e + GUIDE_CHALLENGER, 0xffffffffu);
    g_sharc.InterlockedAnd(GuideDirtyAddress(slot >> 5u), ~(1u << (slot & 31u)));
}

float4 GuideDebugColor(float3 position, float3 geometricNormal, float3 normal, bool axisView)
{
    const GuideKey key = GuideKeyCentre(position, geometricNormal);
    const uint e = GuideFind(key);
    if (e == GUIDE_INVALID) return float4(0.45f, 0.025f, 0.35f, 0.0f);
    const uint active = (g_sharc.Load(e + GUIDE_COUNTS) >> 4u) & 15u;
    if (active == 0u && !GuideUsable(GuideFind(GuideParentKey(key)), true))
        return float4(0.65f, 0.32f, 0.025f, 0.0f);
    const GuideSet g = GuideBuildAt(key, position, normal, true);
    if (axisView)
    {
        float best = 0.0f;
        float3 color = 0.0f;
        [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
        {
            const float w = f16tof32(g.cone[c].y >> 16u);
            if (w > best) { best = w; color = abs(UnpackNormal(g.cone[c].x)); }
        }
        return float4(color * (0.15f + 0.85f * saturate(g.q / max(GUIDE_QMAX, 1e-3f))), 0.0f);
    }
    const GuideSet measured = GuideBuildAt(key, position, normal, false);
    const float q = GUIDE_QMAX > 0.0f ? saturate(measured.q / GUIDE_QMAX) : 0.0f;
    return float4(0.06f, 0.06f + 0.9f * q, 0.06f + 0.4f * (float)active / (float)GUIDE_SLOTS, 0.0f);
}

// Maintenance shared by the prepare pass and the tests: reset the dirty words and guide entries,
// evict aged or relocated cache entries. Returns the instance of an entry that survives so the
// caller can evict it when the instance moved, or SHARC_INVALID.
uint SharcPrepareIndex(uint index)
{
    if ((sharc_reset & 1u) != 0u && index < SHARC_DIRTY_WORDS)
        g_sharc.Store(SharcDirtyAddress(index), 0u);
    if ((sharc_reset & 1u) != 0u && index < GUIDE_DIRTY_WORDS)
        g_sharc.Store(GuideDirtyAddress(index), 0u);

    if (index < GUIDE_CAPACITY) GuidePrepareEntry(index);
    if (index >= SHARC_CAPACITY) return SHARC_INVALID;
    const uint stateAddress = SharcStateAddress(index);
    const uint e = SharcEntryAddress(index);
    bool evict = (sharc_reset & 1u) != 0u;
    if (!evict && g_sharc.Load(stateAddress) != 0u)
    {
        const uint4 key = g_sharc.Load4(e + SHARC_NODE);
        const SharcHistory h = SharcLoadHistory(e);
        const uint level = key.w & 31u;
        const float3 position = SharcNodeLocal(asint(key.xyz), level);
        const float desired = SharcLevel(position);
        evict = (float)(sharc_frame - h.lastUpdate) > SharcAgeLimit(h) || abs((float)level - desired) > 3.0f;
        if (!evict) return g_sharc.Load(e + SHARC_NODE + 16u);
    }
    if (evict) g_sharc.Store(stateAddress, 0u);
    return SHARC_INVALID;
}

void SharcEvictEntry(uint index) { g_sharc.Store(SharcStateAddress(index), 0u); }
