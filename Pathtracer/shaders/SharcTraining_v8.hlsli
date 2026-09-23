#pragma once
// Cache and guide training, used by the training pass and its tests only: the propagation state
// a training path carries, deposits, guide entry insertion and observation, and the per-lane guide
// roots. The propagation state spills to the path-state buffer (see PathStateLayout.h).
#include "SharcGuide_v8.hlsli"


#define SHARC_PROPAGATION_DEPTH 2u

struct SharcTrainingState
{
    uint address[SHARC_PROPAGATION_DEPTH];
    float splat[SHARC_PROPAGATION_DEPTH];
    uint spill;
    uint count;

    float suffixLuma;

    uint fresh;
};

uint SharcTrainingSpillAddress(uint lane) { return ps_addr_trainSpill(lane); }
float3 SharcTrainingRadianceOf(SharcTrainingState s, uint i)
{
    return asfloat(g_pathStateBuffer.Load3(s.spill + i * 24u));
}
void SharcTrainingSetRadiance(inout SharcTrainingState s, uint i, float3 v)
{
    g_pathStateBuffer.Store3(s.spill + i * 24u, asuint(v));
}
float3 SharcTrainingWeightOf(SharcTrainingState s, uint i)
{
    return asfloat(g_pathStateBuffer.Load3(s.spill + i * 24u + 12u));
}
void SharcTrainingSetWeight(inout SharcTrainingState s, uint i, float3 v)
{
    g_pathStateBuffer.Store3(s.spill + i * 24u + 12u, asuint(v));
}

void SharcTrainingInit(out SharcTrainingState state, uint spillPixel = 0u)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
    {
        state.address[i] = SHARC_INVALID;
        state.splat[i] = 0.0f;
    }
    state.spill = SharcTrainingSpillAddress(spillPixel);
    state.count = 0u;
    state.suffixLuma = 1.0f;
    state.fresh = SHARC_INVALID;
}

void SharcTrainingRadiance(inout SharcTrainingState state, float3 value)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count)
            SharcTrainingSetRadiance(state, i, SharcTrainingRadianceOf(state, i) + SharcTrainingWeightOf(state, i) * value);
}

void SharcTrainingRadianceSplit(inout SharcTrainingState state, float3 full, float3 diffuseOnly)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count)
            SharcTrainingSetRadiance(state, i, SharcTrainingRadianceOf(state, i) +
                SharcTrainingWeightOf(state, i) * (i == state.fresh ? diffuseOnly : full));
}

void SharcTrainingScatter(inout SharcTrainingState state, float3 weight)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count) SharcTrainingSetWeight(state, i, SharcTrainingWeightOf(state, i) * weight);
}

void SharcTrainingVertex(inout SharcTrainingState state, SharcSurface surface, float layerTransmission, uint seed)
{
    if (state.count >= SHARC_PROPAGATION_DEPTH) return;
    uint address; float splat;
    if (!SharcAllocateDeposit(surface, seed, address, splat)) return;

    bool duplicate = false;
    [unroll] for (uint j = 0u; j < SHARC_PROPAGATION_DEPTH; ++j)
        if (j < state.count && state.address[j] == address) duplicate = true;
    if (duplicate) return;
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
    {
        if (i != state.count) continue;
        state.address[i] = address;
        state.splat[i] = splat;
        SharcTrainingSetRadiance(state, i, 0.0f);

        SharcTrainingSetWeight(state, i, rcp(surface.demodulator * max(layerTransmission, 1e-3f)));
    }
    state.fresh = state.count;
    ++state.count;
    state.suffixLuma = 1.0f;
}

float SharcTrainingSurvival(SharcTrainingState state, float3 scatterWeight)
{
    return clamp(state.suffixLuma * Luma(scatterWeight), 0.1f, 1.0f);
}

// Advance training throughput after BSDF and roulette weighting.
void SharcTrainingAdvance(inout SharcTrainingState state, float3 full, float3 diffuseOnly, float rrWeight)
{
    state.suffixLuma *= Luma(full);
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count)
            SharcTrainingSetWeight(state, i, SharcTrainingWeightOf(state, i) * (i == state.fresh ? diffuseOnly : full) * rrWeight);
    state.fresh = SHARC_INVALID;
}

void SharcTrainingCommit(SharcTrainingState state)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count) SharcAccumulate(state.address[i], SharcTrainingRadianceOf(state, i), state.splat[i]);
}

void GuideInitializeEntry(uint e, GuideKey k)
{
    g_sharc.Store4(e + GUIDE_CELL, uint4(asuint(k.cell), k.meta));
    [unroll] for (uint b = 16u; b < GUIDE_ENTRY_BYTES; b += 16u)
        g_sharc.Store4(e + b, 0u);
    g_sharc.Store4(e + GUIDE_CHALLENGER, 0xffffffffu);
    g_sharc.Store(e + GUIDE_EVIDENCE, sharc_frame);
    g_sharc.Store(e + GUIDE_LAST_TOUCH, sharc_frame);
}

// Bucket locks publish initialized guide entries atomically.
uint GuideFindOrInsert(GuideKey k)
{
    uint hash = GuideHash(k);
    uint bucket = GuideBucketOf(hash);
    uint states[SHARC_BUCKET_SIZE];
    GuideLoadBucket(bucket, states);
    uint slot = GUIDE_INVALID;
    bool contended = false;

    uint matchingSlots = 0u;
    [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
    {
        if (states[p] == hash) matchingSlots |= 1u << p;
        else if (states[p] == SHARC_LOCKED) contended = true;
        else if (states[p] == 0u && slot == GUIDE_INVALID) slot = bucket * SHARC_BUCKET_SIZE + p;
    }
    [loop] while (matchingSlots != 0u)
    {
        const uint p = (uint)firstbitlow(matchingSlots);
        matchingSlots &= matchingSlots - 1u;
        uint e = GuideEntryAddress(bucket * SHARC_BUCKET_SIZE + p);
        DeviceMemoryBarrier();
        if (GuideKeyMatches(GuideLoadKey(e), k))
        {
            g_sharc.Store(e + GUIDE_LAST_TOUCH, sharc_frame);
            return e;
        }
    }

    if (contended) return GUIDE_INVALID;
    uint expected = 0u;
    if (slot == GUIDE_INVALID)
    {
        // No slot is locked here (that returned as contended above).
        uint victimAge = 0u;
        [loop] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        {
            uint candidate = bucket * SHARC_BUCKET_SIZE + p;
            uint age = sharc_frame - g_sharc.Load(GuideEntryAddress(candidate) + GUIDE_LAST_TOUCH);
            if (age > victimAge)
            {
                victimAge = age;
                slot = candidate;
            }
        }
        if (slot == GUIDE_INVALID || victimAge < SHARC_REPLACE_AGE) return GUIDE_INVALID;
        expected = SharcBucketState(states, slot - bucket * SHARC_BUCKET_SIZE);
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

// Accumulate explained mass and challenger directions atomically.
void GuideObserve(uint e, float3 wRef, float invDist, float w, float ratio, uint seed)
{

    if (e < GuideEntryAddress(0u) || e >= GuideEntryAddress(GUIDE_CAPACITY)) return;
    if (!isfinite(w) || w < 0.0f || !all(isfinite(wRef))) return;
    if (!(invDist > 0.0f) || !isfinite(invDist)) invDist = 0.0f;
    if (!(ratio > 0.0f) || !isfinite(ratio)) ratio = 0.0f;
    const uint count = (g_sharc.Load(e + GUIDE_COUNTS) >> 4u) & 15u;

    const float wFit = min(w, GUIDE_CLIP);
    uint best = GUIDE_INVALID;
    float bestD = 1e30f, bestA = 0.0f;
    [loop] for (uint j = 0u; j < count; ++j)
    {
        const uint2 lw = g_sharc.Load2(GuideLobeAddress(e, j));
        const float3 dv = wRef - UnpackNormal(lw.x);
        const float d = 0.5f * dot(dv, dv);
        if (d < bestD) { bestD = d; best = j; bestA = f16tof32(lw.y & 0xffffu); }
    }
    const bool explained = best != GUIDE_INVALID && bestD <= GuideGate(bestA);

    const bool narrow = explained && bestA <= GUIDE_NARROW && bestD <= bestA;
    const uint share = (uint)(min(ratio, 64.0f) * (float)GUIDE_OPPORTUNITY_SCALE + 0.5f);
    g_sharc.InterlockedAdd64(e + GUIDE_OPPORTUNITIES,
        ((uint64_t)share << 32u) | (uint64_t)GUIDE_OPPORTUNITY_SCALE);
    if (explained)
    {
        const uint m = GuideMomentAddress(e, best);
        if (share != 0u) g_sharc.InterlockedAdd(m + GUIDE_MOMENT_PHI, share);
        if (wFit > 0.0f)
        {
            g_sharc.InterlockedAdd64(m, GuideMassWord(wFit, 1.0f));
            g_sharc.InterlockedAdd64(m + 8u, GuideFixedSigned(wFit * wRef.x));
            g_sharc.InterlockedAdd64(m + 16u, GuideFixedSigned(wFit * wRef.y));
            g_sharc.InterlockedAdd64(m + 24u, GuideFixedSigned(wFit * wRef.z));
            if (invDist > 0.0f) g_sharc.InterlockedAdd64(m + 32u, GuideFixed(wFit * invDist));
            g_sharc.InterlockedMax(m + GUIDE_MOMENT_PEAK, asuint(wFit));
        }
    }
    else if (wFit > 0.0f) g_sharc.InterlockedAdd64(e + GUIDE_UNEXPLAINED, GuideFixed(wFit));
    if (w > 0.0f && !narrow)
    {

        uint s = seed ^ Hash32(e);
        const uint dirBits = PackNormal(wRef);
        [unroll] for (uint r = 0u; r < 2u; ++r)
        {
            const float key = -log(max(RandomFloatSingle(s), 1e-7f)) / w;
            g_sharc.InterlockedMin64(e + GUIDE_CHALLENGER + r * 8u,
                ((uint64_t)asuint(key) << 32u) | (uint64_t)dirBits);
        }
    }
    const uint slot = GuideSlotOf(e);
    g_sharc.InterlockedOr(GuideDirtyAddress(slot >> 5u), 1u << (slot & 31u));
}


static const uint GUIDE_ROOTS = 2u;
static const uint GUIDE_ROOT_HIT = 0x80000000u;

void GuideRootOpen(uint lane, uint r, uint entry, uint parent, float3 dir)
{
    const uint a = ps_addr_guideRoot(lane, r);
    g_pathStateBuffer.Store4(a, uint4(0u, 0u, entry, parent));
    g_pathStateBuffer.Store4(a + 16u, uint4(asuint(dir), 0u));
}

void GuideRootSetWeight(uint lane, uint r, float3 B, float ratio)
{
    const uint a = ps_addr_guideRoot(lane, r);
    g_pathStateBuffer.Store(a, PackRGB9E5(B));
    g_pathStateBuffer.Store(a + 28u, asuint(ratio));
}

void GuideRootsAddSource(uint lane, uint active, float3 A)
{
    if (!any(A > 0.0f)) return;
    [unroll] for (uint r = 0u; r < GUIDE_ROOTS; ++r)
    {
        if ((active & (1u << r)) == 0u) continue;
        const uint a = ps_addr_guideRoot(lane, r);
        const uint2 bc = g_pathStateBuffer.Load2(a);
        const float c = asfloat(bc.y) + Luma(UnpackRGB9E5(bc.x) * A);
        g_pathStateBuffer.Store(a + 4u, asuint(c));
    }
}

void GuideRootsScale(uint lane, uint active, float3 T)
{
    [unroll] for (uint r = 0u; r < GUIDE_ROOTS; ++r)
    {
        if ((active & (1u << r)) == 0u) continue;
        const uint a = ps_addr_guideRoot(lane, r);
        g_pathStateBuffer.Store(a, PackRGB9E5(UnpackRGB9E5(g_pathStateBuffer.Load(a)) * T));
    }
}

void GuideRootAim(uint lane, uint r, bool hit, float3 hitPosition, float3 dir)
{
    const uint a = ps_addr_guideRoot(lane, r);
    if (hit)
    {

        g_pathStateBuffer.Store(a + 28u, g_pathStateBuffer.Load(a + 28u) | GUIDE_ROOT_HIT);
        g_pathStateBuffer.Store3(a + 16u, asuint(hitPosition));
    }
    else g_pathStateBuffer.Store3(a + 16u, asuint(dir));
}

void GuideRootObserve(uint entry, bool hit, float3 endpoint, float w, float ratio, uint seed)
{
    if (entry < GuideEntryAddress(0u) || entry >= GuideEntryAddress(GUIDE_CAPACITY)) return;
    float3 wRef = endpoint;
    float invDist = 0.0f;
    if (hit)
    {
        const float3 v = endpoint - GuideCellCenter(GuideLoadKey(entry));
        const float d2 = dot(v, v);
        if (!(d2 > 1e-8f) || !isfinite(d2)) return;
        invDist = rsqrt(d2);
        wRef = v * invDist;
    }
    GuideObserve(entry, wRef, invDist, w, ratio, seed);
}

void GuideRootsFinalize(uint lane, uint active, uint seed)
{
    [unroll] for (uint r = 0u; r < GUIDE_ROOTS; ++r)
    {
        if ((active & (1u << r)) == 0u) continue;
        const uint a = ps_addr_guideRoot(lane, r);
        const uint4 w = g_pathStateBuffer.Load4(a);
        const uint4 t = g_pathStateBuffer.Load4(a + 16u);
        const bool hit = (t.w & GUIDE_ROOT_HIT) != 0u;
        const float ratio = asfloat(t.w & ~GUIDE_ROOT_HIT);
        const uint s = seed + r * 0x68e31da4u;
        GuideRootObserve(w.z, hit, asfloat(t.xyz), asfloat(w.y), ratio, s);
        GuideRootObserve(w.w, hit, asfloat(t.xyz), asfloat(w.y), ratio, s ^ 0x5bd1e995u);
    }
}
