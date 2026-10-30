// Tessellates the selected quadtree tiles straight into the global vertex buffer.

#include "OceanLayout.h"

cbuffer OceanPush : register(b0) {
    uint gTileCount;
    uint gU1;
    uint gU2;
    uint gU3;
    float gTime;
    float gDt;
    float gF2;
    float gF3;
};

SamplerState g_sampler : register(s0);

#include "Ocean_v8.hlsli"

struct OceanVertexOut {
    float3 vertex;
    uint packedNormal;
    half2 texCoord;
};

// Must match UnpackNormal_INT (Compression_v8.hlsli): octahedral, 2 x signed 16-bit.
uint OceanPackNormal(float3 n) {
    const float3 p = n / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20f);
    float2 f = p.xy;
    if (p.z < 0.0f) {
        f = float2((1.0f - abs(p.y)) * (p.x >= 0.0f ? 1.0f : -1.0f),
                   (1.0f - abs(p.x)) * (p.y >= 0.0f ? 1.0f : -1.0f));
    }
    const int ix = (int)(clamp(f.x, -1.0f, 1.0f) * 32767.0f);
    const int iy = (int)(clamp(f.y, -1.0f, 1.0f) * 32767.0f);
    return ((uint)(iy & 0xFFFF) << 16) | (uint)(ix & 0xFFFF);
}

// One thread per vertex, one group row per tile.
[numthreads(64, 1, 1)]
void OceanTiles(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OCEAN_TILE_VERTS || tid.y >= gTileCount)
        return;

    StructuredBuffer<OceanTileGPU> tiles = ResourceDescriptorHeap[OCEAN_SRV_TILES];
    RWStructuredBuffer<OceanVertexOut> verts = ResourceDescriptorHeap[OCEAN_UAV_VERTS];

    const OceanTileGPU tile = tiles[tid.y];
    const OceanParamsGPU P = OceanParams();
    const uint G = OCEAN_TILE_GRID;
    const uint i = tid.x % OCEAN_TILE_EDGE_VERTS;
    const uint j = tid.x / OCEAN_TILE_EDGE_VERTS;

    const float step = tile.size / (float)G;
    const OceanVertexSource src = OceanStitchSource(i, j, tile.stitch);
    const float2 localLo = float2(src.lo) * step;
    const float2 localHi = float2(src.hi) * step;
    const float2 local = float2(i, j) * step; // == lerp(localLo, localHi, src.w)

    const float2 qLo = tile.anchor.xz + localLo;
    OceanSample s = OceanSampleSurface(qLo, OceanGeometryWidth(qLo, P));
    if (src.w > 0.0f) {
        // Lerp the coarser edge's samples so the vertex lies on it.
        const float2 qHi = tile.anchor.xz + localHi;
        const OceanSample b = OceanSampleSurface(qHi, OceanGeometryWidth(qHi, P));
        s.displacement = lerp(s.displacement, b.displacement, src.w);
        s.gradient = lerp(s.gradient, b.gradient, src.w);
        s.stretch = lerp(s.stretch, b.stretch, src.w);
    }

    // Lerp the drop too, to stay on the neighbour's straight edge.
    const float curveLo = tile.curveBase + dot(tile.curveGrad, localLo) + dot(localLo, localLo) * 0.5f * tile.invCurveRadius;
    const float curveHi = tile.curveBase + dot(tile.curveGrad, localHi) + dot(localHi, localHi) * 0.5f * tile.invCurveRadius;
    const float curve = lerp(curveLo, curveHi, src.w);
    const float2 curveSlope = tile.curveGrad + local * tile.invCurveRadius;

    OceanVertexOut v;
    v.vertex = float3(local.x + s.displacement.x, s.displacement.y - curve, local.y + s.displacement.z);
    // Replaced at the hit; must pass EvalSurfaceStateImpl's interpolation guard.
    v.packedNormal = OceanPackNormal(OceanNormalFromSample(s, curveSlope));
    // Unit per tile; the hit recovers the undisplaced position from it.
    v.texCoord = (half2)(float2(i, j) / (float)G);

    verts[tile.vertexBase + tid.x] = v;
}

// Whitecaps, after Crest's foam simulation (wave-harmonic/crest, MIT).
OceanFoamState OceanLoadFoamState(RWByteAddressBuffer stats) {
    return stats.Load<OceanFoamState>(OCEAN_FOAM_STATE_OFFSET);
}

// Whitecap share of this cover, as foamCover counts it; lace excluded.
float OceanFoamWhite(float cover) {
    return saturate((cover - 0.3f) / 0.4f);
}

// Foam yield along a breaking crest's path, mean 1: downwind streak noise.
float OceanFoamStreaks(float2 world, float texel, float2 wind) {
    const float2 w = OceanFoamFrame(world, wind);
    float g = 0.0f, amp = 1.0f, total = 0.0f, across = 1.2f;
    [unroll] for (uint o = 0u; o < 3u; ++o) {
        const float keep = saturate(2.0f - 4.0f * texel / across);
        g += amp * keep * OceanFoamGrad(float2(w.x / (5.0f * across), w.y / across), 111u + o);
        total += amp;
        amp *= 0.6f;
        across *= 0.5f;
    }
    return max(0.0f, 1.0f + 1.6f * g / total);
}

[numthreads(8, 8, 1)]
void OceanFoam(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OCEAN_FOAM_SIZE || tid.y >= OCEAN_FOAM_SIZE)
        return;
    const OceanParamsGPU P = OceanParams();
    const uint level = tid.z;
    const bool first = (P.foamParity & 1u) == 0u;
    RWTexture2DArray<float> dst = ResourceDescriptorHeap[first ? OCEAN_UAV_FOAM0_MIPS : OCEAN_UAV_FOAM1_MIPS];
    Texture2DArray<float> prev = ResourceDescriptorHeap[first ? OCEAN_SRV_FOAM1 : OCEAN_SRV_FOAM0];
    RWByteAddressBuffer stats = ResourceDescriptorHeap[OCEAN_UAV_FOAM_STATS];
    if (P.foamStrength <= 0.0f) {
        dst[tid] = 0.0f;
        return;
    }

    const float texel = OceanFoamTexel(level);
    const float4 at = P.foamLevel[level];
    const float2 q = at.xy + (float2(tid.xy) + 0.5f) * texel;
    float amount = 0.0f;
    if (P.foamHistory != 0u) {
        const int2 src = int2(tid.xy) + int2(at.zw);
        if (all(src >= 0) && all(src < OCEAN_FOAM_SIZE)) {
            amount = prev.Load(int4(src, level, 0));
        } else if (level + 1u < OCEAN_FOAM_LEVELS) {
            // Scrolled in: read the next coarser level.
            const float4 coarse = P.foamLevel[level + 1u];
            const float coarseTexel = 2.0f * texel;
            const int2 c = int2(floor((q - (coarse.xy - coarse.zw * coarseTexel)) / coarseTexel));
            if (all(c >= 0) && all(c < OCEAN_FOAM_SIZE))
                amount = prev.Load(int4(c, level + 1u, 0));
        }
    }
    const float fresh = smoothstep(0.05f, 0.4f, amount);
    amount *= exp(-P.foamDecayRate * gDt * lerp(1.0f, OCEAN_FOAM_FRESH_DECAY, fresh));

    const OceanSample s = OceanSampleSurface(q, max(OCEAN_FOAM_BREAK_WIDTH, texel));
    const float3 A = s.stretch;

    // Jacobian of the unit-gain sea (Tessendorf 2001); OceanFoamAt applies the gain.
    const float3 u = float3(1.0f, 1.0f, 0.0f) + (A - float3(1.0f, 1.0f, 0.0f)) / max(s.gain, 0.05f);
    const float jacobian = u.x * u.y - u.z * u.z;
    if (((tid.x | tid.y) & (OCEAN_FOAM_HIST_STRIDE - 1u)) == 0u) {
        const float bin = floor((jacobian - OCEAN_FOAM_HIST_MIN) / OCEAN_FOAM_HIST_STEP);
        if (bin < OCEAN_FOAM_BINS)
            stats.InterlockedAdd((level * OCEAN_FOAM_BINS + (uint)max(bin, 0.0f)) * 4u, 1u);
    }
    const OceanFoamState state = OceanLoadFoamState(stats);
    const float breaking = saturate((state.threshold[level] - jacobian) / max(state.softness[level], 1e-3f));
    if (breaking > 0.0f) {
        // Each point piles up only as much as its streak holds.
        const float2 world = (float2(P.foamCell[level].xy + int2(tid.xy)) + 0.5f) * texel;
        const float streak = OceanFoamStreaks(world, texel, P.foamWind);
        const float most = saturate(0.2f + 0.65f * streak);
        if (amount < most)
            amount = min(amount + P.foamRate * gDt * breaking * streak, most);
    }
    dst[tid] = amount;
    if (level == OCEAN_FOAM_REFERENCE_LEVEL && ((tid.x | tid.y) & (OCEAN_FOAM_HIST_STRIDE - 1u)) == 0u) {
        const uint white = WaveActiveSum((uint)(OceanFoamWhite(amount) * OCEAN_FOAM_WHITE_SCALE + 0.5f));
        if (WaveIsFirstLane())
            stats.InterlockedAdd(OCEAN_FOAM_WHITE_OFFSET, white);
    }
}

// gsFoamCount[i] = a level's samples in bins 0..i; the histogram is cleared.
groupshared uint gsFoamCount[OCEAN_FOAM_BINS];
static const float OCEAN_FOAM_SAMPLES =
    float((OCEAN_FOAM_SIZE / OCEAN_FOAM_HIST_STRIDE) * (OCEAN_FOAM_SIZE / OCEAN_FOAM_HIST_STRIDE));
void OceanFoamHistogram(RWByteAddressBuffer stats, uint level, uint i) {
    const uint addr = (level * OCEAN_FOAM_BINS + i) * 4u;
    GroupMemoryBarrierWithGroupSync(); // the last level's sums have been read
    gsFoamCount[i] = stats.Load(addr);
    stats.Store(addr, 0u);
    GroupMemoryBarrierWithGroupSync();
    [unroll] for (uint o = 1u; o < OCEAN_FOAM_BINS; o <<= 1u) {
        const uint add = i >= o ? gsFoamCount[i - o] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gsFoamCount[i] += add;
        GroupMemoryBarrierWithGroupSync();
    }
}
float OceanFoamQuantile(float share) {
    const float target = share * OCEAN_FOAM_SAMPLES;
    if (target <= 0.0f)
        return OCEAN_FOAM_HIST_MIN;
    uint lo = 0u, hi = OCEAN_FOAM_BINS;
    [loop] while (lo < hi) {
        const uint mid = (lo + hi) >> 1u;
        if (float(gsFoamCount[mid]) >= target) hi = mid;
        else lo = mid + 1u;
    }
    if (lo >= OCEAN_FOAM_BINS)
        return OCEAN_FOAM_HIST_MIN + OCEAN_FOAM_BINS * OCEAN_FOAM_HIST_STEP;
    const float below = lo > 0u ? float(gsFoamCount[lo - 1u]) : 0.0f;
    return OCEAN_FOAM_HIST_MIN + (float(lo) + (target - below) / max(float(gsFoamCount[lo]) - below, 1.0f)) * OCEAN_FOAM_HIST_STEP;
}
float OceanFoamShareBelow(float jacobian) {
    const float b = (jacobian - OCEAN_FOAM_HIST_MIN) / OCEAN_FOAM_HIST_STEP;
    if (b <= 0.0f)
        return 0.0f;
    const uint i = min((uint)b, OCEAN_FOAM_BINS - 1u);
    const float below = i > 0u ? float(gsFoamCount[i - 1u]) : 0.0f;
    return lerp(below, float(gsFoamCount[i]), saturate(b - float(i))) / OCEAN_FOAM_SAMPLES;
}
// Mean cover of level `coarse` over level `coarse - k`'s ground.
float OceanFoamCoverOver(uint base, uint coarse, uint k) {
    const uint mip = OCEAN_FOAM_MIPS - 2u - k;
    RWTexture2DArray<float> m = ResourceDescriptorHeap[base + mip];
    const uint c = ((uint)OCEAN_FOAM_SIZE >> mip) / 2u;
    return 0.25f * (m[uint3(c - 1u, c - 1u, coarse)] + m[uint3(c, c - 1u, coarse)] +
                    m[uint3(c - 1u, c, coarse)] + m[uint3(c, c, coarse)]);
}

// Next frame's breaking points: histogram quantiles of a share steered toward foamCover.
[numthreads(OCEAN_FOAM_BINS, 1, 1)]
void OceanFoamStats(uint3 gtid : SV_GroupThreadID) {
    const OceanParamsGPU P = OceanParams();
    RWByteAddressBuffer stats = ResourceDescriptorHeap[OCEAN_UAV_FOAM_STATS];
    const uint base = (P.foamParity & 1u) == 0u ? OCEAN_UAV_FOAM0_MIPS : OCEAN_UAV_FOAM1_MIPS;
    RWTexture2DArray<float> top = ResourceDescriptorHeap[base + OCEAN_FOAM_MIPS - 1u];
    const uint i = gtid.x;
    const uint R = OCEAN_FOAM_REFERENCE_LEVEL;
    OceanFoamState state = OceanLoadFoamState(stats);
    const bool valid = state.valid != 0u;

    float coverage[OCEAN_FOAM_LEVELS];
    [unroll] for (uint l = 0u; l < OCEAN_FOAM_LEVELS; ++l)
        coverage[l] = top[uint3(0u, 0u, l)];
    const float white = (float)stats.Load(OCEAN_FOAM_WHITE_OFFSET) / (OCEAN_FOAM_WHITE_SCALE * OCEAN_FOAM_SAMPLES);

    // After a restart, hold the steering for two fresh-foam lifetimes.
    const float whiteRate = max(P.foamDecayRate * OCEAN_FOAM_FRESH_DECAY, 1e-3f);
    float settle = valid ? max(state.settle - gDt, 0.0f) : min(2.0f / whiteRate, 10.0f);
    if (P.foamHistory == 0u)
        settle = min(2.0f / whiteRate, 10.0f);
    const bool steer = settle <= 0.0f && gDt > 0.0f;
    // White cover ~ share * lifetime: seeds the share, rescales it when either changes.
    const float target = P.foamCover;
    float share = valid ? state.share : min(target * whiteRate, 0.5f);
    if (valid && state.target > 0.0f && target > 0.0f)
        share *= target / state.target;
    if (valid && state.decayRate > 0.0f)
        share *= P.foamDecayRate / state.decayRate;
    if (steer) {
        const float eps = 0.05f * target + 1e-6f;
        share *= exp(P.foamSteer * gDt * clamp(log((target + eps) / (white + eps)), -1.0f, 1.0f));
    }
    share = target > 0.0f ? clamp(share, 1e-6f, 0.5f) : 0.0f;

    // Steer each level to the reference level's cover on shared ground.
    float match[OCEAN_FOAM_LEVELS];
    [unroll] for (uint l = 0u; l < OCEAN_FOAM_LEVELS; ++l) {
        match[l] = valid && l != R ? state.match[l] : 1.0f;
        if (l == R || !steer)
            continue;
        const float mine = l < R ? coverage[l] : OceanFoamCoverOver(base, l, l - R);
        const float reference = l < R ? OceanFoamCoverOver(base, R, R - l) : coverage[R];
        const float eps = 0.05f * reference + 1e-6f;
        match[l] = clamp(match[l] * exp(P.foamSteer * gDt * clamp(log((reference + eps) / (mine + eps)), -1.0f, 1.0f)),
                         0.125f, 8.0f);
    }

    float threshold[OCEAN_FOAM_LEVELS], softness[OCEAN_FOAM_LEVELS];
    OceanFoamHistogram(stats, R, i);
    threshold[R] = min(OceanFoamQuantile(share), P.foamBreakMax);
    softness[R] = threshold[R] - min(OceanFoamQuantile(0.25f * share), threshold[R] - 0.02f);
    const float breaking = OceanFoamShareBelow(threshold[R]);
    // Anti-windup when the sea cannot break the share.
    share = min(share, 1.5f * breaking + 1e-6f);
    [unroll] for (uint l = 0u; l < OCEAN_FOAM_LEVELS; ++l) {
        if (l == R)
            continue;
        OceanFoamHistogram(stats, l, i);
        const float s = min(breaking * match[l], 0.5f);
        float t = OceanFoamQuantile(s);
        // Finer levels test the same surface, so foamBreakMax applies too.
        if (l < R)
            t = min(t, P.foamBreakMax);
        threshold[l] = t;
        softness[l] = t - min(OceanFoamQuantile(0.25f * s), t - 0.02f);
    }

    if (i == 0u) {
        // Settle over ~0.1 s; the first set is taken as is.
        const float follow = valid ? 1.0f - exp(-gDt / 0.1f) : 1.0f;
        [unroll] for (uint l = 0u; l < OCEAN_FOAM_LEVELS; ++l) {
            state.threshold[l] = lerp(state.threshold[l], threshold[l], follow);
            state.softness[l] = lerp(state.softness[l], softness[l], follow);
            state.match[l] = match[l];
            state.coverage[l] = coverage[l];
        }
        state.white = white;
        state.share = share;
        state.breaking = breaking;
        state.target = target;
        state.decayRate = P.foamDecayRate;
        state.settle = settle;
        state.valid = 1u;
        stats.Store<OceanFoamState>(OCEAN_FOAM_STATE_OFFSET, state);
        stats.Store(OCEAN_FOAM_WHITE_OFFSET, 0u);
    }
}

// One mip of every foam level. gU1: destination mip.
[numthreads(8, 8, 1)]
void OceanFoamMip(uint3 tid : SV_DispatchThreadID) {
    const uint dstSize = max(1u, (uint)OCEAN_FOAM_SIZE >> gU1);
    if (tid.x >= dstSize || tid.y >= dstSize)
        return;
    const uint base = (OceanParams().foamParity & 1u) == 0u ? OCEAN_UAV_FOAM0_MIPS : OCEAN_UAV_FOAM1_MIPS;
    RWTexture2DArray<float> src = ResourceDescriptorHeap[base + gU1 - 1u];
    RWTexture2DArray<float> dst = ResourceDescriptorHeap[base + gU1];
    const uint2 s = tid.xy * 2u;
    const uint l = tid.z;
    dst[uint3(tid.xy, l)] = 0.25f * (src[uint3(s, l)] + src[uint3(s + uint2(1, 0), l)] +
                                     src[uint3(s + uint2(0, 1), l)] + src[uint3(s + 1u, l)]);
}
