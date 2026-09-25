// Tessellates the selected ocean quadtree leaves straight into the renderer's global vertex
// buffer, so the acceleration structures build from geometry that never touches the CPU.
//
// Each vertex is displaced by the cascades filtered to this tile's vertex spacing: a coarse tile
// carries only the waves it can represent, and everything finer reaches the image through the
// full-resolution normal the hit evaluator samples. Vertices on an edge whose neighbour is one
// level coarser are collapsed onto that neighbour's sample positions, which closes the T-junction
// cracks that would otherwise let rays leak through the surface.

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

// Matches UnpackNormal_INT in Compression_v8.hlsli: octahedral, two signed 16-bit components.
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
    const float2 local = float2(i, j) * step; // lerp(localLo, localHi, w): stitching moves no vertex sideways

    const float2 qLo = tile.anchor.xz + localLo;
    OceanSample s = OceanSampleSurface(qLo, OceanGeometryWidth(qLo, P));
    if (src.w > 0.0f) {
        // Interpolate the two samples of the coarser edge rather than displacing this point, so
        // the vertex lands exactly on the segment the neighbour's edge spans.
        const float2 qHi = tile.anchor.xz + localHi;
        const OceanSample b = OceanSampleSurface(qHi, OceanGeometryWidth(qHi, P));
        s.displacement = lerp(s.displacement, b.displacement, src.w);
        s.gradient = lerp(s.gradient, b.gradient, src.w);
        s.stretch = lerp(s.stretch, b.stretch, src.w);
    }

    // The Earth's drop is interpolated like the displacement: evaluating the quadratic at the
    // interpolated point would bow the vertex off the neighbour's straight edge.
    const float curveLo = tile.curveBase + dot(tile.curveGrad, localLo) + dot(localLo, localLo) * 0.5f * tile.invCurveRadius;
    const float curveHi = tile.curveBase + dot(tile.curveGrad, localHi) + dot(localHi, localHi) * 0.5f * tile.invCurveRadius;
    const float curve = lerp(curveLo, curveHi, src.w);
    const float2 curveSlope = tile.curveGrad + local * tile.invCurveRadius;

    OceanVertexOut v;
    v.vertex = float3(local.x + s.displacement.x, s.displacement.y - curve, local.y + s.displacement.z);
    // The hit evaluator replaces this with the full-resolution normal; it only has to agree with
    // the triangle well enough for the interpolation guard in EvalSurfaceStateImpl to accept it.
    v.packedNormal = OceanPackNormal(OceanNormalFromSample(s, curveSlope));
    // Texture coordinates are unit-per-tile: the hit evaluator recovers the undisplaced sea
    // position from them, and derives its tangent frame and beam footprint from them.
    v.texCoord = (half2)(float2(i, j) / (float)G);

    verts[tile.vertexBase + tid.x] = v;
}

// ---------------------------------------------------------------------------------------------
// Whitecaps, after Crest's foam simulation (wave-harmonic/crest, MIT). One thread per texel of each
// camera-centred level (z). Last frame's foam at the same world point fades; where the composite
// surface - every cascade, filtered to the breaking scale - is squeezed past breaking, more
// gathers. The levels move in whole texels, so carrying the foam over is an exact texel offset;
// what scrolls in from outside a level is taken from the next coarser one.
// ---------------------------------------------------------------------------------------------
OceanFoamState OceanLoadFoamState(RWByteAddressBuffer stats) {
    return stats.Load<OceanFoamState>(OCEAN_FOAM_STATE_OFFSET);
}

// How plainly white foam of this cover is: what a whitecap census photographs as foam, and what
// the cover asked for (foamCover) is measured in - the cap and the raft it leaves while that is
// still a sheet with holes in it (OceanFoamRaft). The lace it dissolves into is left out: counted,
// old trails met the measured cover and the sea stopped breaking.
float OceanFoamWhite(float cover) {
    return saturate((cover - 0.3f) / 0.4f);
}

// How much foam a breaking crest throws up at this point of its run, one on average. A crest does not
// spill evenly along its length, and the foam it leaves comes off it in streaks along its path, so a
// trail is streaked rather than one smooth band: noise drawn out five times as long downwind as it is
// across, from a metre across down to a third of that. At the texel's absolute world position, so
// every level lays down the same streaks, each keeping the octaves its grid holds (four texels across
// at least).
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
            // Ground the level scrolled onto was held last frame by the next coarser one.
            const float4 coarse = P.foamLevel[level + 1u];
            const float coarseTexel = 2.0f * texel;
            const int2 c = int2(floor((q - (coarse.xy - coarse.zw * coarseTexel)) / coarseTexel));
            if (all(c >= 0) && all(c < OCEAN_FOAM_SIZE))
                amount = prev.Load(int4(c, level + 1u, 0));
        }
    }
    // A cap and the sheet it leaves collapse within a couple of seconds, as the bubbles under them
    // rise and burst; the lace of strands that is left lingers - foamDecayRate is the lace's rate.
    // How the cover comes apart is drawn where it is shown (OceanFoamAt), from the cover alone.
    // Foam fades for as long as simulated time moves; a paused sea keeps its caps.
    const float fresh = smoothstep(0.05f, 0.4f, amount);
    amount *= exp(-P.foamDecayRate * gDt * lerp(1.0f, OCEAN_FOAM_FRESH_DECAY, fresh));

    // Each level tests the surface at the breaking scale, or at its own texel where that is
    // coarser: a far level records the breaking of the waves long enough to show there, as caps on
    // their crests, rather than aliasing the small breakers it cannot hold. The coarser levels'
    // surfaces are the smoother for it, which is why each level has its own breaking point.
    const OceanSample s = OceanSampleSurface(q, max(OCEAN_FOAM_BREAK_WIDTH, texel));
    const float3 A = s.stretch;

    // Crests are tested on the unit-gain sea, the kilometre-scale sea-state field divided back
    // out, so one breaking point holds for the whole ocean wherever the camera is; how much more
    // a rough patch breaks is laid on where the foam is shown (OceanFoamAt).
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
        // Where the crest throws up little, its foam also never gets past a sheet with holes in it:
        // however long a crest keeps breaking, each point only piles up as much as its streak holds.
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

// One level's histogram, cleared behind it and summed into gsFoamCount[i] = samples in bins 0..i.
// One thread per bin.
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
// The Jacobian below which `share` of the histogrammed sea lies. Nothing asked for breaks nothing;
// more than the histogram holds breaks everything it holds.
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
// Share of the histogrammed sea below this Jacobian.
float OceanFoamShareBelow(float jacobian) {
    const float b = (jacobian - OCEAN_FOAM_HIST_MIN) / OCEAN_FOAM_HIST_STEP;
    if (b <= 0.0f)
        return 0.0f;
    const uint i = min((uint)b, OCEAN_FOAM_BINS - 1u);
    const float below = i > 0u ? float(gsFoamCount[i - 1u]) : 0.0f;
    return lerp(below, float(gsFoamCount[i]), saturate(b - float(i))) / OCEAN_FOAM_SAMPLES;
}
// Mean cover of level `coarse` over the ground of the level `k` steps finer, the middle 2^-k of
// it: the middle two texels each way of mip 10 - k (the levels are aligned to within a texel of
// the coarser one).
float OceanFoamCoverOver(uint base, uint coarse, uint k) {
    const uint mip = OCEAN_FOAM_MIPS - 2u - k;
    RWTexture2DArray<float> m = ResourceDescriptorHeap[base + mip];
    const uint c = ((uint)OCEAN_FOAM_SIZE >> mip) / 2u;
    return 0.25f * (m[uint3(c - 1u, c - 1u, coarse)] + m[uint3(c, c - 1u, coarse)] +
                    m[uint3(c - 1u, c, coarse)] + m[uint3(c, c, coarse)]);
}

// After the foam update and its mips: set each level's breaking point for the next frame from the
// histograms the update just filled.
//
// How much breaks is steered, not assumed. The share of the reference level that breaks is the
// crests squeezed hardest, and it is walked until the foam they leave covers foamCover of the sea
// - the share that takes is the waves' and the foam lifetime's business, and far from
// proportional: a light sea's few breakers barely graze the breaking point and leave faint foam, a
// gale's pitch far past it. No crest squeezed less than foamBreakMax breaks at all, though, so a
// sea that cannot break that much stays under its cover. Every other level breaks the reference
// level's share times its OceanFoamState.match, walked until the two lay the same cover on the
// ground they share. A crest breaks in full at the Jacobian a quarter of the share lies below, so
// the core of every breaker is solid foam whatever the tail of the sea's Jacobian looks like.
//
// The steering has to be slower than the foam it steers, whose response lags by its lifetime, so
// its rate comes with it (foamSteer). A change of the cover asked for is taken at once as the same
// change of the share - with every breaker's core solid, the two go roughly in proportion - and
// the steering only trims after. One group of a thread per bin; every thread works the same
// scalars.
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
    // The cover the steering answers to: how much of the reference level is plainly white.
    const float white = (float)stats.Load(OCEAN_FOAM_WHITE_OFFSET) / (OCEAN_FOAM_WHITE_SCALE * OCEAN_FOAM_SAMPLES);

    // Foam that started over (the first frame, a jump) has to build up for a couple of lifetimes
    // before its cover says anything about the share. The white the steering reads is fresh foam,
    // which fades OCEAN_FOAM_FRESH_DECAY times faster than the lace.
    const float whiteRate = max(P.foamDecayRate * OCEAN_FOAM_FRESH_DECAY, 1e-3f);
    float settle = valid ? max(state.settle - gDt, 0.0f) : min(2.0f / whiteRate, 10.0f);
    if (P.foamHistory == 0u)
        settle = min(2.0f / whiteRate, 10.0f);
    const bool steer = settle <= 0.0f && gDt > 0.0f;
    // White cover is about the share times the white's lifetime, so that is where the share
    // starts, and where a change of either moves it at once.
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

    // Each other level against the reference one, over the ground the finer of the two holds.
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
    // A sea that cannot break the share asked for does not have the steering wind up asking for
    // ever more.
    share = min(share, 1.5f * breaking + 1e-6f);
    [unroll] for (uint l = 0u; l < OCEAN_FOAM_LEVELS; ++l) {
        if (l == R)
            continue;
        OceanFoamHistogram(stats, l, i);
        const float s = min(breaking * match[l], 0.5f);
        float t = OceanFoamQuantile(s);
        // The finer levels test the same filtered surface as the reference one, so no less
        // squeezed a crest breaks on them either.
        if (l < R)
            t = min(t, P.foamBreakMax);
        threshold[l] = t;
        softness[l] = t - min(OceanFoamQuantile(0.25f * s), t - 0.02f);
    }

    if (i == 0u) {
        // Breaking points settle within a tenth of a second rather than jumping with every frame's
        // histogram; the first set is taken as it is.
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
