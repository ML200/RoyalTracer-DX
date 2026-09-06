#ifndef SHARC_DEBUG_V8_HLSLI
#define SHARC_DEBUG_V8_HLSLI
#include "Sharc_v8.hlsli"

struct SharcDebugSample
{
    uint address;
    uint colorKey;
    float3 fraction; // position relative to the nearest node, in cell widths
    bool room;       // a probed bucket still has an empty slot
};

// Inspect one nearest node, not the filtered estimator. Search all signed
// normal bins and the complete collision bucket, including holes. Unlike a
// rendering query, this also exposes cold / zero-only / uncertain histories.
SharcDebugSample SharcDebugLookup(SharcSurface s, uint level)
{
    int3 base; float3 f;
    SharcGrid(s.position, level, base, f);
    int3 corner = int3(f >= 0.5f);
    int3 node = base + corner;
    SharcDebugSample result;
    result.address = SHARC_INVALID;
    result.colorKey = 0u;
    result.fraction = f - float3(corner);
    result.room = false;
    float3 relative = result.fraction * SharcCellSize(level);
    float3 normalWeights = SharcNormalWeights(s.geometricNormal);
    float best = 0.0f;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (normalWeights[axis] < 1e-4f) continue;
        uint meta = SharcMeta(s, level, axis);
        uint hash = SharcHash(node, meta, s.instance, s.material);
        uint bucket = SharcBucketOf(hash);
        uint states[SHARC_BUCKET_SIZE];
        SharcLoadBucket(bucket, states);
        const bool merged = SharcMergedNode(states, hash);
        [unroll] for (uint p = 0u; p < SHARC_BUCKET_SIZE; ++p)
        {
            if (states[p] == 0u) result.room = true;
            if (states[p] != hash) continue;
            uint e = SharcEntryAddress(bucket * SHARC_BUCKET_SIZE + p);
            SharcDescriptor d = SharcLoadDescriptor(e);
            if (!SharcKeyMatches(d, node, meta, s)) continue;
            // Geometric normals, both planes, material, reflectance and
            // roughness checks keep separate sheets separate; merged nodes of
            // sub-cell detail match on orientation only, as the tracer does.
            float score = normalWeights[axis] * (merged ? SharcMergedWeight(d, s)
                : SharcSurfaceWeight(d, s, relative, SharcCellSize(level)));
            if (score > best)
            {
                best = score;
                result.address = e;
                result.colorKey = hash;
            }
        }
    }
    return result;
}

// RGB with w=0 is a display-space diagnostic color. w=1 is raw scene radiance
// and receives the normal exposure/tonemap only at presentation.
float4 SharcDebugColor(SharcSurface s, uint level, uint mode)
{
    SharcDebugSample cell = SharcDebugLookup(s, level);
    if (cell.address == SHARC_INVALID)
    {
        // Dark red: every slot of the node's bucket holds a live record, so no
        // observation could be inserted yet. Magenta / grey: nothing observed.
        if (!cell.room)
            return float4(mode == SHARC_DEBUG_CELLS ? float3(0.30f, 0.04f, 0.04f)
                : float3(0.55f, 0.04f, 0.04f), 0.0f);
        return float4(mode == SHARC_DEBUG_CELLS ? float3(0.10f, 0.10f, 0.10f)
            : float3(0.45f, 0.025f, 0.35f), 0.0f);
    }

    uint e = cell.address;
    if (mode == SHARC_DEBUG_LIGHTING)
    {
        if (asfloat(g_sharc.Load(e + SHARC_HISTORY_W)) <= 0.0f)
            return float4(0.65f, 0.32f, 0.025f, 0.0f);
        // Remodulate with the RECORD'S reflectance to show constant cell values,
        // rather than reintroducing the visible surface's texture into the view.
        float3 radiance = asfloat(g_sharc.Load3(e + SHARC_MEAN)) * asfloat(g_sharc.Load3(e + SHARC_DEMODULATOR));
        if (any(!isfinite(radiance))) return float4(1, 0, 0, 0);
        return float4(max(radiance, 0.0f), 1.0f);
    }

    uint h = Hash32(cell.colorKey);
    float3 color = 0.25f + 0.75f * float3(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u) / 255.0f;
    color *= lerp(0.4f, 1.0f, SharcConfidence(SharcLoadHistory(e)));
    // Nearest-node cells have boundaries at +/- half a grid spacing. Ignore
    // axes normal to the surface so a coplanar wall does not become all border.
    float3 edge = 0.5f - abs(cell.fraction);
    float3 tangent = 1.0f - s.geometricNormal * s.geometricNormal;
    float distanceToEdge = 1.0f;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
        if (tangent[axis] > 1e-3f) distanceToEdge = min(distanceToEdge, edge[axis]);
    color *= lerp(0.15f, 1.0f, smoothstep(0.015f, 0.04f, distanceToEdge));
    return float4(color, 0.0f);
}
#endif
