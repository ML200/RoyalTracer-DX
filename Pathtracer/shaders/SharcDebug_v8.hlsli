#pragma once
#include "Sharc_v8.hlsli"

struct SharcDebugSample
{
    uint address;
    uint colorKey;
    float3 fraction;
    bool room;
};

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
            uint4 key; uint2 idm;
            SharcLoadKey(e, key, idm);
            if (!SharcKeyMatches(key, idm, node, meta, s)) continue;
            uint check; SharcDescriptor d; SharcHistory h;
            SharcLoadQueryRecord(e, SharcCellSize(level), check, d, h);

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

float4 SharcDebugColor(SharcSurface s, uint level, uint mode)
{
    SharcDebugSample cell = SharcDebugLookup(s, level);
    if (cell.address == SHARC_INVALID)
    {

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

        float3 radiance = asfloat(g_sharc.Load3(e + SHARC_MEAN)) * SharcLoadDemodulator(e);
        if (any(!isfinite(radiance))) return float4(1, 0, 0, 0);
        return float4(max(radiance, 0.0f), 1.0f);
    }

    uint h = Hash32(cell.colorKey);
    float3 color = 0.25f + 0.75f * float3(h & 255u, (h >> 8u) & 255u, (h >> 16u) & 255u) / 255.0f;
    color *= lerp(0.4f, 1.0f, SharcConfidence(SharcLoadHistory(e)));

    float3 edge = 0.5f - abs(cell.fraction);
    float3 tangent = 1.0f - s.geometricNormal * s.geometricNormal;
    float distanceToEdge = 1.0f;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
        if (tangent[axis] > 1e-3f) distanceToEdge = min(distanceToEdge, edge[axis]);
    color *= lerp(0.15f, 1.0f, smoothstep(0.015f, 0.04f, distanceToEdge));
    return float4(color, 0.0f);
}
