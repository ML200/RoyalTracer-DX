// Tessellates the selected ocean quadtree leaves straight into the renderer's global vertex
// buffer, so the acceleration structures build from geometry that never touches the CPU.
//
// Each vertex is displaced by the cascades filtered to this tile's vertex spacing: a coarse tile
// carries only the waves it can represent, and everything finer reaches the image through the
// slope map and the BRDF instead. Vertices on an edge whose neighbour is one level coarser are
// collapsed onto that neighbour's sample positions, which closes the T-junction cracks that would
// otherwise let rays leak through the surface.

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

// Surface slope at the vertex scale. The shading pass re-evaluates this per pixel at the ray
// footprint; the vertex normal only has to be consistent enough with the triangle it belongs to
// that the interpolation guard in EvalSurfaceStateImpl accepts it.
float3 OceanVertexNormal(float2 posXZ, float widthM, float2 curveSlope) {
    return OceanNormal(posXZ, widthM, curveSlope);
}

[numthreads(8, 8, 1)]
void OceanTiles(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OCEAN_TILE_EDGE_VERTS || tid.y >= OCEAN_TILE_EDGE_VERTS || tid.z >= gTileCount)
        return;

    StructuredBuffer<OceanTileGPU> tiles = ResourceDescriptorHeap[OCEAN_SRV_TILES];
    RWStructuredBuffer<OceanVertexOut> verts = ResourceDescriptorHeap[OCEAN_UAV_VERTS];
    const OceanParamsGPU P = OceanParams();

    const OceanTileGPU tile = tiles[tid.z];
    const uint G = OCEAN_TILE_GRID;
    uint i = tid.x;
    uint j = tid.y;

    const float step = tile.size / (float)G;

    // A stitched edge samples at the coarse neighbour's spacing, and its odd vertices land on the
    // midpoint of the even ones, exactly reproducing the neighbour's edge.
    const bool onNegX = (i == 0u) && (tile.stitch & OCEAN_EDGE_NEG_X);
    const bool onPosX = (i == G) && (tile.stitch & OCEAN_EDGE_POS_X);
    const bool onNegZ = (j == 0u) && (tile.stitch & OCEAN_EDGE_NEG_Z);
    const bool onPosZ = (j == G) && (tile.stitch & OCEAN_EDGE_POS_Z);
    const bool stitched = onNegX || onPosX || onNegZ || onPosZ;

    // Along a stitched edge the odd index is the collapsed one; which axis runs along the edge
    // depends on which edge it is.
    uint2 lo = uint2(i, j);
    uint2 hi = uint2(i, j);
    bool collapse = false;
    if (onNegX || onPosX) {
        collapse = (j & 1u) != 0u;
        lo.y = j - 1u;
        hi.y = min(j + 1u, G);
    } else if (onNegZ || onPosZ) {
        collapse = (i & 1u) != 0u;
        lo.x = i - 1u;
        hi.x = min(i + 1u, G);
    }

    const float widthM = stitched ? step * 2.0f : step;

    const float2 localLo = float2(lo) * step;
    const float2 localHi = float2(hi) * step;
    const float2 local = collapse ? 0.5f * (localLo + localHi) : float2(i, j) * step;
    const float2 world = tile.anchor.xz + local;

    float3 d;
    if (collapse) {
        // Average the two coarse samples rather than displacing the midpoint, so this vertex
        // lands exactly on the segment the neighbour's edge spans.
        d = 0.5f * (OceanDisplacement(tile.anchor.xz + localLo, widthM) +
                    OceanDisplacement(tile.anchor.xz + localHi, widthM));
    } else {
        d = OceanDisplacement(world, widthM);
    }

    const float curve =
        tile.curveBase + dot(tile.curveGrad, local) + dot(local, local) * 0.5f * tile.invCurveRadius;
    const float3 outPos = float3(local.x + d.x, d.y - curve, local.y + d.z);

    const float2 curveSlope = tile.curveGrad + local * tile.invCurveRadius;
    const float3 n = OceanVertexNormal(world, widthM, curveSlope);

    OceanVertexOut v;
    v.vertex = outPos;
    v.packedNormal = OceanPackNormal(n);
    // Texture coordinates are unit-per-tile: nothing samples them, but the hit evaluator derives
    // its tangent frame and beam footprint from them and needs a non-degenerate parameterisation.
    v.texCoord = (half2)(float2(i, j) / (float)G);

    verts[tile.vertexBase + j * OCEAN_TILE_EDGE_VERTS + i] = v;
}


