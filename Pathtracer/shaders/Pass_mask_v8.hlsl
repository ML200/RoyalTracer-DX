#define COMPUTE_PASS
#include "Includes_v8.hlsli"

// Step 1: Normal-difference edge detection + temporal accumulation.
// Output raw accumulated value for visual verification.
//
// Scratch slices ping-pong by frame parity (6 <-> 8).

#define DECAY          0.86f
#define NORMAL_THRESH  0.36f
#define DEPTH_THRESH   0.4f

// Binary edge: 1 if any pair in the 2x2 patch (self, right, down, diagonal)
// fails the normal test (dot < 0.36) or the plane-distance test (> 0.4).
// Matches RejectNormal_GI / RejectDistance_GI thresholds from temporal ReSTIR.
inline float ComputeEdge(RWByteAddressBuffer buf, float2 dims, int2 pixel)
{
    static const int2 off[4] = { int2(0,0), int2(1,0), int2(0,1), int2(1,1) };
    float3 n[4];
    float3 pos[4];
    bool   dead[4];

    [unroll]
    for (uint i = 0; i < 4; i++)
    {
        int2 p = pixel + off[i];
        if (p.x < 0 || p.y < 0 || p.x >= (int)dims.x || p.y >= (int)dims.y)
            { n[i] = float3(0,0,0); pos[i] = float3(0,0,0); dead[i] = true; continue; }
        uint px = MapPixelID(dims, (uint2)p);
        uint iID = load_instID(buf, px);
        dead[i] = load_isEmitter(buf, px) || (iID == 0xFFFFFFFFu);
        if (dead[i])
            { n[i] = float3(0,0,0); pos[i] = float3(0,0,0); continue; }
        n[i] = load_n1_s(buf, px);
        uint pID   = load_primID(buf, px);
        float2 bar = load_bary(buf, px);
        pos[i] = ReconstructPosition(iID, pID, bar);
    }

    // Mixed dead/live patch is always an edge.
    bool hasDead = dead[0] || dead[1] || dead[2] || dead[3];
    bool hasLive = !dead[0] || !dead[1] || !dead[2] || !dead[3];
    if (hasDead && hasLive) return 1.0f;
    if (hasDead) return 0.0f;

    // Check all 6 pairs: normal similarity + plane distance.
    [unroll] for (uint a = 0; a < 3; a++)
    [unroll] for (uint b = a + 1; b < 4; b++)
    {
        if (dot(n[a], n[b]) < NORMAL_THRESH) return 1.0f;
        if (abs(dot(pos[b] - pos[a], n[a])) > DEPTH_THRESH) return 1.0f;
    }

    return 0.0f;
}

inline int2 Reproject(float2 dims, uint instID, uint primID, float2 bary)
{
    float3 wpos   = ReconstructPosition(instID, primID, bary);
    float2 prevPx = GetLastFramePixelCoordinates_Float(
                        wpos, prevView, prevProjection, dims, instID);
    if (prevPx.x < 0.0f) return int2(-1, -1);
    int2 pc = int2(floor(prevPx + 0.5f));
    if (pc.x < 1 || pc.y < 1 || pc.x >= (int)dims.x - 1 || pc.y >= (int)dims.y - 1)
        return int2(-1, -1);
    return pc;
}

inline float SafeRead(RWTexture2DArray<float4> tex, uint3 coord)
{
    float v = tex[coord].r;
    return (isnan(v) || isinf(v) || v < 0.0f) ? 0.0f : v;
}

[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const float2 dims  = float2(IMG_W, IMG_H);
    const int2   pixel = int2(tid.xy);

    const uint frameBit = (uint)time & 1u;
    const uint sliceR   = frameBit ? 8u : 6u;
    const uint sliceW   = frameBit ? 6u : 8u;

    uint gpx    = MapPixelID(dims, (uint2)pixel);
    uint instID = load_instID(g_sample_current, gpx);
    bool isDead = load_isEmitter(g_sample_current, gpx) || (instID == 0xFFFFFFFFu);

    // Dead pixels (sky, emitters) are not edges — clear and bail.
    if (isDead)
    {
        gOutput[uint3(tid.xy, 3)]           = float4(0, 0, 0, 1.0f);
        gScratchPing[uint3(tid.xy, sliceW)] = float4(0, 0, 0, 0);
        return;
    }

    float edge = ComputeEdge(g_sample_current, dims, pixel);

    float prevAccum = 0.0f;
    uint   primID = load_primID(g_sample_current, gpx);
    float2 bary   = load_bary(g_sample_current, gpx);
    int2   reproj = Reproject(dims, instID, primID, bary);

    if (reproj.x >= 0)
        prevAccum = SafeRead(gScratchPing, uint3(reproj, sliceR));

    float accumulated = max(edge, prevAccum * DECAY);

    float m = (accumulated > 0.1f) ? 1.0f : 0.0f;
    gOutput[uint3(tid.xy, 3)] = float4(m, m, m, 1.0f);
    gScratchPing[uint3(tid.xy, sliceW)] = float4(accumulated, 0, 0, 0);
}
