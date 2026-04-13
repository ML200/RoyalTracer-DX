#define COMPUTE_PASS
#include "Includes_v8.hlsli"

// Step 1: Normal-difference edge detection + temporal accumulation.
// Output raw accumulated value for visual verification.
//
// Scratch slices ping-pong by frame parity (6 <-> 8).

#define DECAY          0.86f
#define NORMAL_THRESH  0.36f

// Binary edge: 1 if any 3x3 neighbor has dot(n_center, n_neighbor) < NORMAL_THRESH.
inline float ComputeEdge(RWByteAddressBuffer buf, float2 dims, int2 pixel)
{
    uint px = MapPixelID(dims, (uint2)pixel);
    if (load_isEmitter(buf, px)) return 1.0f;

    float3 nc = load_n1_s(buf, px);

    static const int2 off[8] = {
        int2(-1,-1), int2(0,-1), int2(1,-1),
        int2(-1, 0),             int2(1, 0),
        int2(-1, 1), int2(0, 1), int2(1, 1)
    };

    [unroll]
    for (uint i = 0; i < 8; i++)
    {
        int2 np = pixel + off[i];
        if (np.x < 0 || np.y < 0 || np.x >= (int)dims.x || np.y >= (int)dims.y)
            continue;
        uint npx = MapPixelID(dims, (uint2)np);
        if (load_isEmitter(buf, npx))
            return 1.0f;
        float3 nn = load_n1_s(buf, npx);
        if (dot(nc, nn) < NORMAL_THRESH)
            return 1.0f;
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
