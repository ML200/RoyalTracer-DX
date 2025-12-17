#include "Includes_v8.hlsli"

// ============================================================================
// HELPERS
// ============================================================================

inline float2 OctEncode(float3 n)
{
    // Normalize and project onto octahedron
    // Add epsilon to avoid div-by-zero if direction is degenerate
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-6f);
    float2 enc = n.xy;

    // Wrap the lower hemisphere
    if (n.z < 0.0f)
    {
        enc = (1.0f - abs(enc.yx))
            * float2(enc.x >= 0.0f ? 1.0f : -1.0f,
                     enc.y >= 0.0f ? 1.0f : -1.0f);
    }

    // Map [-1,1] -> [0,1]
    return enc * 0.5f + 0.5f;
}

inline uint EncodeDirection8(float3 dir)
{
    // Robust normalization
    float len = length(dir);
    float3 nd = (len > 1e-6f) ? (dir / len) : float3(0,0,1);

    float2 e  = OctEncode(nd);

    // 4 bits per axis (0..15)
    uint2 q = (uint2)(saturate(e) * 15.0f + 0.5f);

    // Pack: 4 bits X, 4 bits Y => 8 bits
    return (q.x & 0xFu) | ((q.y & 0xFu) << 4);
}

// Expands a 5-bit integer to 15 bits (..x..x..x)
// Input: 0..31
inline uint ExpandBits5(uint v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 15-bit Morton code (fits in 16 bits) from a normalized position
// Grid: 32x32x32
inline uint Morton3D_16Bit(float3 t)
{
    t = saturate(t);
    // Quantize to 5 bits (0..31)
    uint3 u = (uint3)(t * 31.0f + 0.5f);
    // Interleave: ZYX
    return ExpandBits5(u.x) | (ExpandBits5(u.y) << 1) | (ExpandBits5(u.z) << 2);
}

// ============================================================================
// COMPUTE SHADERS
// ============================================================================

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint elementIdx = tid.x;

    // 1. Get Active Count
    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);
    if (elementIdx >= activeCount) return;

    // 2. Load Input Stack Entry
    uint2 entry = LoadStack(g_InputStackIdx, elementIdx);

    // 3. Load Ray Geometry
    uint pixelIdx = entry.x;
    RayGeometry rgeo = LoadRayGeometry(g_pathStateBuffer, pixelIdx); // Direct load
    float3 origin = rgeo.origin;
    float dir = rgeo.dir;

    // 4. Load Previous Frame's Bounds (Offsets 0..20)
    // We normalize the current ray against where the rays WERE last frame.
    float3 minOrg, maxOrg;
    minOrg.x = OrderedUintToFloat(g_SortBounds.Load(0));
    minOrg.y = OrderedUintToFloat(g_SortBounds.Load(4));
    minOrg.z = OrderedUintToFloat(g_SortBounds.Load(8));

    maxOrg.x = OrderedUintToFloat(g_SortBounds.Load(12));
    maxOrg.y = OrderedUintToFloat(g_SortBounds.Load(16));
    maxOrg.z = OrderedUintToFloat(g_SortBounds.Load(20));

    // 5. Normalize & Generate 16-bit Morton Key
    // Ensure extent is non-zero
    float3 extent = max(maxOrg - minOrg, float3(1e-4f, 1e-4f, 1e-4f));
    float3 normalizedPos = (origin - minOrg) / extent;

    // Generate 15-bit key (0..32767)
    uint mortonKey = Morton3D_16Bit(normalizedPos);

    // Store Key in the entry (Using .y component)
    // Since it's 16 bits, it fits easily in the 32-bit uint.
    entry.y = mortonKey;

    StoreStack(g_InputStackIdx, elementIdx, entry);

    // 6. Update Bounds for NEXT Frame (Wave Optimized)
    // Aggregate min/max within the wave to reduce atomic traffic by 64x
    float3 waveMin = origin;
    float3 waveMax = origin;

    waveMin.x = WaveActiveMin(waveMin.x);
    waveMin.y = WaveActiveMin(waveMin.y);
    waveMin.z = WaveActiveMin(waveMin.z);

    waveMax.x = WaveActiveMax(waveMax.x);
    waveMax.y = WaveActiveMax(waveMax.y);
    waveMax.z = WaveActiveMax(waveMax.z);

    if (WaveIsFirstLane())
    {
        // Write to "Next Frame" slots (Offsets 32..52)
        // Ensure you swap these offsets (0 vs 32) in your C++ constant buffer every frame.
        g_SortBounds.InterlockedMin(32, FloatToOrderedUint(waveMin.x));
        g_SortBounds.InterlockedMin(36, FloatToOrderedUint(waveMin.y));
        g_SortBounds.InterlockedMin(40, FloatToOrderedUint(waveMin.z));

        g_SortBounds.InterlockedMax(44, FloatToOrderedUint(waveMax.x));
        g_SortBounds.InterlockedMax(48, FloatToOrderedUint(waveMax.y));
        g_SortBounds.InterlockedMax(52, FloatToOrderedUint(waveMax.z));
    }
}