// =================================================================================
// RIS Reservoir Storage Management
// =================================================================================

struct Reservoir_GI
{
    // Geometry (World Space)
    float3 xn0;
    float3 xn1;

    // Normals (Compressed Octahedral)
    float3 nn0;
    float3 nn1;

    // RIS Weights
    float  W;
    float  wsum; // Stored for local reuse

    // Radiance of remaining path segment & target function for the current path stored in the reservoir (RGB9E5 Compressed)
    float3 L;
    float3 F;

    uint   rSeed;

    // Confidence weight
    uint   M;

    // View Direction (Octahedral 24-bit combined with M)
    float3 V;

    // Albedo (RGB9E5 Compressed)
    float3 Kd0;
    float3 Kd1;

    // Surface Parameters (Packed 4x 8-bit)
    // Pr = Roughness, Pm = Metallicness
    float  Pr0;
    float  Pr1;
    float  Pm0;
    float  Pm1;

    // Object & Material IDs
    uint   objID0;
    uint   objID1;
    uint   mID0;   // 16-bit
    uint   mID1;   // 16-bit
};

// =================================================================================
// Memory Layout
// =================================================================================
// Size: 80 Bytes
static const uint BYTES_GI    = 80u;

static const uint O_GI_PACK1  =  0u;   // float4: xn0, nn0(pk)
static const uint O_GI_PACK2  = 16u;   // float4: xn1, nn1(pk)
static const uint O_GI_PACK3  = 32u;   // float4: W, wsum, L(pk), F(pk)
static const uint O_GI_PACK4  = 48u;   // uint4:  obj0, obj1, mIDs(pk), seed
static const uint O_GI_PACK5  = 64u;   // uint4:  Kd0(pk), Kd1(pk), Params(pk), V_M(pk)

uint pixelBaseAddrGI(uint pixelIdx) { return pixelIdx * BYTES_GI; }

// =================================================================================
// Compression Helpers
// =================================================================================

// -- Material IDs: 2x 16-bit --
uint PackMatIDs(uint m0, uint m1) { return (m0 & 0xFFFFu) | (m1 << 16); }
void UnpackMatIDs(uint v, out uint m0, out uint m1) { m0 = v & 0xFFFFu; m1 = v >> 16; }

// -- Surface Params: 4x 8-bit (Roughness/Metalness) --
uint PackSurfaceParams(float r0, float r1, float m0, float m1)
{
    uint p0 = uint(saturate(r0) * 255.0f);
    uint p1 = uint(saturate(r1) * 255.0f);
    uint p2 = uint(saturate(m0) * 255.0f);
    uint p3 = uint(saturate(m1) * 255.0f);
    return p0 | (p1 << 8) | (p2 << 16) | (p3 << 24);
}

void UnpackSurfaceParams(uint v, out float r0, out float r1, out float m0, out float m1)
{
    const float s = 1.0f / 255.0f;
    r0 = float(v & 0xFFu) * s;
    r1 = float((v >> 8) & 0xFFu) * s;
    m0 = float((v >> 16) & 0xFFu) * s;
    m1 = float(v >> 24) * s;
}

// -- View Dir (24-bit Oct) + M (8-bit) --
// Encodes Normal to 2x12-bit Snorm, leaves 8 bits for M
uint PackV_M(float3 V, uint M)
{
    // Octahedral Encode
    float3 n = V / (abs(V.x) + abs(V.y) + abs(V.z));

    // Fix: Use step() instead of ternary for vector conditions
    // equivalent to: (n.xy >= 0.0f ? 1.0f : -1.0f)
    float2 signVal = step(0.0f, n.xy) * 2.0f - 1.0f;

    // The outer ternary is fine because the condition (n.z >= 0.0f) is scalar
    float2 oct = n.z >= 0.0f ? n.xy : (1.0f - abs(n.yx)) * signVal;

    // 12-bit quantization [0, 4095]
    uint x = uint(saturate(oct.x * 0.5f + 0.5f) * 4095.0f);
    uint y = uint(saturate(oct.y * 0.5f + 0.5f) * 4095.0f);

    return x | (y << 12) | ((M & 0xFFu) << 24);
}

void UnpackV_M(uint v, out float3 V, out uint M)
{
    M = v >> 24;

    float2 oct;
    oct.x = float(v & 0xFFFu) * (1.0f / 4095.0f) * 2.0f - 1.0f;
    oct.y = float((v >> 12) & 0xFFFu) * (1.0f / 4095.0f) * 2.0f - 1.0f;

    float3 n = float3(oct, 1.0f - abs(oct.x) - abs(oct.y));
    float t = max(-n.z, 0.0f);

    // These ternaries are safe because the conditions are scalar
    n.x += n.x >= 0.0f ? -t : t;
    n.y += n.y >= 0.0f ? -t : t;

    V = normalize(n);
}

// =================================================================================
// Main Accessors
// =================================================================================

void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    // Pack 1: xn0, nn0
    buf.Store4(base + O_GI_PACK1, uint4(
        asuint(r.xn0),
        PackNormal(r.nn0)
    ));

    // Pack 2: xn1, nn1
    buf.Store4(base + O_GI_PACK2, uint4(
        asuint(r.xn1),
        PackNormal(r.nn1)
    ));

    // Pack 3: W, wsum, L, F
    buf.Store4(base + O_GI_PACK3, uint4(
        asuint(r.W),
        asuint(r.wsum),
        PackRGB9E5(r.L),
        PackRGB9E5(r.F)
    ));

    // Pack 4: objID0, objID1, mIDs, rSeed
    buf.Store4(base + O_GI_PACK4, uint4(
        r.objID0,
        r.objID1,
        PackMatIDs(r.mID0, r.mID1),
        r.rSeed
    ));

    // Pack 5: Kd0, Kd1, Params, V+M
    buf.Store4(base + O_GI_PACK5, uint4(
        PackRGB9E5(r.Kd0),
        PackRGB9E5(r.Kd1),
        PackSurfaceParams(r.Pr0, r.Pr1, r.Pm0, r.Pm1),
        PackV_M(r.V, r.M)
    ));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddrGI(pixelIdx);
    Reservoir_GI r;

    // Load full cache lines
    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);
    uint4 p4 = buf.Load4(base + O_GI_PACK4);
    uint4 p5 = buf.Load4(base + O_GI_PACK5);

    // Unpack 1
    r.xn0 = asfloat(p1.xyz);
    r.nn0 = UnpackNormal(p1.w);

    // Unpack 2
    r.xn1 = asfloat(p2.xyz);
    r.nn1 = UnpackNormal(p2.w);

    // Unpack 3
    r.W    = asfloat(p3.x);
    r.wsum = asfloat(p3.y);
    r.L    = UnpackRGB9E5(p3.z);
    r.F    = UnpackRGB9E5(p3.w);

    // Unpack 4
    r.objID0 = p4.x;
    r.objID1 = p4.y;
    UnpackMatIDs(p4.z, r.mID0, r.mID1);
    r.rSeed  = p4.w;

    // Unpack 5
    r.Kd0 = UnpackRGB9E5(p5.x);
    r.Kd1 = UnpackRGB9E5(p5.y);
    UnpackSurfaceParams(p5.z, r.Pr0, r.Pr1, r.Pm0, r.Pm1);
    UnpackV_M(p5.w, r.V, r.M);

    return r;
}

// =================================================================================
// Fast Partial Accessors
// =================================================================================

// Optimized load for wsum only
float load_wsum(RWByteAddressBuffer buf, uint pixelIdx)
{
    // wsum is in O_GI_PACK3 at offset .y (4 bytes in)
    return asfloat(buf.Load(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3 + 4u));
}

// Optimized store for wsum only
void store_wsum(RWByteAddressBuffer buf, uint pixelIdx, float wsum)
{
    buf.Store(pixelBaseAddrGI(pixelIdx) + O_GI_PACK3 + 4u, asuint(wsum));
}

// Fast loader for current sample geometry (Sample 0)
// Used for ReSTIR geometric similarity
void load_Geom0_fast(RWByteAddressBuffer buf, uint pixelIdx,
                     out float3 xn0, out float3 nn0, out uint objID0)
{
    uint base = pixelBaseAddrGI(pixelIdx);
    uint4 p1 = buf.Load4(base + O_GI_PACK1); // xn0, nn0
    xn0 = asfloat(p1.xyz);
    nn0 = UnpackNormal(p1.w);

    // objID0 is in PACK4.x
    objID0 = buf.Load(base + O_GI_PACK4);
}