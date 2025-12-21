// ============================================================================
// PayloadPath_v8.hlsli
// SM 6.9 compatible tiny path payload + pack/unpack helpers + raygen initializer
// ============================================================================

#ifndef PAYLOAD_PATH_V1_HLSLI
#define PAYLOAD_PATH_V1_HLSLI

// --------------------------------------------
// Flags (stored in meta1, 10 bits available)
// --------------------------------------------
static const uint PF_TERMINATE      = (1u << 0);
static const uint PF_TRANSMIT_BACK  = (1u << 1); // dot(Ng, s) < 0
static const uint PF_FIRST_BOUNCE   = (1u << 2);

// Sentinel for "no medium" (fits in 15 bits)
static const uint MEDIUM_INVALID_15 = 0x7FFFu;

// --------------------------------------------
// Half2 pack/unpack (for iors) - uniquely named
// --------------------------------------------
inline uint PackHalf2_payload(float2 v)
{
    return (uint(f32tof16(v.x)) & 0xFFFFu) | (uint(f32tof16(v.y)) << 16);
}

inline float2 UnpackHalf2_payload(uint u)
{
    return float2(f16tof32(u & 0xFFFFu), f16tof32(u >> 16));
}

// --------------------------------------------
// Shading normal packing (octahedral snorm16x2) - uniquely named
// Packed into a uint: low16 = x, high16 = y
// --------------------------------------------
inline uint PackOctSnorm16_payload(float3 n)
{
    n = normalize(n);
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-20f);

    float2 p = (n.z >= 0.0f) ? n.xy : (1.0f - abs(n.yx)) * sign(n.xy);
    p = clamp(p, -1.0f, 1.0f);

    int2 q = int2(round(p * 32767.0f));
    return (uint(q.x) & 0xFFFFu) | (uint(q.y) << 16);
}

inline float3 UnpackOctSnorm16_payload(uint u)
{
    int x = (int)(u << 16) >> 16; // sign-extend low16
    int y = (int)u >> 16;         // sign-extend high16

    float2 p = float2(x, y) / 32767.0f;

    float3 n = float3(p.x, p.y, 1.0f - abs(p.x) - abs(p.y));
    float t = clamp(-n.z, 0.0f, 1.0f);
    n.xy += sign(n.xy) * t;

    return normalize(n);
}

// --------------------------------------------
// Direction representation: float2 oct (no quantization) - uniquely named
// Helpers for encode/decode float3 <-> float2 (ray dir stored as float2)
// --------------------------------------------
inline float2 OctEncodeFloat2_payload(float3 n)
{
    n = normalize(n);
    n /= (abs(n.x) + abs(n.y) + abs(n.z) + 1e-20f);
    return (n.z >= 0.0f) ? n.xy : (1.0f - abs(n.yx)) * sign(n.xy);
}

inline float3 OctDecodeFloat2_payload(float2 e)
{
    float3 n = float3(e.x, e.y, 1.0f - abs(e.x) - abs(e.y));
    float t = clamp(-n.z, 0.0f, 1.0f);
    n.xy += sign(n.xy) * t;
    return normalize(n);
}

// --------------------------------------------
// Metadata packing - uniquely named
//
// Fields:
// - depth:        uint5
// - matID:        uint15
// - mediumMatID:  uint15
// - iorPointer:   int in [-1..3] => 3 bits with bias (0..4)
// - flags:        up to 10 bits in meta1
//
// meta0 (32):
//  0..4   depth (5)
//  5..7   iorPtrBiased (3)  (-1..3 -> 0..4)
//  8..22  matID (15)
// 23..31  mediumID low 9
//
// meta1 (32):
//  0..5   mediumID high 6
//  6..15  flags (10)
// 16..31  spare
// --------------------------------------------
inline uint PackIorPtr_payload(int iorPtr) { return (uint)(iorPtr + 1); } // -1->0, 0->1, 3->4
inline int  UnpackIorPtr_payload(uint b)   { return (int)b - 1; }

inline uint PackMeta0_payload(uint depth5, int iorPtr, uint mat15, uint medium15)
{
    uint d   = depth5 & 0x1Fu;
    uint ip  = PackIorPtr_payload(iorPtr) & 0x7u;
    uint m   = mat15 & 0x7FFFu;
    uint lo9 = medium15 & 0x1FFu;

    return (d) | (ip << 5) | (m << 8) | (lo9 << 23);
}

inline uint PackMeta1_payload(uint medium15, uint flags10)
{
    uint hi6 = (medium15 >> 9) & 0x3Fu;
    uint fl  = flags10 & 0x3FFu;
    return (hi6) | (fl << 6);
}

inline void UnpackMeta_payload(uint meta0, uint meta1,
                               out uint depth5, out int iorPtr, out uint mat15, out uint medium15, out uint flags10)
{
    depth5 = meta0 & 0x1Fu;
    iorPtr = UnpackIorPtr_payload((meta0 >> 5) & 0x7u);
    mat15  = (meta0 >> 8) & 0x7FFFu;

    uint lo9 = (meta0 >> 23) & 0x1FFu;
    uint hi6 = meta1 & 0x3Fu;
    medium15 = (hi6 << 9) | lo9;

    flags10 = (meta1 >> 6) & 0x3FFu;
}

// Convenience “stores” - uniquely named
inline void SetDepth_payload(inout uint meta0, uint depth5)
{
    meta0 = (meta0 & ~0x1Fu) | (depth5 & 0x1Fu);
}

inline void SetMatID_payload(inout uint meta0, uint mat15)
{
    meta0 = (meta0 & ~(0x7FFFu << 8)) | ((mat15 & 0x7FFFu) << 8);
}

inline void SetIorPtr_payload(inout uint meta0, int iorPtr)
{
    meta0 = (meta0 & ~(0x7u << 5)) | ((PackIorPtr_payload(iorPtr) & 0x7u) << 5);
}

inline void SetMediumID_payload(inout uint meta0, inout uint meta1, uint medium15)
{
    uint lo9 = (medium15 & 0x1FFu);
    uint hi6 = (medium15 >> 9) & 0x3Fu;

    meta0 = (meta0 & ~(0x1FFu << 23)) | (lo9 << 23);
    meta1 = (meta1 & ~0x3Fu) | hi6;
}

inline uint GetFlags_payload(uint meta1) { return (meta1 >> 6) & 0x3FFu; }

inline void SetFlags_payload(inout uint meta1, uint flags10)
{
    meta1 = (meta1 & ~(0x3FFu << 6)) | ((flags10 & 0x3FFu) << 6);
}

inline void AddFlags_payload(inout uint meta1, uint flags10)
{
    uint f = GetFlags_payload(meta1);
    SetFlags_payload(meta1, f | (flags10 & 0x3FFu));
}

inline void ClearFlags_payload(inout uint meta1, uint flags10)
{
    uint f = GetFlags_payload(meta1);
    SetFlags_payload(meta1, f & ~(flags10 & 0x3FFu));
}

// --------------------------------------------
// Payload (SM 6.9): [raypayload]
// --------------------------------------------
struct [raypayload] PathRayPayload
{
    // in/out: current direction oct float2 (no quantization)
    float2 dir2             : read(caller) : write(closesthit, miss);

    // out: packed shading normal (Ns) for raygen origin offset, etc.
    uint   packedNs         : read(caller) : write(closesthit, miss);

    // in: packed state; out: below surface check
    uint   meta0            : read(caller) : write(closesthit, miss);
    uint   meta1            : read(caller) : write(closesthit, miss);

    // in/out: RNG
    uint   seed             : read(caller) : write(closesthit, miss);
    // in IORs
    uint   iorsPacked       : read(caller) : write(closesthit, miss);

    // Throughput (RGB9E5)
    uint   packedThroughput : read(caller) : write(closesthit, miss);

    // out: BSDF PDF (Uncompressed)
    float  bsdfPdf          : read(caller) : write(closesthit, miss);
};

// --------------------------------------------
// Uncompressed Payload Struct
// --------------------------------------------
struct PathPayloadUncompressed
{
    float3 dir;         // Expanded from dir2 (oct encoded)
    float3 normal;      // Expanded from packedNs (snorm16x2 oct)
    uint   depth;       // From meta0 (5 bits)
    int    iorPointer;  // From meta0 (3 bits, biased)
    uint   matID;       // From meta0 (15 bits)
    uint   mediumMatID; // From meta0/meta1 (15 bits)
    uint   flags;       // From meta1 (10 bits)
    uint   seed;        // Direct copy
    float2 iors;        // Expanded from iorsPacked (half2)
    float3 throughput;  // Expanded from packedThroughput (RGB9E5)
    float  bsdfPdf;     // Direct copy
};

// --------------------------------------------
// Raygen initializer: takes float3 dir and stores float2 oct
// --------------------------------------------
inline PathRayPayload InitPayload_Raygen_payload(
    float3 dir,
    float3 normal,
    uint   depth,
    int    iorPointer,
    uint   matID,
    uint   mediumMatID,
    uint   flags,
    uint   seed,
    float2 iors,
    float3 throughput,
    float  bsdfPdf
)
{
    PathRayPayload p = (PathRayPayload)0;

    // Direction and Normal
    p.dir2     = OctEncodeFloat2_payload(dir);
    p.packedNs = PackOctSnorm16_payload(normal);

    // Metadata
    p.meta0 = PackMeta0_payload(depth, iorPointer, matID, mediumMatID);
    p.meta1 = PackMeta1_payload(mediumMatID, flags);

    // RNG + IORs
    p.seed       = seed;
    p.iorsPacked = PackHalf2_payload(iors);

    // Throughput
    p.packedThroughput = PackRGB9E5(throughput);

    // BSDF PDF
    p.bsdfPdf = bsdfPdf;

    return p;
}

// --------------------------------------------
// Load: Compressed -> Uncompressed
// --------------------------------------------
inline PathPayloadUncompressed UnpackPayload_payload(PathRayPayload packed)
{
    PathPayloadUncompressed u;

    // 1. Decode Direction and Normal
    u.dir    = OctDecodeFloat2_payload(packed.dir2);
    u.normal = UnpackOctSnorm16_payload(packed.packedNs);

    // 2. Unpack Metadata
    // Note: meta0/meta1 are split inside the helper
    UnpackMeta_payload(packed.meta0, packed.meta1,
                       u.depth, u.iorPointer, u.matID, u.mediumMatID, u.flags);

    // 3. Unpack RNG and IORs
    u.seed = packed.seed;
    u.iors = UnpackHalf2_payload(packed.iorsPacked);

    // 4. Unpack Throughput
    u.throughput = UnpackRGB9E5(packed.packedThroughput);

    // 5. BSDF PDF
    u.bsdfPdf = packed.bsdfPdf;

    return u;
}

// --------------------------------------------
// Pack: Uncompressed -> Compressed
// --------------------------------------------
inline PathRayPayload PackPayload_payload(PathPayloadUncompressed u)
{
    PathRayPayload p;

    // 1. Encode Direction and Normal
    p.dir2     = OctEncodeFloat2_payload(u.dir);
    p.packedNs = PackOctSnorm16_payload(u.normal);

    // 2. Pack Metadata
    // Masking is handled inside PackMeta functions, but inputs are passed directly
    p.meta0 = PackMeta0_payload(u.depth, u.iorPointer, u.matID, u.mediumMatID);
    p.meta1 = PackMeta1_payload(u.mediumMatID, u.flags);

    // 3. Pack RNG and IORs
    p.seed       = u.seed;
    p.iorsPacked = PackHalf2_payload(u.iors);

    // 4. Pack Throughput
    p.packedThroughput = PackRGB9E5(u.throughput);

    // 5. BSDF PDF
    p.bsdfPdf = u.bsdfPdf;

    return p;
}
#endif // PAYLOAD_PATH_V1_HLSLI