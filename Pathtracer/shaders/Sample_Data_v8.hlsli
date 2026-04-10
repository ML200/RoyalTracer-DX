/*
    Compact G-buffer: 32 bytes per pixel.
    Stores minimal identifiers + cached normals/UV to avoid scattered vertex
    buffer access during per-neighbor MIS evaluation.

    Layout (per pixel):
        Offset  0: instID (bits 0-15) + emitter flag (bit 31)   [uint32]
        Offset  4: primID (FlatPrimID)                           [uint32]
        Offset  8: bary.x                                        [float32]
        Offset 12: bary.y                                        [float32]
        Offset 16: etai / etat                                   [half2 -> uint32]
        Offset 20: n1_g (geometric normal, object space, packed) [uint32]
        Offset 24: n1_s (shading normal, object space, packed)   [uint32]
        Offset 28: uv (texture coordinates)                      [half2 -> uint32]
    Total: 32 bytes.

    n1_s + uv are cached to allow BuildVertexLight (neighbor reconstruction)
    to skip 12 scattered vertex-buffer loads (normals + UVs) per neighbor.

    Emitter flag (bit 31 of word 0):
        Set when the primary hit is an emitter or sky/miss.
        When set, L1 has already been written to gScratchPing[pixel, 1] and [pixel, 2]
        by the raygen/hit shader.  Temporal/spatial passes check this flag and skip.

    Sky sentinel:
        instID == 0xFFFF with emitter flag set -> sky/miss (no surface geometry).
*/

static const uint BYTES_SD = 32u;

uint pixelBaseAddr_SD(uint pixelIdx)
{
    return pixelIdx * BYTES_SD;
}

// -- OtW / WtO helpers --
float3 WorldToObjectPos(uint id, float3 Pw)
{
    return mul(instanceProps[id].objectToWorldInverse, float4(Pw, 1.0)).xyz;
}
float3 ObjectToWorldPos(uint id, float3 Po)
{
    return mul(instanceProps[id].objectToWorld, float4(Po, 1.0)).xyz;
}
float3 ObjectToWorldNrm(uint id, float3 No)
{
    return normalize(mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)).xyz);
}
float3 WorldToObjectNrm(uint id, float3 Nw)
{
    float3x3 MT = transpose((float3x3)instanceProps[id].objectToWorld);
    return normalize(mul(MT, Nw));
}

// -- Individual stores --

// Store instID (bits 0-15) + emitter flag (bit 31)
void store_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID, bool isEmitter)
{
    uint packed = (instID & 0xFFFFu) | (isEmitter ? 0x80000000u : 0u);
    buf.Store(pixelBaseAddr_SD(pixelIdx), packed);
}

void store_primID(RWByteAddressBuffer buf, uint pixelIdx, uint primID)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 4u, primID);
}

void store_bary(RWByteAddressBuffer buf, uint pixelIdx, float2 bary)
{
    buf.Store2(pixelBaseAddr_SD(pixelIdx) + 8u, asuint(bary));
}

void store_etai_etat(RWByteAddressBuffer buf, uint pixelIdx, float etai, float etat)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 16u, PackFloat2x16(etai, etat));
}

// Store geometric normal in object space (for fast rejection + visibility)
void store_n1_g_world(RWByteAddressBuffer buf, uint pixelIdx, float3 n1g_world, uint instID)
{
    float3 n1g_obj = (instID < 0xFFFEu) ? WorldToObjectNrm(instID, n1g_world) : n1g_world;
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 20u, PackNormal(n1g_obj));
}

// Store shading normal in object space (cached for neighbor MIS)
void store_n1_s_world(RWByteAddressBuffer buf, uint pixelIdx, float3 n1s_world, uint instID)
{
    float3 n1s_obj = (instID < 0xFFFEu) ? WorldToObjectNrm(instID, n1s_world) : n1s_world;
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 24u, PackNormal(n1s_obj));
}

// Store UV (cached for neighbor MIS)
void store_uv(RWByteAddressBuffer buf, uint pixelIdx, float2 uv)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 28u, PackFloat2x16(uv.x, uv.y));
}

// Bulk store for sky/miss (zeros everything except instID+emitter flag)
void store_sky(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    buf.Store4(base, uint4(0xFFFFu | 0x80000000u, 0u, 0u, 0u));
    buf.Store4(base + 16u, uint4(PackFloat2x16(1.0f, 1.0f), 0u, 0u, 0u));
}

// -- Individual loads --

uint load_instID_raw(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx));
}

uint load_instID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx)) & 0xFFFFu;
}

bool load_isEmitter(RWByteAddressBuffer buf, uint pixelIdx)
{
    return (buf.Load(pixelBaseAddr_SD(pixelIdx)) & 0x80000000u) != 0u;
}

uint load_primID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u);
}

float2 load_bary(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load2(pixelBaseAddr_SD(pixelIdx) + 8u));
}

float load_etai(RWByteAddressBuffer buf, uint pixelIdx)
{
    return f16tof32_custom(buf.Load(pixelBaseAddr_SD(pixelIdx) + 16u) & 0xFFFFu);
}

float load_etat(RWByteAddressBuffer buf, uint pixelIdx)
{
    return f16tof32_custom(buf.Load(pixelBaseAddr_SD(pixelIdx) + 16u) >> 16);
}

// Load geometric normal: stored in object space, returned in world space
float3 load_n1_g(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint instID = load_instID(buf, pixelIdx);
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 20u));
    return (instID < 0xFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

// Load geometric normal given an already-loaded instID (avoids redundant load)
float3 load_n1_g_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 20u));
    return (instID < 0xFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

// Load shading normal: stored in object space, returned in world space
float3 load_n1_s(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint instID = load_instID(buf, pixelIdx);
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 24u));
    return (instID < 0xFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

float3 load_n1_s_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 24u));
    return (instID < 0xFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

// Load UV (half2)
float2 load_uv(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint packed = buf.Load(pixelBaseAddr_SD(pixelIdx) + 28u);
    float a, b;
    UnpackFloat2x16(packed, a, b);
    return float2(a, b);
}

// Copy sample data for temporal reuse
void copySampleData(RWByteAddressBuffer dst, RWByteAddressBuffer src, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    dst.Store4(base,       src.Load4(base));
    dst.Store4(base + 16u, src.Load4(base + 16u));
}
