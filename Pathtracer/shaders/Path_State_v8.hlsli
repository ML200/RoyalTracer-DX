// =================================================================================
// 1. DATA STRUCTURE DEFINITIONS (UNPACKED)
// =================================================================================

// STREAM 0: Geometry (Used for Traversal)
struct RayGeometry {
    float3 origin;
    float3 dir;
};

// STREAM 1: Payload (Used for Shading & RNG)
struct PathPayload {
    float3 throughput;
    uint   seed;
};

// STREAM 2: Volume IOR (Used for Refraction/Fresnel)
struct VolumeIOR {
    float ior_stack[4]; // Range [1.0, 3.0]
    int   pointer;      // Range [-1, 3]
};

// STREAM 3: Volume Aux (Used for Medium Logic)
struct VolumeAux {
    uint matID_stack[4];
    uint objID_stack[4];
};

// STREAM 4: Hit Info (Used for Visibility/Shading)
struct HitState {
    uint   instanceID;
    uint   primitiveID;
    float2 bary;
};

// =================================================================================
// 2. MEMORY LAYOUT CONSTANTS
// =================================================================================

static uint GetTotalPixels() { return gImageSize.x * gImageSize.y; }

// --- STREAM 0: RAY GEOMETRY ---
// Layout: [ Origin.xyz (12B) | PackedDir (4B) ] = 16 Bytes (float4)
static const uint STRIDE_RAY_GEO = 16;
static const uint OFF_RAY_GEO    = 0;

// --- STREAM 1: PATH PAYLOAD ---
// Layout: [ PackedThroughput (4B) | Seed (4B) ] = 8 Bytes (uint2)
static const uint STRIDE_PAYLOAD = 8;
static const uint OFF_PAYLOAD    = 16; 

// --- STREAM 2: VOLUME IOR ---
// Layout: [ Packed IORs + Pointer ] = 8 Bytes (uint2)
static const uint STRIDE_VOL_IOR = 8;
static const uint OFF_VOL_IOR    = 24;

// --- STREAM 3: VOLUME AUX ---
// Layout: [ Packed Mats (4B) | Packed Mats (4B) | Packed Prio (4B) ] = 12 Bytes (uint3)
static const uint STRIDE_VOL_MAT = 12;
static const uint OFF_VOL_MAT    = 32;

// --- STREAM 4: HIT INFO ---
// Layout: [ Instance (4B) | Prim (4B) | Bary (8B) ] = 16 Bytes (uint4)
static const uint STRIDE_HIT     = 16;
static const uint OFF_HIT        = 44;

// TOTAL BYTES PER PIXEL: 60

// Address Calculation Helper
uint GetAddr(uint pixelIdx, uint streamOffset, uint streamStride) {
    return (GetTotalPixels() * streamOffset) + (pixelIdx * streamStride);
}

// =================================================================================
// 3. PACKING HELPERS
// =================================================================================

// Packs 2 floats (Range 0.0-4.0) and hides 2 bits of the pointer
static uint PackIORPair(float fA, float fB, uint pBits) {
    // Range 0.0 to 4.0 allows us to store 0.0 exactly.
    // Scale: f / 4.0.
    // Precision: 4.0 / 32767 ~= 0.00012 (Sufficient for IOR)

    uint iA = (uint)(saturate(fA * 0.25f) * 32767.0f) & 0x7FFF;
    uint iB = (uint)(saturate(fB * 0.25f) * 32767.0f) & 0x7FFF;

    // Hide pointer bits in MSB (Bit 15 and Bit 31)
    if ((pBits & 1) != 0) iA |= 0x8000;
    if ((pBits & 2) != 0) iB |= 0x8000;

    return iA | (iB << 16);
}

// Unpacks 2 floats (Range 0.0-4.0) and retrieves 2 bits of the pointer
static float2 UnpackIORPair(uint raw, out uint bits) {
    uint bitA = (raw >> 15) & 1;
    uint bitB = (raw >> 31) & 1;

    // Combine the hidden bits
    bits = bitA | (bitB << 1);

    // Unpack float (Mask out the pointer bit first)
    float fA = (float)(raw & 0x7FFF) / 32767.0f;
    float fB = (float)((raw >> 16) & 0x7FFF) / 32767.0f;

    // Scale back to 0.0 - 4.0 range
    return float2(fA * 4.0f, fB * 4.0f);
}

// --- PACKING: IOR Stack (15-bit) + Pointer (Hidden in MSBs) ---
// IOR Range: [1.0, 3.0] mapped to 15 bits.
// Pointer: [-1..3] mapped to [0..4] (Requires 3 bits).
uint2 PackIORStackAndPtr(float stack[4], int ptr) {
    uint p = (uint)(ptr + 1) & 0xF; // Map -1->0, 3->4.

    uint2 packed;
    // Pack indices 0 and 1, hiding bits 0 and 1 of p
    packed.x = PackIORPair(stack[0], stack[1], p & 0x3);

    // Pack indices 2 and 3, hiding bits 2 and 3 of p
    packed.y = PackIORPair(stack[2], stack[3], (p >> 2) & 0x3);

    return packed;
}

void UnpackIORStackAndPtr(uint2 packed, out float stack[4], out int ptr) {
    uint bits01;
    uint bits23;

    // Unpack pair 1 (Float 0, Float 1, Bits 0-1)
    float2 v01 = UnpackIORPair(packed.x, bits01);
    stack[0] = v01.x;
    stack[1] = v01.y;

    // Unpack pair 2 (Float 2, Float 3, Bits 2-3)
    float2 v23 = UnpackIORPair(packed.y, bits23);
    stack[2] = v23.x;
    stack[3] = v23.y;

    // Reconstruct pointer p by combining bits
    uint p = bits01 | (bits23 << 2);
    
    ptr = (int)p - 1;
}

// --- PACKING: Material IDs (16-bit) ---
uint2 PackMatStack16(uint stack[4]) {
    return uint2((stack[0] & 0xFFFF) | (stack[1] << 16),
                 (stack[2] & 0xFFFF) | (stack[3] << 16));
}

void UnpackMatStack16(uint2 packed, out uint stack[4]) {
    stack[0] = packed.x & 0xFFFF; stack[1] = packed.x >> 16;
    stack[2] = packed.y & 0xFFFF; stack[3] = packed.y >> 16;
}

// --- PACKING: Priorities (8-bit) ---
uint PackPrioStack8(uint stack[4]) {
    return (stack[0] & 0xFF) | ((stack[1] & 0xFF) << 8) |
           ((stack[2] & 0xFF) << 16) | ((stack[3] & 0xFF) << 24);
}

void UnpackPrioStack8(uint packed, out uint stack[4]) {
    stack[0] = packed & 0xFF; stack[1] = (packed >> 8) & 0xFF;
    stack[2] = (packed >> 16) & 0xFF; stack[3] = (packed >> 24) & 0xFF;
}

// =================================================================================
// 4. ACCESSORS (LOAD / STORE)
// =================================================================================

// --- STREAM 0: RAY GEOMETRY ---
RayGeometry LoadRayGeometry(RWByteAddressBuffer buf, uint pixelIdx)
{
    RayGeometry rg;
    uint4 raw = buf.Load4(GetAddr(pixelIdx, OFF_RAY_GEO, STRIDE_RAY_GEO));
    rg.origin = asfloat(raw.xyz);
    rg.dir    = UnpackNormal(raw.w);

    return rg;
}

void StoreRayGeometry(RWByteAddressBuffer buf, uint pixelIdx, RayGeometry rg)
{
    uint4 raw;
    raw.xyz = asuint(rg.origin);
    raw.w   = PackNormal(rg.dir);
    buf.Store4(GetAddr(pixelIdx, OFF_RAY_GEO, STRIDE_RAY_GEO), raw);
}

// --- STREAM 1: PATH PAYLOAD ---
PathPayload LoadPathPayload(RWByteAddressBuffer buf, uint pixelIdx)
{
    PathPayload pp;
    uint2 raw = buf.Load2(GetAddr(pixelIdx, OFF_PAYLOAD, STRIDE_PAYLOAD));
    pp.throughput = UnpackRGB9E5(raw.x);
    pp.seed       = raw.y;
    return pp;
}

void StorePathPayload(RWByteAddressBuffer buf, uint pixelIdx, PathPayload pp)
{
    uint2 raw;
    raw.x = PackRGB9E5(pp.throughput);
    raw.y = pp.seed;
    buf.Store2(GetAddr(pixelIdx, OFF_PAYLOAD, STRIDE_PAYLOAD), raw);
}

// --- STREAM 2: VOLUME IOR ---
VolumeIOR LoadVolumeIOR(RWByteAddressBuffer buf, uint pixelIdx)
{
    VolumeIOR v;
    uint2 raw = buf.Load2(GetAddr(pixelIdx, OFF_VOL_IOR, STRIDE_VOL_IOR));
    UnpackIORStackAndPtr(raw, v.ior_stack, v.pointer);
    return v;
}

void StoreVolumeIOR(RWByteAddressBuffer buf, uint pixelIdx, VolumeIOR v)
{
    uint2 raw = PackIORStackAndPtr(v.ior_stack, v.pointer);
    buf.Store2(GetAddr(pixelIdx, OFF_VOL_IOR, STRIDE_VOL_IOR), raw);
}

// --- STREAM 3: VOLUME AUX ---
VolumeAux LoadVolumeAux(RWByteAddressBuffer buf, uint pixelIdx)
{
    VolumeAux v;
    uint3 raw = buf.Load3(GetAddr(pixelIdx, OFF_VOL_MAT, STRIDE_VOL_MAT));
    UnpackMatStack16(raw.xy, v.matID_stack);
    UnpackPrioStack8(raw.z, v.objID_stack);
    return v;
}

void StoreVolumeAux(RWByteAddressBuffer buf, uint pixelIdx, VolumeAux v)
{
    uint3 raw;
    raw.xy = PackMatStack16(v.matID_stack);
    raw.z  = PackPrioStack8(v.objID_stack);
    buf.Store3(GetAddr(pixelIdx, OFF_VOL_MAT, STRIDE_VOL_MAT), raw);
}

// --- STREAM 4: HIT INFO ---
HitState LoadHitState(RWByteAddressBuffer buf, uint pixelIdx)
{
    HitState h;
    uint4 raw = buf.Load4(GetAddr(pixelIdx, OFF_HIT, STRIDE_HIT));
    h.instanceID  = raw.x;
    h.primitiveID = raw.y;
    h.bary        = asfloat(raw.zw);
    return h;
}

void StoreHitState(RWByteAddressBuffer buf, uint pixelIdx, HitState h)
{
    uint4 raw;
    raw.x  = h.instanceID;
    raw.y  = h.primitiveID;
    raw.zw = asuint(h.bary);
    buf.Store4(GetAddr(pixelIdx, OFF_HIT, STRIDE_HIT), raw);
}


// -------------------------------------------------------------
// Partial load helpers

// Optimized accessor to fetch just the throughput without the seed.
float3 LoadPathThroughput(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint packed = buf.Load(GetAddr(pixelIdx, OFF_PAYLOAD, STRIDE_PAYLOAD));
    return UnpackRGB9E5(packed);
}

// Optimized accessor to fetch just the ray origin.
float3 LoadRayOrigin(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint3 raw = buf.Load3(GetAddr(pixelIdx, OFF_RAY_GEO, STRIDE_RAY_GEO));
    return asfloat(raw);
}