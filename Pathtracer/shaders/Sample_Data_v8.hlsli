//====================================
//COMPACT G-BUFFER 24 BYTES PER PIXEL
//====================================
//cached normals/UV so BuildVertexLight skips scattered vertex loads per neighbor
//offset 0  instID, offset 4 primID bits 0-30 + emitter flag bit 31
//offset 8  bary.x, offset 12 bary.y
//offset 16 n1_s object-space packed, offset 20 uv half2
//emitter flag set, primary hit is emitter or sky, L1 already in gScratchPing
//sky sentinel, instID=0xFFFFFFFF with emitter flag

static const uint BYTES_SD = 24u;

uint pixelBaseAddr_SD(uint pixelIdx)
{
    return pixelIdx * BYTES_SD;
}

//====================================
//OBJECT-WORLD HELPERS
//====================================
//env/miss sentinels skip the transform, triangle-light uses its real instID
float3 WorldToObjectPos(uint id, float3 Pw)
{
    if (id >= MATID_LIGHT_TRI) return Pw;
    return mul(instanceProps[id].objectToWorldInverse, float4(Pw, 1.0)).xyz;
}
float3 ObjectToWorldPos(uint id, float3 Po)
{
    if (id >= MATID_LIGHT_TRI) return Po;
    return mul(instanceProps[id].objectToWorld, float4(Po, 1.0)).xyz;
}
float3 ObjectToWorldNrm(uint id, float3 No)
{
    if (id >= MATID_LIGHT_TRI) return No;
    return normalize(mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)).xyz);
}
float3 WorldToObjectNrm(uint id, float3 Nw)
{
    if (id >= MATID_LIGHT_TRI) return Nw;
    float3x3 MT = transpose((float3x3)instanceProps[id].objectToWorld);
    return normalize(mul(MT, Nw));
}

//====================================
//STORES
//====================================

void store_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx), instID);
}

void store_primID(RWByteAddressBuffer buf, uint pixelIdx, uint primID, bool isEmitter)
{
    uint packed = (primID & 0x7FFFFFFFu) | (isEmitter ? 0x80000000u : 0u);
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 4u, packed);
}

void store_bary(RWByteAddressBuffer buf, uint pixelIdx, float2 bary)
{
    buf.Store2(pixelBaseAddr_SD(pixelIdx) + 8u, asuint(bary));
}

void store_n1_s_world(RWByteAddressBuffer buf, uint pixelIdx, float3 n1s_world, uint instID)
{
    float3 n1s_obj = (instID < 0xFFFFFFFEu) ? WorldToObjectNrm(instID, n1s_world) : n1s_world;
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 16u, PackNormal(n1s_obj));
}

void store_uv(RWByteAddressBuffer buf, uint pixelIdx, float2 uv)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 20u, PackFloat2x16(uv.x, uv.y));
}

void store_sky(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    buf.Store4(base, uint4(0xFFFFFFFFu, 0x80000000u, 0u, 0u));
    buf.Store2(base + 16u, uint2(0u, 0u));
}

//====================================
//LOADS
//====================================
uint load_instID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx));
}

bool load_isEmitter(RWByteAddressBuffer buf, uint pixelIdx)
{
    return (buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u) & 0x80000000u) != 0u;
}

uint load_primID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u) & 0x7FFFFFFFu;
}

float2 load_bary(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load2(pixelBaseAddr_SD(pixelIdx) + 8u));
}

float3 load_n1_s(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint instID = load_instID(buf, pixelIdx);
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 16u));
    return (instID < 0xFFFFFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

float3 load_n1_s_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 16u));
    return (instID < 0xFFFFFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

float2 load_uv(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint packed = buf.Load(pixelBaseAddr_SD(pixelIdx) + 20u);
    float a, b;
    UnpackFloat2x16(packed, a, b);
    return float2(a, b);
}

//====================================
//SAMPLE DATA COPY FOR TEMPORAL REUSE
//====================================
void copySampleData(RWByteAddressBuffer dst, RWByteAddressBuffer src, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    dst.Store4(base,       src.Load4(base));
    dst.Store2(base + 16u, src.Load2(base + 16u));
}
