//====================================
//COMPACT G-BUFFER 36 BYTES PER PIXEL
//====================================
//Resolved primary-hit surface for every pass AFTER Pass_raygen. Raygen bakes
//the fully texture-resolved material here, so no later pass ever touches
//per-triangle data (indices / BTriVertex / materialIDs / instanceProps) or
//re-samples a material texture. That is what lets streamed planet terrain -
//which has no per-triangle data at shade time - shade through the exact same
//path as scene meshes: the surface is reconstructed from what is stored here,
//never from (instID, primID, bary).
//
//offset 0  instID  scene index, terrain >= TERRAIN_INSTANCE_BASE, sky 0xFFFFFFFF
//offset 4  flags    bit0 isEmitter, bit1 backface
//offset 8  matID    resolved material id (sentinels: env / light-tri / terrain)
//offset 12 Kd       texture-resolved albedo, RGB9E5
//offset 16 PrPm     texture-resolved roughness + metalness, half2
//offset 20 n1_s     shading normal (normal-mapped), object-space packed
//offset 24 x1       exact world hit position (float3) - baked by raygen
//
//sky sentinel, instID=0xFFFFFFFF with the isEmitter flag set, matID=env-miss

static const uint BYTES_SD = 36u;

//flag bits in the offset-4 word
static const uint SD_FLAG_EMITTER  = 1u;
static const uint SD_FLAG_BACKFACE = 2u;

uint pixelBaseAddr_SD(uint pixelIdx)
{
    return pixelIdx * BYTES_SD;
}

//====================================
//OBJECT-WORLD HELPERS
//====================================
//Only the env/miss sentinel (instID 0xFFFFFFFF) skips the transform; every
//real instance - scene mesh AND terrain cell - has an instanceProps entry.
//(Terrain's transform is a pure translation, so its normal transform is the
//identity, but it still goes through the same path.)
float3 WorldToObjectPos(uint id, float3 Pw)
{
    if (id == 0xFFFFFFFFu) return Pw;
    return mul(instanceProps[id].objectToWorldInverse, float4(Pw, 1.0)).xyz;
}
float3 ObjectToWorldPos(uint id, float3 Po)
{
    if (id == 0xFFFFFFFFu) return Po;
    return mul(instanceProps[id].objectToWorld, float4(Po, 1.0)).xyz;
}
float3 ObjectToWorldNrm(uint id, float3 No)
{
    if (id == 0xFFFFFFFFu) return No;
    return normalize(mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)).xyz);
}
float3 WorldToObjectNrm(uint id, float3 Nw)
{
    if (id == 0xFFFFFFFFu) return Nw;
    float3x3 MT = transpose((float3x3)instanceProps[id].objectToWorld);
    return normalize(mul(MT, Nw));
}

//====================================
//STORES
//====================================

void store_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 0u, instID);
}

//bit0 isEmitter, bit1 backface. backface lets the NRC debug query pass the
//correct side bit at the primary hit, matching what raygen feeds NRC.
void store_flags(RWByteAddressBuffer buf, uint pixelIdx, bool isEmitter, bool backface)
{
    uint f = (isEmitter ? SD_FLAG_EMITTER  : 0u)
           | (backface  ? SD_FLAG_BACKFACE : 0u);
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 4u, f);
}

void store_matID(RWByteAddressBuffer buf, uint pixelIdx, uint matID)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 8u, matID);
}

void store_kd(RWByteAddressBuffer buf, uint pixelIdx, float3 kd)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 12u, PackRGB9E5(kd));
}

void store_prpm(RWByteAddressBuffer buf, uint pixelIdx, float pr, float pm)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 16u, PackFloat2x16(pr, pm));
}

void store_n1_s_world(RWByteAddressBuffer buf, uint pixelIdx, float3 n1s_world, uint instID)
{
    float3 n1s_obj = WorldToObjectNrm(instID, n1s_world);   // sentinel handled inside
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 20u, PackNormal(n1s_obj));
}

//x1 is stored in OBJECT space (like x2/n2), NOT raw shifted-world, so it
//survives a floating-origin snap: the world position is reconstructed from the
//CURRENT objectToWorld at load time. Storing shifted-world here froze the point
//in the origin it was written in, so cross-frame (temporal) reuse connected a
//current-origin shading point to a last-origin x1 - a full snap-quantum apart -
//and every reconnection/visibility ray missed once the camera left its spawn
//cell. The env-miss sentinel (instID 0xFFFFFFFF) passes through unchanged.
void store_x1(RWByteAddressBuffer buf, uint pixelIdx, float3 x1, uint instID)
{
    float3 x1_obj = WorldToObjectPos(instID, x1);   // sentinel handled inside
    buf.Store3(pixelBaseAddr_SD(pixelIdx) + 24u, asuint(x1_obj));
}

void store_sky(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    buf.Store4(base,       uint4(0xFFFFFFFFu, SD_FLAG_EMITTER, MATID_ENV_MISS, 0u));
    buf.Store4(base + 16u, uint4(0u, 0u, 0u, 0u));
    buf.Store (base + 32u, 0u);
}

//====================================
//LOADS
//====================================
uint load_instID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx) + 0u);
}

bool load_isEmitter(RWByteAddressBuffer buf, uint pixelIdx)
{
    return (buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u) & SD_FLAG_EMITTER) != 0u;
}

//raw flags word - lets a pass test SD_FLAG_EMITTER and SD_FLAG_BACKFACE
//from one fetch instead of two separate loads of the same word
uint load_flagsWord(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u);
}

bool load_backface(RWByteAddressBuffer buf, uint pixelIdx)
{
    return (buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u) & SD_FLAG_BACKFACE) != 0u;
}

uint load_matID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx) + 8u);
}

float3 load_kd(RWByteAddressBuffer buf, uint pixelIdx)
{
    return UnpackRGB9E5(buf.Load(pixelBaseAddr_SD(pixelIdx) + 12u));
}

void load_prpm(RWByteAddressBuffer buf, uint pixelIdx, out float pr, out float pm)
{
    UnpackFloat2x16(buf.Load(pixelBaseAddr_SD(pixelIdx) + 16u), pr, pm);
}

float3 load_n1_s(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint instID = load_instID(buf, pixelIdx);
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 20u));
    return ObjectToWorldNrm(instID, raw);   // sentinel handled inside
}

float3 load_n1_s_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 20u));
    return ObjectToWorldNrm(instID, raw);   // sentinel handled inside
}

//reconstruct world x1 from the stored object-space point via the CURRENT
//instance transform (see store_x1). The _with_instID variant skips the extra
//instID load when the caller already holds it, mirroring load_n1_s /
//load_n1_s_with_instID.
float3 load_x1_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 x1_obj = asfloat(buf.Load3(pixelBaseAddr_SD(pixelIdx) + 24u));
    return ObjectToWorldPos(instID, x1_obj);   // sentinel handled inside
}

float3 load_x1(RWByteAddressBuffer buf, uint pixelIdx)
{
    return load_x1_with_instID(buf, pixelIdx, load_instID(buf, pixelIdx));
}

//====================================
//SAMPLE DATA COPY FOR TEMPORAL REUSE
//====================================
//No longer called per frame: the renderer ping-pongs the current/last
//sample-buffer bindings instead (Renderer::SwapSampleBuffers), which
//replaced the 72 B/px copy the merge pass used to do here.
void copySampleData(RWByteAddressBuffer dst, RWByteAddressBuffer src, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    dst.Store4(base,       src.Load4(base));
    dst.Store4(base + 16u, src.Load4(base + 16u));
    dst.Store (base + 32u, src.Load (base + 32u));
}
