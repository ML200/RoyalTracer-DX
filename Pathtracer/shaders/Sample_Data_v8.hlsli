static const uint BYTES_SD = 36u;

static const uint SD_FLAG_EMITTER  = 1u;
static const uint SD_FLAG_BACKFACE = 2u;

static const uint SD_FLAG_NOBOUNCE = 4u;
static const uint SD_FLAG_CAMERA_WATER = 8u;

uint pixelBaseAddr_SD(uint pixelIdx)
{
    return pixelIdx * BYTES_SD;
}

float3 WorldToObjectPos(uint id, float3 Pw)
{
    if (id == 0xFFFFFFFFu) return Pw;
    return mul(instanceProps[id].objectToWorldInverse, float4(Pw, 1.0));
}
float3 ObjectToWorldPos(uint id, float3 Po)
{
    if (id == 0xFFFFFFFFu) return Po;
    return mul(instanceProps[id].objectToWorld, float4(Po, 1.0));
}
float3 ObjectToWorldNrm(uint id, float3 No)
{
    if (id == 0xFFFFFFFFu) return No;
    return normalize(mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)));
}
float3 WorldToObjectNrm(uint id, float3 Nw)
{
    if (id == 0xFFFFFFFFu) return Nw;
    float3x3 MT = transpose((float3x3)instanceProps[id].objectToWorld);
    return normalize(mul(MT, Nw));
}

void store_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 0u, instID);
}

void store_flags(RWByteAddressBuffer buf, uint pixelIdx, bool isEmitter, bool backface)
{
    uint f = (isEmitter ? SD_FLAG_EMITTER  : 0u)
           | (backface  ? SD_FLAG_BACKFACE : 0u);
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 4u, f);
}

void store_flagsWord(RWByteAddressBuffer buf, uint pixelIdx, uint flags)
{
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 4u, flags);
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
    float3 n1s_obj = WorldToObjectNrm(instID, n1s_world);
    buf.Store(pixelBaseAddr_SD(pixelIdx) + 20u, PackNormal(n1s_obj));
}

void store_x1(RWByteAddressBuffer buf, uint pixelIdx, float3 x1, uint instID)
{
    float3 x1_obj = WorldToObjectPos(instID, x1);
    buf.Store3(pixelBaseAddr_SD(pixelIdx) + 24u, asuint(x1_obj));
}

void store_sky(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddr_SD(pixelIdx);
    buf.Store4(base,       uint4(0xFFFFFFFFu, SD_FLAG_EMITTER, MATID_ENV_MISS, 0u));
    buf.Store4(base + 16u, uint4(0u, 0u, 0u, 0u));
    buf.Store (base + 32u, 0u);
}

uint load_instID(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr_SD(pixelIdx) + 0u);
}

bool load_isEmitter(RWByteAddressBuffer buf, uint pixelIdx)
{
    return (buf.Load(pixelBaseAddr_SD(pixelIdx) + 4u) & SD_FLAG_EMITTER) != 0u;
}

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
    return ObjectToWorldNrm(instID, raw);
}

float3 load_n1_s_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(pixelBaseAddr_SD(pixelIdx) + 20u));
    return ObjectToWorldNrm(instID, raw);
}

float3 load_x1_with_instID(RWByteAddressBuffer buf, uint pixelIdx, uint instID)
{
    float3 x1_obj = asfloat(buf.Load3(pixelBaseAddr_SD(pixelIdx) + 24u));
    return ObjectToWorldPos(instID, x1_obj);
}

float3 load_x1(RWByteAddressBuffer buf, uint pixelIdx)
{
    return load_x1_with_instID(buf, pixelIdx, load_instID(buf, pixelIdx));
}

struct SDRecord {
    uint   instID;
    uint   flags;
    uint   matID;
    float3 Kd;
    float  Pr;
    float  Pm;
    float3 n1_s;
    float3 x1;
};

SDRecord load_SD(RWByteAddressBuffer buf, uint pixelIdx)
{
    const uint  base = pixelBaseAddr_SD(pixelIdx);
    const uint4 a = buf.Load4(base);
    const uint4 b = buf.Load4(base + 16u);
    const uint  c = buf.Load (base + 32u);

    SDRecord r;
    r.instID = a.x;
    r.flags  = a.y;
    r.matID  = a.z;
    r.Kd     = UnpackRGB9E5(a.w);
    UnpackFloat2x16(b.x, r.Pr, r.Pm);
    r.n1_s   = ObjectToWorldNrm(r.instID, UnpackNormal(b.y));
    r.x1     = ObjectToWorldPos(r.instID, asfloat(uint3(b.z, b.w, c)));
    return r;
}

void load_SD_header(RWByteAddressBuffer buf, uint pixelIdx,
                    out uint flags, out uint matID, out float pr, out float pm)
{
    const uint4 a = buf.Load4(pixelBaseAddr_SD(pixelIdx) + 4u);
    flags = a.x;
    matID = a.y;
    UnpackFloat2x16(a.w, pr, pm);
}
