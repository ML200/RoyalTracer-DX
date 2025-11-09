// RIS reservoir for direct lighting
struct Reservoir_DI
{
    float3 x2_di;
    float3 n2_di;
    float W_di;
    float w_sum_di;
    float3 L2_di;
    uint M_di;
    uint objID_di;
};

static const uint BYTES_DI   = 28u;

static const uint O_PACK1 = 0u;   // float4
static const uint O_PACK2 = 16u;  // float2
static const uint O_PACK3 = 24u;  // uint (via float)

// helpers
uint pixelBaseAddr(uint pixelIdx) { return pixelIdx * BYTES_DI; }

uint  PackObjID_M(uint objID, uint M)      { return (objID & 0xFFFFu) | (M << 16); }
void  UnpackObjID_M(uint v, out uint o, out uint m) { o = v & 0xFFFFu;  m = v >> 16; }

void storeReservoirDI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_DI r)
{
    uint base = pixelBaseAddr(pixelIdx);
    float3 xO = WorldToObjectPos (r.objID_di, r.x2_di);
    float3 nO = WorldToObjectNrm(r.objID_di, r.n2_di);
    buf.Store4(base + O_PACK1, uint4(asuint(xO), PackNormal(nO)));
    buf.Store2(base + O_PACK2, uint2(PackRGB9E5(r.L2_di), asuint(r.W_di)));
    buf.Store (base + O_PACK3, PackObjID_M(r.objID_di, r.M_di));
}

Reservoir_DI loadReservoirDI(RWByteAddressBuffer buf, uint pixelIdx)
{
    Reservoir_DI r;
    uint base = pixelBaseAddr(pixelIdx);

    uint4 p1 = buf.Load4(base + O_PACK1);
    float3 xO   = asfloat(p1.xyz);
    uint   nEnc = p1.w;

    uint2 p2   = buf.Load2(base + O_PACK2);
    uint  Lenc = p2.x;
    float W    = asfloat(p2.y);

    uint  p3;
    p3 = buf.Load(base + O_PACK3);
    UnpackObjID_M(p3, r.objID_di, r.M_di);

    r.x2_di = ObjectToWorldPos (r.objID_di, xO);
    r.n2_di = ObjectToWorldNrm(r.objID_di, UnpackNormal(nEnc));
    r.L2_di = UnpackRGB9E5(Lenc);
    r.W_di  = W;
    r.w_sum_di = 0.0f;

    return r;
}

// Loader functions to load single
float3 load_x2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint4 p1 = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1);
    return ObjectToWorldPos(objID, asfloat(p1.xyz));
}

float3 load_n2_di(RWByteAddressBuffer buf, uint pixelIdx, uint objID)
{
    uint  nEnc = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1).w;
    return ObjectToWorldNrm(objID, UnpackNormal(nEnc));
}

float3 load_L2_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint Lenc = buf.Load2(pixelBaseAddr(pixelIdx) + O_PACK2).x;
    return UnpackRGB9E5(Lenc);
}

float  load_W_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return asfloat(buf.Load2(pixelBaseAddr(pixelIdx) + O_PACK2).y);
}

uint   load_M_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr(pixelIdx) + O_PACK3) >> 16;
}

uint   load_objID_di(RWByteAddressBuffer buf, uint pixelIdx)
{
    return buf.Load(pixelBaseAddr(pixelIdx) + O_PACK3) & 0xFFFFu;
}

// Fast loaders (no speed impact)
inline void load_x2_n2_fast_di(RWByteAddressBuffer buf,
                             uint                pixelIdx,
                             uint                objID,
                             out float3          x2_world,
                             out float3          n2_world)
{
    uint4 p1 = buf.Load4(pixelBaseAddr(pixelIdx) + O_PACK1);
    float3 xO  = asfloat(p1.xyz);
    uint   nEnc= p1.w;
    x2_world   = ObjectToWorldPos (objID, xO);
    n2_world   = ObjectToWorldNrm(objID, UnpackNormal(nEnc));
}

inline void load_L2_W_fast_di(RWByteAddressBuffer buf,
                            uint               pixelIdx,
                            out float3         L2,
                            out float          W)
{
    uint2 p2 = buf.Load2(pixelBaseAddr(pixelIdx) + O_PACK2);
    L2 = UnpackRGB9E5(p2.x);
    W  = asfloat(p2.y);
}

inline void load_IDs_fast_di(RWByteAddressBuffer buf,
                           uint               pixelIdx,
                           out uint           objID,
                           out uint           M)
{
    UnpackObjID_M(buf.Load(pixelBaseAddr(pixelIdx) + O_PACK3), objID, M);
}

