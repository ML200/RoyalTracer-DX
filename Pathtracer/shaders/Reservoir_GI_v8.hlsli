// RIS reservoir for direct lighting
struct Reservoir_GI
{
    float3 x2_gi;
    float3 n2_gi;
    float  W_gi;
    float  w_sum_gi;
    float3 L2_gi;
    float3 V2_gi;

    float F_gi;
    float4 J_gi;
    uint   lobe0_gi;
    uint   lobe1_gi;

    uint   M_gi;
    uint   objID_gi;
    uint   matID_gi;
    uint   rSeed_gi;
    uint   rIndex_gi;
};

// Data management
static const uint BYTES_GI    = 64u;

static const uint O_GI_PACK1  =  0u;   // float4
static const uint O_GI_PACK2  = 16u;   // float4
static const uint O_GI_PACK3  = 32u;   // float4
static const uint O_GI_PACK4  = 48u;   // float4

uint pixelBaseAddrGI(uint pixelIdx) { return pixelIdx * BYTES_GI; }

uint  PackMatID_M(uint matID, uint M) { return (matID & 0xFFFFu) | (M << 16); }
void  UnpackMatID_M(uint v, out uint matID, out uint M)
{ matID = v & 0xFFFFu;  M = v >> 16; }

uint PackLobes16(uint a, uint b) { return ( (b & 0xFFu) << 8 ) | (a & 0xFFu); }
void UnpackLobes16(uint v16, out uint a, out uint b)
{
    a =  v16       & 0xFFu;
    b = (v16 >> 8) & 0xFFu;
}

void storeReservoirGI(RWByteAddressBuffer buf, uint pixelIdx, const Reservoir_GI r)
{
    uint base = pixelBaseAddrGI(pixelIdx);
    float3 xO = WorldToObjectPos (r.objID_gi, r.x2_gi);
    float3 nO = WorldToObjectNrm(r.objID_gi, r.n2_gi);
    buf.Store4(base + O_GI_PACK1, uint4(asuint(xO), PackNormal(nO)));
    buf.Store4(base + O_GI_PACK2, uint4(PackRGB9E5(r.L2_gi),
                                        PackNormal (r.V2_gi),
                                        r.objID_gi,
                                        PackMatID_M(r.matID_gi, r.M_gi)));
    buf.Store4(base + O_GI_PACK3, asuint(r.J_gi));
    uint lobes16 = PackLobes16(r.lobe0_gi, r.lobe1_gi);
    uint rIndex16 = r.rIndex_gi & 0xFFFFu;
    uint rIndexAndLobes = rIndex16 | (lobes16 << 16);
    buf.Store4(base + O_GI_PACK4, uint4(asuint(r.W_gi),
                                        r.rSeed_gi,
                                        rIndexAndLobes,
                                        r.F_gi));
}

Reservoir_GI loadReservoirGI(RWByteAddressBuffer buf, uint pixelIdx)
{
    uint base = pixelBaseAddrGI(pixelIdx);

    uint4 p1 = buf.Load4(base + O_GI_PACK1);
    uint4 p2 = buf.Load4(base + O_GI_PACK2);
    uint4 p3 = buf.Load4(base + O_GI_PACK3);
    uint4 p4 = buf.Load4(base + O_GI_PACK4);

    Reservoir_GI r;
    r.objID_gi = p2.z;
    UnpackMatID_M(p2.w, r.matID_gi, r.M_gi);

    r.x2_gi     = ObjectToWorldPos (r.objID_gi, asfloat(p1.xyz));
    r.n2_gi     = ObjectToWorldNrm(r.objID_gi, UnpackNormal(p1.w));
    r.L2_gi     = UnpackRGB9E5(p2.x);
    r.V2_gi     = UnpackNormal (p2.y);

    r.J_gi      = asfloat(p3);

    r.W_gi      = asfloat(p4.x);
    r.rSeed_gi  = p4.y;

    uint rIndexAndLobes = p4.z;
    uint lobes16 = (rIndexAndLobes >> 16) & 0xFFFFu;
    r.rIndex_gi  =  rIndexAndLobes        & 0x0000FFFFu;
    UnpackLobes16(lobes16, r.lobe0_gi, r.lobe1_gi);

    r.F_gi      = p4.w;

    r.w_sum_gi = 0.0f; // not stored in memory
    return r;
}


// fast loaders
float3 load_x2_gi(RWByteAddressBuffer b, uint id, uint obj)
{ return ObjectToWorldPos(obj, asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1).xyz)); }

float3 load_n2_gi(RWByteAddressBuffer b, uint id, uint obj)
{ return ObjectToWorldNrm(obj, UnpackNormal(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1).w)); }

float3 load_L2_gi(RWByteAddressBuffer b, uint id)
{ return UnpackRGB9E5(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).x); }

float3 load_V2_gi(RWByteAddressBuffer b, uint id)
{ return UnpackNormal(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).y); }

uint   load_objID_gi(RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).z; }

uint   load_matID_gi(RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).w & 0xFFFFu; }

uint   load_M_gi(RWByteAddressBuffer b, uint id)
{ return b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).w >> 16; }

float4 load_J_gi (RWByteAddressBuffer b, uint id)
{ return asfloat(b.Load4(pixelBaseAddrGI(id)+O_GI_PACK3)); }

float load_W_gi(RWByteAddressBuffer b, uint id)
{return asfloat(b.Load4(pixelBaseAddrGI(id) + O_GI_PACK4).x); }

inline float3 load_F_gi(RWByteAddressBuffer b, uint id)
{
    uint4 p4 = b.Load4(pixelBaseAddrGI(id) + O_GI_PACK4);
    return p4.w;
}

inline void load_x2_n2_fast_gi(RWByteAddressBuffer b, uint id, uint obj,
                               out float3 x2, out float3 n2)
{
    uint4 p1 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK1);
    x2 = ObjectToWorldPos (obj, asfloat(p1.xyz));
    n2 = ObjectToWorldNrm(obj, UnpackNormal(p1.w));
}

inline void load_L2_V2_fast_gi(RWByteAddressBuffer b, uint id,
                               out float3 L2, out float3 V2)
{
    uint4 p2 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2);
    L2 = UnpackRGB9E5(p2.x);
    V2 = UnpackNormal (p2.y);
}

inline void load_IDs_fast_gi(RWByteAddressBuffer b, uint id,
                             out uint objID, out uint matID, out uint M)
{
    uint v = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).w;
    objID = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK2).z;
    UnpackMatID_M(v, matID, M);
}

void   load_WSeedIndexLobes_F(RWByteAddressBuffer b, uint id,
                              out float W, out uint rSeed, out uint rIndex,
                              out uint l0, out uint l1, out float3 F)
{
    uint4 p4 = b.Load4(pixelBaseAddrGI(id)+O_GI_PACK4);
    W      = asfloat(p4.x);
    rSeed  = p4.y;
    uint v = p4.z;
    rIndex = v & 0xFFFFu;
    uint lobes16 = (v >> 16) & 0xFFFFu;
    UnpackLobes16(lobes16, l0, l1);
    F = p4.w;
}

