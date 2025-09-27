/*
The sample data is managed completely by the GPU in a single large buffer. The entries are structured like this (v=variable):v1_1,v1_2,v1_3...v1_n,v2_1,v2_2...v2_n,...
This extension provides the functions to efficiently load and save data from and to the buffer.
*/
// Struct version for in-pass caching
struct SampleData{
    float3 x1;
    float3 n1;
    float3 L1;
    float3 o;
    uint objID;
    uint matID;
};

static const uint BYTES_SD = 28u;

static const uint O_PACK1_SD = 0u;     // float4
static const uint O_PACK2_SD = 16u;    // float2
static const uint O_PACK3_SD = 24u;    // uint

// helpers
uint pixelBaseAddr_SD(uint pixelIdx)
{
    return pixelIdx * BYTES_SD;
}
uint  PackID16(uint objID, uint matID) { return (objID & 0xFFFFu) | (matID << 16); }
void  UnpackID16(uint v, out uint objID, out uint matID)
{ objID = v & 0xFFFFu;  matID = v >> 16; }

void storeSampleData(RWByteAddressBuffer buf,
                     uint               pixelIdx,
                     const SampleData   s)
{
    const uint base = pixelBaseAddr_SD(pixelIdx);
    buf.Store4(base + O_PACK1_SD,
               uint4(asuint(s.x1), PackNormal(s.n1)));
    buf.Store2(base + O_PACK2_SD,
               uint2(PackRGB9E5(s.L1), PackNormal(s.o)));
    buf.Store (base + O_PACK3_SD, PackID16(s.objID, s.matID));
}

SampleData loadSampleData(RWByteAddressBuffer buf, uint pixelIdx)
{
    SampleData s;
    const uint base = pixelBaseAddr_SD(pixelIdx);
    uint4 p1 = buf.Load4(base + O_PACK1_SD);
    s.x1 = asfloat(p1.xyz);
    s.n1 = UnpackNormal(p1.w);
    uint2 p2 = buf.Load2(base + O_PACK2_SD);
    s.L1 = UnpackRGB9E5(p2.x);
    s.o  = UnpackNormal (p2.y);
    UnpackID16(buf.Load(base + O_PACK3_SD), s.objID, s.matID);
    return s;
}

// simple single laoders
float3 load_x1  (RWByteAddressBuffer b, uint id){return asfloat(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).xyz);}
float3 load_n1  (RWByteAddressBuffer b, uint id){return UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).w);}
float3 load_L1  (RWByteAddressBuffer b, uint id){return UnpackRGB9E5(b.Load2(pixelBaseAddr_SD(id)+O_PACK2_SD).x);}
float3 load_o   (RWByteAddressBuffer b, uint id){return UnpackNormal (b.Load2(pixelBaseAddr_SD(id)+O_PACK2_SD).y);}
uint   load_objID (RWByteAddressBuffer b, uint id){return b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD) & 0xFFFFu;}
uint   load_matID (RWByteAddressBuffer b, uint id){return b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD) >> 16;}

// fast loaders
inline void load_x1_n1_fast(RWByteAddressBuffer buf,
                             uint                pixelIdx,
                             out float3          x1,
                             out float3          n1)
{
    uint4 p1 = buf.Load4(pixelBaseAddr_SD(pixelIdx) + O_PACK1_SD);   // 16-byte read
    x1 = asfloat(p1.xyz);
    n1 = UnpackNormal(p1.w);
}
inline void load_L1_o_fast(RWByteAddressBuffer buf,
                            uint               pixelIdx,
                            out float3         L1,
                            out float3         o)
{
    uint2 p2 = buf.Load2(pixelBaseAddr_SD(pixelIdx) + O_PACK2_SD);
    L1 = UnpackRGB9E5(p2.x);
    o  = UnpackNormal (p2.y);
}
inline void load_IDs_fast(RWByteAddressBuffer buf,
                           uint               pixelIdx,
                           out uint           objID,
                           out uint           matID)
{
    UnpackID16(buf.Load(pixelBaseAddr_SD(pixelIdx) + O_PACK3_SD), objID, matID);
}




// OtW / WtO helpers

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
    return normalize( mul(instanceProps[id].objectToWorldNormal, No) );
}

float3 WorldToObjectNrm(uint id, float3 Nw)
{
    float3x3 MT = transpose( (float3x3)instanceProps[id].objectToWorld );
    return normalize( mul( MT, Nw ) );
}

