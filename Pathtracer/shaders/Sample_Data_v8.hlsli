/*
The sample data is managed completely by the GPU in a single large buffer. The entries are structured like this (v=variable):v1_1,v1_2,v1_3...v1_n,v2_1,v2_2...v2_n,...
This extension provides the functions to efficiently load and save data from and to the buffer.
*/
// Struct version for in-pass caching
struct SampleData{
    float3 x1;
    float3 n1_s;
    float3 n1_g;
    float3 L1;
    float3 o;
    uint objID;
    uint matID;
    float2 uv;
    float etai;
    float etat;
};

// Pack1(16) + Pack2(16) + Pack3(12) = 44 bytes
static const uint BYTES_SD = 44u;

static const uint O_PACK1_SD = 0u;     // float4: x1 + n1_s
static const uint O_PACK2_SD = 16u;    // float4: L1 + o + n1_g + IDs
static const uint O_PACK3_SD = 32u;    // uint3:  uv.x(4) + uv.y(4) + etai_etat(half2=4)


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

    // Pack 1: x1 (float3) + n1_s (packed)
    buf.Store4(base + O_PACK1_SD,
               uint4(asuint(s.x1), PackNormal(s.n1_s)));

    // Pack 2: L1 (packed) + o (packed) + n1_g (packed) + IDs (packed)
    buf.Store4(base + O_PACK2_SD,
               uint4(PackRGB9E5(s.L1),
                     PackNormal(s.o),
                     PackNormal(s.n1_g),
                     PackID16(s.objID, s.matID)));

    // Pack 3: uv.x (float) + uv.y (float) + etai/etat (half2)
    buf.Store3(base + O_PACK3_SD,
               uint3(asuint(s.uv.x),
                     asuint(s.uv.y),
                     PackFloat2x16(s.etai, s.etat)));
}

SampleData loadSampleData(RWByteAddressBuffer buf, uint pixelIdx)
{
    SampleData s;
    const uint base = pixelBaseAddr_SD(pixelIdx);

    // Load Pack 1
    uint4 p1 = buf.Load4(base + O_PACK1_SD);
    s.x1 = asfloat(p1.xyz);
    s.n1_s = UnpackNormal(p1.w);

    // Load Pack 2
    uint4 p2 = buf.Load4(base + O_PACK2_SD);
    s.L1 = UnpackRGB9E5(p2.x);
    s.o  = UnpackNormal (p2.y);
    s.n1_g = UnpackNormal(p2.z);
    UnpackID16(p2.w, s.objID, s.matID);

    // Load Pack 3
    uint3 p3 = buf.Load3(base + O_PACK3_SD);
    s.uv.x = asfloat(p3.x);
    s.uv.y = asfloat(p3.y);
    UnpackFloat2x16(p3.z, s.etai, s.etat);

    return s;
}

// --- single loaders ---
float3 load_x1   (RWByteAddressBuffer b, uint id){return asfloat(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).xyz);}
float3 load_n1_s (RWByteAddressBuffer b, uint id){return UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).w);}
float3 load_n1_g (RWByteAddressBuffer b, uint id){return UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).z);}
float3 load_L1   (RWByteAddressBuffer b, uint id){return UnpackRGB9E5(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).x);}
float3 load_o    (RWByteAddressBuffer b, uint id){return UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).y);}
uint   load_objID(RWByteAddressBuffer b, uint id){return (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).w) & 0xFFFFu;}
uint   load_matID(RWByteAddressBuffer b, uint id){return (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).w) >> 16;}
float2 load_uv   (RWByteAddressBuffer b, uint id){return float2(asfloat(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD)), asfloat(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+4u)));}
float  load_etai (RWByteAddressBuffer b, uint id){return f16tof32_custom(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+8u) & 0xFFFFu);}
float  load_etat (RWByteAddressBuffer b, uint id){return f16tof32_custom(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+8u) >> 16);}



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
    return normalize( mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)).xyz);
}

float3 WorldToObjectNrm(uint id, float3 Nw)
{
    float3x3 MT = transpose( (float3x3)instanceProps[id].objectToWorld );
    return normalize( mul( MT, Nw ) );
}
