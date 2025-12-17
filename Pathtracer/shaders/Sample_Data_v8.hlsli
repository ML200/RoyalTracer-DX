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
    float3 localKd;
    float localPr;
    float localPm;
};

// CHANGED: Increased size from 32 to 48 bytes to accommodate new local material properties.
static const uint BYTES_SD = 48u;

static const uint O_PACK1_SD = 0u;     // float4: (x1.xyz, n1_s)
static const uint O_PACK2_SD = 16u;    // float4: (L1, o, n1_g, IDs)
// ADDED: New pack for local material properties.
static const uint O_PACK3_SD = 32u;    // float4: (localKd, localPr+localPm, unused, unused)


// --- Packing helpers for float <-> half ---
uint f32tof16(float val)
{
    uint f32 = asuint(val);
    uint sign = (f32 >> 31) & 0x1;
    uint exp = (f32 >> 23) & 0xff;
    uint mant = f32 & 0x7fffff;

    if (exp == 0xff) // Inf / NaN
        return (sign << 15) | 0x7c00 | (mant != 0 ? 0x200 : 0);
    if (exp == 0) // Denorm
        return (sign << 15);

    int new_exp = exp - 127;
    if (new_exp < -14) // Underflow to zero
        return (sign << 15);
    if (new_exp > 15) // Overflow to infinity
        return (sign << 15) | 0x7c00;

    return (sign << 15) | ((new_exp + 15) << 10) | (mant >> 13);
}

float f16tof32(uint val)
{
    uint sign = (val >> 15) & 0x1;
    uint exp = (val >> 10) & 0x1f;
    uint mant = val & 0x3ff;

    if (exp == 0x1f) // Inf / NaN
        return asfloat((sign << 31) | 0x7f800000 | (mant != 0 ? 0x400000 : 0));
    if (exp == 0) // Denorm or zero
    {
        if (mant == 0) return asfloat(sign << 31);
        while ((mant & 0x400) == 0) {
            mant <<= 1;
            exp--;
        }
        mant &= 0x3ff;
        return asfloat((sign << 31) | ((exp + 127) << 23) | (mant << 13));
    }

    return asfloat((sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13));
}

uint PackFloat2x16(float u, float v)
{
    return f32tof16(u) | (f32tof16(v) << 16);
}

void UnpackFloat2x16(uint p, out float u, out float v)
{
    u = f16tof32(p & 0xFFFFu);
    v = f16tof32(p >> 16);
}

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

    // ADDED Pack 3: localKd (packed) + localPr/Pm (packed)
    buf.Store2(base + O_PACK3_SD,
               uint2(PackRGB9E5(s.localKd),
                     PackFloat2x16(s.localPr, s.localPm)));
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

    // ADDED Load Pack 3
    uint2 p3 = buf.Load2(base + O_PACK3_SD);
    s.localKd = UnpackRGB9E5(p3.x);
    UnpackFloat2x16(p3.y, s.localPr, s.localPm);

    return s;
}

// --- simple single loaders ---
float3 load_x1   (RWByteAddressBuffer b, uint id){return asfloat(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).xyz);}
float3 load_n1_s (RWByteAddressBuffer b, uint id){return UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).w);}
float3 load_L1   (RWByteAddressBuffer b, uint id){return UnpackRGB9E5(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).x);}
float3 load_o    (RWByteAddressBuffer b, uint id){return UnpackNormal (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).y);}
float3 load_n1_g (RWByteAddressBuffer b, uint id){return UnpackNormal (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).z);}
uint   load_objID(RWByteAddressBuffer b, uint id){return (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).w) & 0xFFFFu;}
uint   load_matID(RWByteAddressBuffer b, uint id){return (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).w) >> 16;}
// ADDED loaders for new fields
float3 load_localKd(RWByteAddressBuffer b, uint id){return UnpackRGB9E5(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD));}
float  load_localPr(RWByteAddressBuffer b, uint id){return f16tof32(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+4u) & 0xFFFFu);}
float  load_localPm(RWByteAddressBuffer b, uint id){return f16tof32(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+4u) >> 16);}


// --- fast loaders ---
// Loads the first 16-byte pack
inline void load_x1_n1_s_fast(RWByteAddressBuffer buf,
                             uint                pixelIdx,
                             out float3          x1,
                             out float3          n1_s)
{
    uint4 p1 = buf.Load4(pixelBaseAddr_SD(pixelIdx) + O_PACK1_SD);   // 16-byte read
    x1 = asfloat(p1.xyz);
    n1_s = UnpackNormal(p1.w);
}

// Loads the second 16-byte pack
inline void load_L1_o_n1g_IDs_fast(RWByteAddressBuffer buf,
                                    uint               pixelIdx,
                                    out float3         L1,
                                    out float3         o,
                                    out float3         n1_g,
                                    out uint           objID,
                                    out uint           matID)
{
    uint4 p2 = buf.Load4(pixelBaseAddr_SD(pixelIdx) + O_PACK2_SD);
    L1   = UnpackRGB9E5(p2.x);
    o    = UnpackNormal(p2.y);
    n1_g = UnpackNormal(p2.z);
    UnpackID16(p2.w, objID, matID);
}

// ADDED: Loads the third 16-byte pack
inline void load_localKd_Pr_Pm_fast(RWByteAddressBuffer buf,
                                    uint                pixelIdx,
                                    out float3          localKd,
                                    out float           localPr,
                                    out float           localPm)
{
    uint2 p3 = buf.Load2(pixelBaseAddr_SD(pixelIdx) + O_PACK3_SD);
    localKd = UnpackRGB9E5(p3.x);
    UnpackFloat2x16(p3.y, localPr, localPm);
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
    return normalize( mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)).xyz);
}

float3 WorldToObjectNrm(uint id, float3 Nw)
{
    float3x3 MT = transpose( (float3x3)instanceProps[id].objectToWorld );
    return normalize( mul( MT, Nw ) );
}