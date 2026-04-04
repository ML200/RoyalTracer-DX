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

// OtW / WtO helpers (must precede store/load which use them)
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

void storeSampleData(RWByteAddressBuffer buf,
                     uint               pixelIdx,
                     const SampleData   s)
{
    const uint base = pixelBaseAddr_SD(pixelIdx);

    // Store x1 and normals in OBJECT SPACE so they track with the surface.
    // On load, they get transformed back to world space using the current transform.
    float3 x1Store = s.x1;
    float3 n1sStore = s.n1_s;
    float3 n1gStore = s.n1_g;
    float3 oStore   = s.o;

    if (s.objID < 0xFFFEu) // valid surface hit (not sky/env)
    {
        x1Store  = WorldToObjectPos(s.objID, s.x1);
        n1sStore = WorldToObjectNrm(s.objID, s.n1_s);
        n1gStore = WorldToObjectNrm(s.objID, s.n1_g);
        oStore   = WorldToObjectNrm(s.objID, s.o);
    }

    // Pack 1: x1 (float3, object space) + n1_s (packed, object space)
    buf.Store4(base + O_PACK1_SD,
               uint4(asuint(x1Store), PackNormal(n1sStore)));

    // Pack 2: L1 (packed) + o (packed, object space) + n1_g (packed, object space) + IDs
    buf.Store4(base + O_PACK2_SD,
               uint4(PackRGB9E5(s.L1),
                     PackNormal(oStore),
                     PackNormal(n1gStore),
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

    // Load Pack 2 first to get objID (needed for object→world transform)
    uint4 p2 = buf.Load4(base + O_PACK2_SD);
    s.L1 = UnpackRGB9E5(p2.x);
    UnpackID16(p2.w, s.objID, s.matID);

    // Load Pack 1
    uint4 p1 = buf.Load4(base + O_PACK1_SD);
    float3 x1Raw   = asfloat(p1.xyz);
    float3 n1sRaw  = UnpackNormal(p1.w);
    float3 oRaw    = UnpackNormal(p2.y);
    float3 n1gRaw  = UnpackNormal(p2.z);

    // Transform from object space back to world space using CURRENT transform.
    // This makes positions/normals track with the surface when objects move.
    if (s.objID < 0xFFFEu)
    {
        s.x1   = ObjectToWorldPos(s.objID, x1Raw);
        s.n1_s = ObjectToWorldNrm(s.objID, n1sRaw);
        s.n1_g = ObjectToWorldNrm(s.objID, n1gRaw);
        s.o    = ObjectToWorldNrm(s.objID, oRaw);
    }
    else
    {
        s.x1   = x1Raw;
        s.n1_s = n1sRaw;
        s.n1_g = n1gRaw;
        s.o    = oRaw;
    }

    // Load Pack 3
    uint3 p3 = buf.Load3(base + O_PACK3_SD);
    s.uv.x = asfloat(p3.x);
    s.uv.y = asfloat(p3.y);
    UnpackFloat2x16(p3.z, s.etai, s.etat);

    return s;
}

// --- single loaders (object space → world space) ---
// objID must be loaded first for geometry fields; L1/matID/uv/ior don't need transform.
uint   load_objID(RWByteAddressBuffer b, uint id){return (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).w) & 0xFFFFu;}
uint   load_matID(RWByteAddressBuffer b, uint id){return (b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).w) >> 16;}
float3 load_L1   (RWByteAddressBuffer b, uint id){return UnpackRGB9E5(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).x);}
float2 load_uv   (RWByteAddressBuffer b, uint id){return float2(asfloat(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD)), asfloat(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+4u)));}
float  load_etai (RWByteAddressBuffer b, uint id){return f16tof32_custom(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+8u) & 0xFFFFu);}
float  load_etat (RWByteAddressBuffer b, uint id){return f16tof32_custom(b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD+8u) >> 16);}

// Geometry loaders: stored in object space, returned in world space
float3 load_x1(RWByteAddressBuffer b, uint id){
    uint oid = load_objID(b, id);
    float3 raw = asfloat(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).xyz);
    return (oid < 0xFFFEu) ? ObjectToWorldPos(oid, raw) : raw;
}
float3 load_n1_s(RWByteAddressBuffer b, uint id){
    uint oid = load_objID(b, id);
    float3 raw = UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).w);
    return (oid < 0xFFFEu) ? ObjectToWorldNrm(oid, raw) : raw;
}
float3 load_n1_g(RWByteAddressBuffer b, uint id){
    uint oid = load_objID(b, id);
    float3 raw = UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).z);
    return (oid < 0xFFFEu) ? ObjectToWorldNrm(oid, raw) : raw;
}
float3 load_o(RWByteAddressBuffer b, uint id){
    uint oid = load_objID(b, id);
    float3 raw = UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK2_SD).y);
    return (oid < 0xFFFEu) ? ObjectToWorldNrm(oid, raw) : raw;
}

