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
// Size constants
/*static const uint B_x1   = 12;  // float3
static const uint B_n1   =  4; // packed float3
static const uint B_L1   = 4;  // packed float3
static const uint B_o    = 4;  // packed float3
static const uint B_obj  =  4;
static const uint B_mID  =  4;

// Offset constants
static const uint P_x1   = 0;
static const uint P_n1   = P_x1    + B_x1;
static const uint P_L1   = P_n1   + B_n1;
static const uint P_o    = P_L1    + B_L1;
static const uint P_obj  = P_o     + B_o;
static const uint P_mID  = P_obj    + B_obj;

//__________________________x1_____________________________
float3 load_x1(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_x1 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_x1(float3 x1, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_x1 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}

//__________________________n1_____________________________
float3 load_n1(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_n1 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_n1;
    return UnpackNormal(buffer.Load(addr));
}
void store_n1(float3 n1, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_n1 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_n1;
    buffer.Store(addr, PackNormal(n1));
}

//__________________________L1_____________________________
float3 load_L1(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_L1 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_L1;
    return UnpackRGB9E5(buffer.Load(addr));
}
void store_L1(float3 L1, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_L1 * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_L1;
    buffer.Store(addr, PackRGB9E5(L1));
}

//__________________________o_____________________________
float3 load_o(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_o * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_o;
    return UnpackNormal(buffer.Load(addr));
}
void store_o(float3 o, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_o * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_o;
    buffer.Store(addr, PackNormal(o));
}

//__________________________objID_____________________________
uint load_objID(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_obj * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_obj;
    return asuint(buffer.Load(addr));
}
void store_objID(uint objID, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_obj * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_obj;
    buffer.Store(addr, objID);
}

//__________________________matID_____________________________
uint load_matID(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_mID * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_mID;
    return asuint(buffer.Load(addr));
}
void store_matID(uint matID, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_mID * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_mID;
    buffer.Store(addr, matID);
}


// Load *all* per-pixel data into a single struct
SampleData loadSampleData(RWByteAddressBuffer buffer, uint pixelIdx)
{
    SampleData s;

    s.x1    = load_x1(buffer, pixelIdx);
    s.n1    = load_n1(buffer, pixelIdx);
    s.L1    = load_L1(buffer, pixelIdx);
    s.o     = load_o(buffer, pixelIdx);
    s.objID = load_objID(buffer, pixelIdx);
    s.matID = load_matID(buffer, pixelIdx);

    return s;
}*/

/*────────────────────────────────────────────────────────────────────────────
   COMPACT 28-BYTE SAMPLE DATA  (per pixel)
   ┌────────┬────────┬────────┐
   │ pack1  │ pack2  │ pack3  │
   │16 byte│ 8 byte │ 4 byte │
   │x1.xyz │L1_enc  │objID16 │
   │n1_enc │o_enc   │matID16 │
   └────────┴────────┴────────┘
────────────────────────────────────────────────────────────────────────────*/
static const uint BYTES_SD = 28u;

static const uint O_PACK1_SD = 0u;     // float4
static const uint O_PACK2_SD = 16u;    // float2
static const uint O_PACK3_SD = 24u;    // uint (via float)

/* address helpers ---------------------------------------------------------*/
uint pixelBaseAddr_SD(uint pixelIdx)
{
    return pixelIdx * BYTES_SD;
}

/* objID ‖ matID helpers ---------------------------------------------------*/
uint  PackID16(uint objID, uint matID) { return (objID & 0xFFFFu) | (matID << 16); }
void  UnpackID16(uint v, out uint objID, out uint matID)
{ objID = v & 0xFFFFu;  matID = v >> 16; }

/*────────────────────────────────────────────────────────────────────────────
   FAST, WHOLE-SAMPLE STORE / LOAD
────────────────────────────────────────────────────────────────────────────*/
void storeSampleData(RWByteAddressBuffer buf,
                     uint               pixelIdx,
                     const SampleData   s)
{
    const uint base = pixelBaseAddr_SD(pixelIdx);

    /* pack1 : x1.xyz | n1_enc */
    buf.Store4(base + O_PACK1_SD,
               uint4(asuint(s.x1), PackNormal(s.n1)));

    /* pack2 : L1_enc | o_enc */
    buf.Store2(base + O_PACK2_SD,
               uint2(PackRGB9E5(s.L1), PackNormal(s.o)));

    /* pack3 : objID16 | matID16 */
    buf.Store (base + O_PACK3_SD, PackID16(s.objID, s.matID));
}

SampleData loadSampleData(RWByteAddressBuffer buf, uint pixelIdx)
{
    SampleData s;
    const uint base = pixelBaseAddr_SD(pixelIdx);

    /* pack1 --------------------------------------------------------------*/
    uint4 p1 = buf.Load4(base + O_PACK1_SD);
    s.x1 = asfloat(p1.xyz);
    s.n1 = UnpackNormal(p1.w);

    /* pack2 --------------------------------------------------------------*/
    uint2 p2 = buf.Load2(base + O_PACK2_SD);
    s.L1 = UnpackRGB9E5(p2.x);
    s.o  = UnpackNormal (p2.y);

    /* pack3 --------------------------------------------------------------*/
    UnpackID16(buf.Load(base + O_PACK3_SD), s.objID, s.matID);

    return s;
}

/*────────────────────────────────────────────────────────────────────────────
   FIELD-LEVEL LOAD HELPers   (one aligned read each)
────────────────────────────────────────────────────────────────────────────*/
float3 load_x1  (RWByteAddressBuffer b, uint id){return asfloat(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).xyz);}
float3 load_n1  (RWByteAddressBuffer b, uint id){return UnpackNormal(b.Load4(pixelBaseAddr_SD(id)+O_PACK1_SD).w);}
float3 load_L1  (RWByteAddressBuffer b, uint id){return UnpackRGB9E5(b.Load2(pixelBaseAddr_SD(id)+O_PACK2_SD).x);}
float3 load_o   (RWByteAddressBuffer b, uint id){return UnpackNormal (b.Load2(pixelBaseAddr_SD(id)+O_PACK2_SD).y);}
uint   load_objID (RWByteAddressBuffer b, uint id){return b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD) & 0xFFFFu;}
uint   load_matID (RWByteAddressBuffer b, uint id){return b.Load(pixelBaseAddr_SD(id)+O_PACK3_SD) >> 16;}

/*─────────────────────────────────────────────────────────────────────────
   FAST GROUP LOADERS  (28-byte compact layout)
─────────────────────────────────────────────────────────────────────────*/

/* 1.  x1 + n1  ──────────────────────────────────────────────────────────*/
/*    Returns world-space x1 and decoded normal n1.                      */
inline void load_x1_n1_fast(RWByteAddressBuffer buf,
                             uint                pixelIdx,
                             out float3          x1,
                             out float3          n1)
{
    uint4 p1 = buf.Load4(pixelBaseAddr_SD(pixelIdx) + O_PACK1_SD);   // 16-byte read
    x1 = asfloat(p1.xyz);
    n1 = UnpackNormal(p1.w);
}

/* 2.  L1 + o  ───────────────────────────────────────────────────────────*/
/*    Returns decoded radiance L1 and outgoing dir o (both world space). */
inline void load_L1_o_fast(RWByteAddressBuffer buf,
                            uint               pixelIdx,
                            out float3         L1,
                            out float3         o)
{
    uint2 p2 = buf.Load2(pixelBaseAddr_SD(pixelIdx) + O_PACK2_SD);   // 8-byte read
    L1 = UnpackRGB9E5(p2.x);
    o  = UnpackNormal (p2.y);
}

/* 3.  objID + matID  ────────────────────────────────────────────────────*/
inline void load_IDs_fast(RWByteAddressBuffer buf,
                           uint               pixelIdx,
                           out uint           objID,
                           out uint           matID)
{
    UnpackID16(buf.Load(pixelBaseAddr_SD(pixelIdx) + O_PACK3_SD), objID, matID);
}




// Helpers ---------------------------------------------------------------------

float3 WorldToObjectPos(uint id, float3 Pw)
{
    // Po = M_inv · Pw               (column vector form)
    return mul(instanceProps[id].objectToWorldInverse, float4(Pw, 1.0)).xyz;
}

float3 ObjectToWorldPos(uint id, float3 Po)
{
    // Pw = M · Po
    return mul(instanceProps[id].objectToWorld, float4(Po, 1.0)).xyz;
}

float3 ObjectToWorldNrm(uint id, float3 No)
{
    // Nw = (M-1)ᵀ · No      ← you already have that in objectToWorldNormal
    return normalize( mul(instanceProps[id].objectToWorldNormal, No) );
}

float3 WorldToObjectNrm(uint id, float3 Nw)
{
    // No =  ( (M-1)ᵀ )-1 · Nw   =  (Mᵀ) · Nw
    // We can build Mᵀ on the fly: transpose of the upper-left 3×3 of M
    float3x3 MT = transpose( (float3x3)instanceProps[id].objectToWorld );
    return normalize( mul( MT, Nw ) );
}

