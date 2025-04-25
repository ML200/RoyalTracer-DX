/*
The sample data is managed completely by the GPU in a single large buffer. The entries are structured like this (v=variable):v1_1,v1_2,v1_3...v1_n,v2_1,v2_2...v2_n,...
This extension provides the functions to efficiently load and save data from and to the buffer.
*/

// Size constants
static const uint B_x1   = 12;  // float3
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


// Struct version for in-pass caching
struct SampleData{
    float3 x1;
    float3 n1;
    float3 L1;
    float3 o;
    uint objID;
    uint matID;
};

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
float3 load_objID(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_obj * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_obj;
    return buffer.Load(addr);
}
void store_objID(uint objID, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_obj * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_obj;
    buffer.Store(addr, objID);
}

//__________________________matID_____________________________
float3 load_matID(RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_mID * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_mID;
    return buffer.Load(addr);
}
void store_matID(uint matID, RWByteAddressBuffer buffer, uint pixelIdx){
    uint addr = P_mID * (DispatchRaysDimensions().x * DispatchRaysDimensions().y) + pixelIdx * B_mID;
    buffer.Store(addr, matID);
}