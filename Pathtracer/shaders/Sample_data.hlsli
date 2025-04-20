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

//__________________________x1_____________________________
float3 load_x1(RWByteAddressBuffer buffer){
    // Get the pixel position:
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    // We get the buffer location using the size of the entry (base is 0:
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_x1(float3 x1, RWByteAddressBuffer buffer){
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}

//__________________________n1_____________________________
float3 load_n1(RWByteAddressBuffer buffer){
    // Get the pixel position:
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    // We get the buffer location using the size of the entry (base is 0:
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_n1(float3 x1, RWByteAddressBuffer buffer){
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}

//__________________________L1_____________________________
float3 load_L1(RWByteAddressBuffer buffer){
    // Get the pixel position:
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    // We get the buffer location using the size of the entry (base is 0:
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_L1(float3 x1, RWByteAddressBuffer buffer){
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}

//__________________________o_____________________________
float3 load_o(RWByteAddressBuffer buffer){
    // Get the pixel position:
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    // We get the buffer location using the size of the entry (base is 0:
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_o(float3 x1, RWByteAddressBuffer buffer){
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}

//__________________________objID_____________________________
float3 load_objID(RWByteAddressBuffer buffer){
    // Get the pixel position:
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    // We get the buffer location using the size of the entry (base is 0:
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_objID(float3 x1, RWByteAddressBuffer buffer){
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}

//__________________________matID_____________________________
float3 load_matID(RWByteAddressBuffer buffer){
    // Get the pixel position:
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    // We get the buffer location using the size of the entry (base is 0:
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    return asfloat(buffer.Load3(addr));
}
void store_matID(float3 x1, RWByteAddressBuffer buffer){
    uint pixelIdx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    uint addr = P_x1 * (float)(DispatchRaysDimensions().x +DispatchRaysDimensions().y) + pixelIdx * B_x1;
    buffer.Store3(addr, asuint(x1));
}