// =================================================================================
// Path_State_v8.hlsli — Volume data structures and packing helpers
// =================================================================================

// Volume IOR (Used for Refraction/Fresnel)
struct VolumeIOR {
    float ior_stack[4]; // Range [1.0, 3.0]
    int   pointer;      // Range [-1, 3]
};

// Volume Aux (Used for Medium Logic)
struct VolumeAux {
    uint matID_stack[4];
    uint objID_stack[4];
};

// =================================================================================
// Packing Helpers
// =================================================================================

// Packs 2 floats (Range 0.0-4.0) and hides 2 bits of the pointer
static uint PackIORPair(float fA, float fB, uint pBits) {
    uint iA = (uint)(saturate(fA * 0.25f) * 32767.0f) & 0x7FFF;
    uint iB = (uint)(saturate(fB * 0.25f) * 32767.0f) & 0x7FFF;

    if ((pBits & 1) != 0) iA |= 0x8000;
    if ((pBits & 2) != 0) iB |= 0x8000;

    return iA | (iB << 16);
}

static float2 UnpackIORPair(uint raw, out uint bits) {
    uint bitA = (raw >> 15) & 1;
    uint bitB = (raw >> 31) & 1;
    bits = bitA | (bitB << 1);

    float fA = (float)(raw & 0x7FFF) / 32767.0f;
    float fB = (float)((raw >> 16) & 0x7FFF) / 32767.0f;

    return float2(fA * 4.0f, fB * 4.0f);
}

// --- IOR Stack (15-bit) + Pointer (Hidden in MSBs) ---
uint2 PackIORStackAndPtr(float stack[4], int ptr) {
    uint p = (uint)(ptr + 1) & 0xF;
    uint2 packed;
    packed.x = PackIORPair(stack[0], stack[1], p & 0x3);
    packed.y = PackIORPair(stack[2], stack[3], (p >> 2) & 0x3);
    return packed;
}

void UnpackIORStackAndPtr(uint2 packed, out float stack[4], out int ptr) {
    uint bits01, bits23;
    float2 v01 = UnpackIORPair(packed.x, bits01);
    stack[0] = v01.x; stack[1] = v01.y;
    float2 v23 = UnpackIORPair(packed.y, bits23);
    stack[2] = v23.x; stack[3] = v23.y;
    uint p = bits01 | (bits23 << 2);
    ptr = (int)p - 1;
}

// --- Material IDs (16-bit) ---
uint2 PackMatStack16(uint stack[4]) {
    return uint2((stack[0] & 0xFFFF) | (stack[1] << 16),
                 (stack[2] & 0xFFFF) | (stack[3] << 16));
}

void UnpackMatStack16(uint2 packed, out uint stack[4]) {
    stack[0] = packed.x & 0xFFFF; stack[1] = packed.x >> 16;
    stack[2] = packed.y & 0xFFFF; stack[3] = packed.y >> 16;
}

// --- Priorities (8-bit) ---
uint PackPrioStack8(uint stack[4]) {
    return (stack[0] & 0xFF) | ((stack[1] & 0xFF) << 8) |
           ((stack[2] & 0xFF) << 16) | ((stack[3] & 0xFF) << 24);
}

void UnpackPrioStack8(uint packed, out uint stack[4]) {
    stack[0] = packed & 0xFF; stack[1] = (packed >> 8) & 0xFF;
    stack[2] = (packed >> 16) & 0xFF; stack[3] = (packed >> 24) & 0xFF;
}
