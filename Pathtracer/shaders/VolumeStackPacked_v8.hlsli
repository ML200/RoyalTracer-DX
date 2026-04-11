#ifndef VOLUME_STACK_PACKED_V2_HLSLI
#define VOLUME_STACK_PACKED_V2_HLSLI

// VolumeIOR: uint2 with 2x PackIORPair (each contains 2x 15-bit IOR + hidden pointer bits)
// VolumeAux: uint4 mat stack (4x 32-bit) + uint4 obj stack (4x 32-bit)
struct VolumeIOR_Packed { uint2 raw; };
struct VolumeAux_Packed { uint4 mat32; uint4 obj32; };

// -----------------------------------------------------------------------------
// Helpers: pointer bits are hidden in bit15 and bit31 of each 32-bit word.
// This gives 4 bits; your pointer uses values 0..4 (needs 3 bits), so ok.
// -----------------------------------------------------------------------------
inline int GetVolumePtrFast_packed(VolumeIOR_Packed v)
{
    uint b0 = (v.raw.x >> 15) & 1u;
    uint b1 = (v.raw.x >> 31) & 1u;
    uint b2 = (v.raw.y >> 15) & 1u;
    uint b3 = (v.raw.y >> 31) & 1u;
    uint p  = b0 | (b1 << 1) | (b2 << 2) | (b3 << 3);
    return (int)p - 1; // 0->-1, 1->0, 4->3
}

inline void SetVolumePtrFast_packed(inout VolumeIOR_Packed v, int ptr)
{
    uint p = (uint)(ptr + 1) & 0xFu;

    // Clear pointer bits (bit15 and bit31) in both words
    v.raw.x &= ~(0x8000u | 0x80000000u);
    v.raw.y &= ~(0x8000u | 0x80000000u);

    // Set according to p
    if (p & 1u) v.raw.x |= 0x8000u;
    if (p & 2u) v.raw.x |= 0x80000000u;
    if (p & 4u) v.raw.y |= 0x8000u;
    if (p & 8u) v.raw.y |= 0x80000000u;
}

// -----------------------------------------------------------------------------
// Helpers: read/write IOR slot (0..3) without unpacking arrays.
// Stored IOR is 15-bit unsigned in [0..32767], mapped to [0..4] by *4.
// -----------------------------------------------------------------------------
inline uint GetIOR15RawAtSlot_packed(VolumeIOR_Packed v, int slot)
{
    // slot 0/1 in v.raw.x, slot 2/3 in v.raw.y
    uint w = (slot < 2) ? v.raw.x : v.raw.y;
    uint shift = (uint)(slot & 1) * 16u;
    uint h = (w >> shift) & 0xFFFFu;
    return h & 0x7FFFu; // strip hidden pointer bit
}

inline float GetIORAtSlot_packed(VolumeIOR_Packed v, int slot)
{
    uint q15 = GetIOR15RawAtSlot_packed(v, slot);
    float f = (float)q15 / 32767.0f;
    return f * 4.0f;
}

inline void SetIORAtSlot_packed(inout VolumeIOR_Packed v, int slot, float ior)
{
    // Quantize [0..4] to 15-bit, preserve pointer bit (we'll recompute pointer later anyway)
    uint q15 = (uint)(saturate(ior * 0.25f) * 32767.0f) & 0x7FFFu;

    if (slot < 2)
    {
        uint shift = (uint)(slot & 1) * 16u;
        uint mask  = 0xFFFFu << shift;

        // Preserve existing pointer bit at that halfword (bit15 of the halfword)
        uint oldHalf = (v.raw.x >> shift) & 0xFFFFu;
        uint ptrBit  = oldHalf & 0x8000u;

        uint newHalf = (q15 | ptrBit) & 0xFFFFu;
        v.raw.x = (v.raw.x & ~mask) | (newHalf << shift);
    }
    else
    {
        uint shift = (uint)(slot & 1) * 16u;
        uint mask  = 0xFFFFu << shift;

        uint oldHalf = (v.raw.y >> shift) & 0xFFFFu;
        uint ptrBit  = oldHalf & 0x8000u;

        uint newHalf = (q15 | ptrBit) & 0xFFFFu;
        v.raw.y = (v.raw.y & ~mask) | (newHalf << shift);
    }
}

inline bool SlotOccupied_packed(VolumeIOR_Packed v, int slot)
{
    return GetIOR15RawAtSlot_packed(v, slot) != 0u;
}

// -----------------------------------------------------------------------------
// Helpers: read/write packed aux stacks
// -----------------------------------------------------------------------------
inline uint GetMatAtSlot_packed(VolumeAux_Packed a, int slot)
{
    return a.mat32[slot];
}

inline void SetMatAtSlot_packed(inout VolumeAux_Packed a, int slot, uint matID)
{
    a.mat32[slot] = matID;
}

inline uint GetObjAtSlot_packed(VolumeAux_Packed a, int slot)
{
    return a.obj32[slot];
}

inline void SetObjAtSlot_packed(inout VolumeAux_Packed a, int slot, uint objID)
{
    a.obj32[slot] = objID;
}

inline bool BoundaryMatch_packed(VolumeAux_Packed a, int slot, uint surfaceMatID, uint surfaceObjID)
{
    return (GetMatAtSlot_packed(a, slot) == surfaceMatID) &&
           (GetObjAtSlot_packed(a, slot) == surfaceObjID);
}

// -----------------------------------------------------------------------------
// Packed IOR stack operations
// -----------------------------------------------------------------------------

inline float2 GetIORs_packed(
    VolumeIOR_Packed vIOR,
    VolumeAux_Packed vAux,
    uint surfaceMatID,
    uint surfaceObjID)
{
    float2 iors;

    // Incident IOR (current / dominant medium)
    int ptr = GetVolumePtrFast_packed(vIOR);
    float incident = (ptr >= 0 && ptr < 4) ? GetIORAtSlot_packed(vIOR, ptr) : 1.0f;
    iors.x = incident;

    // Detect whether we're exiting an existing boundary
    int exiting_slot = -1;
    [unroll]
    for (int v = 0; v < 4; ++v)
    {
        if (SlotOccupied_packed(vIOR, v) && BoundaryMatch_packed(vAux, v, surfaceMatID, surfaceObjID))
        {
            exiting_slot = v;
            break;
        }
    }

    if (exiting_slot < 0)
    {
        // ENTERING
        float ni = materials[surfaceMatID].Ni;
        iors.y = (ni <= incident + EPSILON) ? 0.0f : ni;
    }
    else
    {
        // EXITING: destination is the highest-IOR remaining medium
        float max_remaining_ior = 0.0f;

        [unroll]
        for (int v = 0; v < 4; ++v)
        {
            if (v == exiting_slot) continue;
            float n = GetIORAtSlot_packed(vIOR, v);
            if (n > max_remaining_ior) max_remaining_ior = n;
        }

        iors.y = (max_remaining_ior > 0.0f) ? max_remaining_ior : 1.0f;

        if (abs(iors.y - incident) < EPSILON)
            iors.y = 0.0f;
    }

    return iors;
}

inline void UpdateIORStack_packed(
    inout VolumeIOR_Packed vIOR,
    inout VolumeAux_Packed vAux,
    uint surfaceMatID,
    uint surfaceObjID)
{
    int existing_slot = -1;
    int empty_slot    = -1;

    // Find match (exit) and first empty (enter)
    [unroll]
    for (int v = 0; v < 4; ++v)
    {
        bool occupied = SlotOccupied_packed(vIOR, v);

        if (occupied)
        {
            if (BoundaryMatch_packed(vAux, v, surfaceMatID, surfaceObjID))
                existing_slot = v;
        }
        else
        {
            if (empty_slot < 0)
                empty_slot = v;
        }
    }

    // Toggle
    if (existing_slot >= 0)
    {
        // EXIT
        SetIORAtSlot_packed(vIOR, existing_slot, 0.0f);
        SetMatAtSlot_packed(vAux, existing_slot, 0u);
        SetObjAtSlot_packed(vAux, existing_slot, 0u);
    }
    else if (empty_slot >= 0)
    {
        // ENTER
        float ni = materials[surfaceMatID].Ni;
        SetIORAtSlot_packed(vIOR, empty_slot, ni);
        SetMatAtSlot_packed(vAux, empty_slot, surfaceMatID);
        SetObjAtSlot_packed(vAux, empty_slot, surfaceObjID);
    }

    // Recompute pointer = max IOR
    int   new_ptr = -1;
    float max_ior = 0.0f;

    [unroll]
    for (int v = 0; v < 4; ++v)
    {
        float n = GetIORAtSlot_packed(vIOR, v);
        if (n > max_ior)
        {
            max_ior = n;
            new_ptr = v;
        }
    }

    // Update pointer bits to match new_ptr
    SetVolumePtrFast_packed(vIOR, new_ptr);
}

// Get the Material ID of the volume the camera is currently inside
inline uint GetCurrentMediumMaterialID_packed(VolumeIOR_Packed vIOR, VolumeAux_Packed vAux)
{
    int ptr = GetVolumePtrFast_packed(vIOR);
    // Invalid sentinel (no medium)
    return (ptr >= 0 && ptr < 4) ? GetMatAtSlot_packed(vAux, ptr) : 0xFFFFFFFFu;
}

#endif // VOLUME_STACK_PACKED_V2_HLSLI
