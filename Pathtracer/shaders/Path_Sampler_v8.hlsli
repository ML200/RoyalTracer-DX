// Volume stack and IOR management
// Volume boundary matching

inline bool BoundaryMatch(VolumeAux a, int slot, uint matID, uint objID)
{
    return a.matID_stack[slot] == matID && a.objID_stack[slot] == objID;
}

// IOR stack operations

inline float2 GetIORs(
    in VolumeIOR vIOR,
    in VolumeAux vAux,
    uint surfaceMatID,
    uint surfaceObjID)
{
    float2 iors;

    // Incident IOR (current / dominant medium)
    int ptr = vIOR.pointer;
    float incident = (ptr >= 0 && ptr < 4) ? vIOR.ior_stack[ptr] : 1.0f;
    iors.x = incident;

    // Detect whether we're exiting an existing boundary
    int exiting_slot = -1;
    [unroll]
    for (int v = 0; v < 4; ++v)
    {
        if (vIOR.ior_stack[v] > 0.0f && BoundaryMatch(vAux, v, surfaceMatID, surfaceObjID))
        {
            exiting_slot = v;
            break;
        }
    }

    if (exiting_slot < 0)
    {
        // ENTERING: candidate destination is the surface material
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
            float n = vIOR.ior_stack[v];
            if (v != exiting_slot && n > max_remaining_ior)
                max_remaining_ior = n;
        }

        iors.y = (max_remaining_ior > 0.0f) ? max_remaining_ior : 1.0f;

        // Null interface if exiting lower-priority volume
        if (abs(iors.y - incident) < EPSILON)
            iors.y = 0.0f;
    }

    return iors;
}


inline void UpdateIORStack(
    inout VolumeIOR vIOR,
    inout VolumeAux vAux,
    uint surfaceMatID,
    uint surfaceObjID)
{
    int existing_slot = -1;
    int empty_slot    = -1;

    // Find match (exit) and first empty (enter)
    for (int v = 0; v < 4; ++v)
    {
        bool occupied = (vIOR.ior_stack[v] > 0.0f);

        if (occupied)
        {
            if (BoundaryMatch(vAux, v, surfaceMatID, surfaceObjID))
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
        vIOR.ior_stack[existing_slot] = 0.0f;
        vAux.matID_stack[existing_slot] = 0;
        vAux.objID_stack[existing_slot] = 0;
    }
    else if (empty_slot >= 0)
    {
        // ENTER
        float ni = materials[surfaceMatID].Ni;
        vIOR.ior_stack[empty_slot] = ni;
        vAux.matID_stack[empty_slot] = surfaceMatID;
        vAux.objID_stack[empty_slot] = surfaceObjID;
    }

    // Recompute pointer = max IOR
    int   new_ptr = -1;
    float max_ior = 0.0f;

    for (int v = 0; v < 4; ++v)
    {
        float n = vIOR.ior_stack[v];
        if (n > max_ior)
        {
            max_ior = n;
            new_ptr = v;
        }
    }

    vIOR.pointer = new_ptr;
}

// Get the Material ID of the volume the camera is currently inside
inline uint GetCurrentMediumMaterialID(in VolumeIOR vIOR, in VolumeAux vAux)
{
    return (vIOR.pointer >= 0 && vIOR.pointer < 4) ? vAux.matID_stack[vIOR.pointer] : 0x0000FFFF;
}

// Absorption

inline float3 CalculateAbsorptionThroughput(
    float3 tintColor,
    float distanceTraveled)
{
    // Beer-Lambert Law
    float3 throughput = float3(
        exp(-tintColor.x * distanceTraveled),
        exp(-tintColor.y * distanceTraveled),
        exp(-tintColor.z * distanceTraveled)
    );
    return throughput;
}

// Initialization

VolumeIOR InitVolumeIOR()
{
    VolumeIOR vol;
    vol.pointer = -1; // Represents Vacuum/Air

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        vol.ior_stack[i] = 0.0f; // 0.0f means empty slot
    }
    return vol;
}

VolumeAux InitVolumeAux()
{
    VolumeAux aux;
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        aux.matID_stack[i]    = 0;
        aux.objID_stack[i] = 0;
    }
    return aux;
}