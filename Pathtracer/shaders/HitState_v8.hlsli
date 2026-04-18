struct [raypayload] HitBlobPayload
{
    float hitT          : read(caller) : write(closesthit, miss);
    uint  packedNormalS : read(caller) : write(closesthit, miss);
    uint  packedNormalG : read(caller) : write(closesthit, miss);
    uint  objID         : read(caller) : write(closesthit, miss);
    uint  materialID    : read(caller) : write(closesthit, miss);
    uint  lightID       : read(caller) : write(closesthit, miss);
    uint  packedUV      : read(caller) : write(closesthit, miss);
};

// Logical struct
struct HitInfo {
    float3 hitPos;
    float3 hitNormal;
    bool   backface;
    uint   lightID;
    float2 uv;
};

//UNUSED!!!
