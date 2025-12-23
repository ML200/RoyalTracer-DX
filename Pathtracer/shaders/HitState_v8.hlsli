struct [raypayload] HitBlobPayload
{
    float hitT          : read(caller) : write(closesthit, miss);
    uint  packedNormalS : read(caller) : write(closesthit, miss);
    uint  packedNormalG : read(caller) : write(closesthit, miss);
    uint  objID         : read(caller) : write(closesthit, miss);
    uint  materialID    : read(caller) : write(closesthit, miss);
    uint  lightID       : read(caller) : write(closesthit, miss);
    uint  packedKd      : read(caller) : write(closesthit, miss);
    uint  packedParams  : read(caller) : write(closesthit, miss);
};

// Logical struct
struct HitInfo {
    float3 hitNormal;
    float3 hitGNormal;
    uint   lightID;
    float3 localKd;
    half  localPr;
    half  localPm;
};


void CompressToPayload(in HitInfo info, inout HitBlobPayload payload)
{
    payload.lightID       = info.lightID;
    payload.packedNormalS = PackNormal(info.hitNormal);
    payload.packedNormalG = PackNormal(info.hitGNormal);
    payload.packedKd      = PackRGB9E5(info.localKd);
    payload.packedParams  = PackScalars16(info.localPr, info.localPm);
}

HitInfo DecompressHitInfo(in HitBlobPayload p)
{
    HitInfo info;

    // 1. Raw Copies
    info.lightID    = p.lightID;

    // 2. Vector Unpacking (Done once)
    info.hitNormal  = UnpackNormal(p.packedNormalS);
    info.hitGNormal = UnpackNormal(p.packedNormalG);
    info.localKd    = UnpackRGB9E5(p.packedKd);

    // 3. Scalar Unpacking
    float2 params   = UnpackScalars16(p.packedParams);
    info.localPr    = params.x;
    info.localPm    = params.y;

    return info;
}

