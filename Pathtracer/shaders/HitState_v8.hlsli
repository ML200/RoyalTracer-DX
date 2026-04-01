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
    float3 hitNormal;
    float3 hitGNormal;
    uint   lightID;
    float2 uv;
};

void CompressToPayload(in HitInfo info, inout HitBlobPayload payload)
{
    payload.lightID       = info.lightID;
    payload.packedNormalS = PackNormal(info.hitNormal);
    payload.packedNormalG = PackNormal(info.hitGNormal);
    payload.packedUV      = PackFloat2x16(info.uv.x, info.uv.y);
}

HitInfo DecompressHitInfo(in HitBlobPayload p)
{
    HitInfo info;

    info.lightID    = p.lightID;
    info.hitNormal  = UnpackNormal(p.packedNormalS);
    info.hitGNormal = UnpackNormal(p.packedNormalG);
    UnpackFloat2x16(p.packedUV, info.uv.x, info.uv.y);

    return info;
}

