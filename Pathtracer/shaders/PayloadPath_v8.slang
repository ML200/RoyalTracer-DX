// ============================================================================
// PayloadPath_v8.hlsli — DXR ray payload struct (SM 6.9)
// ============================================================================

#ifndef PAYLOAD_PATH_V1_HLSLI
#define PAYLOAD_PATH_V1_HLSLI

// Sentinel for "no medium"
static const uint MEDIUM_INVALID = 0xFFFFFFFFu;

// Payload (SM 6.9): [raypayload]
struct [raypayload] PathRayPayload
{
    float2 dir2             : read(caller, closesthit) : write(caller, closesthit);
    uint   packedNs         : read(caller, closesthit) : write(caller, closesthit);
    uint   meta0            : read(caller, closesthit) : write(caller, closesthit);
    uint   meta1            : read(caller, closesthit) : write(caller, closesthit);
    uint   seed             : read(caller, closesthit) : write(caller, closesthit);
    uint   iorsPacked       : read(caller, closesthit) : write(caller, closesthit);
    uint   packedThroughput : read(caller, closesthit) : write(caller, closesthit);
    float  bsdfPdf          : read(caller, closesthit) : write(caller, closesthit);
};

#endif // PAYLOAD_PATH_V1_HLSLI
