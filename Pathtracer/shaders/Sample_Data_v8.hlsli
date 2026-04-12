/*
    SoA G-buffer: 32 bytes per pixel, 8 planes of 4 bytes each.
    Written once by raygen at depth 0; read-only in all merge passes.

    SoA layout — each field is a contiguous array across all pixels:
        Plane 0: instID          (uint32)
        Plane 1: matID | flags   (uint32) bits[2:31]=matID, bit1=isSky, bit0=isEmitter
        Plane 2: normal_packed   (uint32) object-space octahedral
        Plane 3: localPos.x      (float32) object-space
        Plane 4: localPos.y      (float32) object-space
        Plane 5: localPos.z      (float32) object-space
        Plane 6: Kd_packed       (uint32) PackRGB9E5(albedo)
        Plane 7: PrPm_packed     (uint32) half2(roughness, metalness)

    IOR derivation (camera always in air at x1):
        etai = 1.0
        etat = materials[matID].Ni

    Access:
        addr(plane, pixelIdx) = plane * (W*H*4) + pixelIdx * 4

    Binding:
        g_sample_current (RWByteAddressBuffer u6) — raygen writes
        g_sample_last    (RWByteAddressBuffer u7) — shading writes swap
        g_gbuf_current   (ByteAddressBuffer   t7) — merge/shading reads
        g_gbuf_last      (ByteAddressBuffer   t8) — merge reads (previous frame)
*/

#define GB_NUM_PLANES 8u
#define GB_PLANE_BYTES (IMG_W * IMG_H * 4u)
#define GB_TOTAL_BYTES (GB_NUM_PLANES * GB_PLANE_BYTES)

#define GB_PLANE_INSTID    0u
#define GB_PLANE_MATFLAGS  1u
#define GB_PLANE_NORMAL    2u
#define GB_PLANE_POSX      3u
#define GB_PLANE_POSY      4u
#define GB_PLANE_POSZ      5u
#define GB_PLANE_KD        6u
#define GB_PLANE_PRPM      7u

// Flag bits inside the matFlags plane
#define GB_FLAG_EMITTER 1u
#define GB_FLAG_SKY     2u
#define GB_FLAG_MASK    3u   // emitter | sky

uint gb_addr(uint plane, uint px) { return plane * GB_PLANE_BYTES + px * 4u; }

// ═══════════════════════════════════════════════════════════════════
//  World-space helpers (used by loads)
// ═══════════════════════════════════════════════════════════════════

// -- OtW / WtO helpers --
float3 WorldToObjectPos(uint id, float3 Pw)
{
    return mul(instanceProps[id].objectToWorldInverse, float4(Pw, 1.0)).xyz;
}
float3 ObjectToWorldPos(uint id, float3 Po)
{
    return mul(instanceProps[id].objectToWorld, float4(Po, 1.0)).xyz;
}
float3 ObjectToWorldNrm(uint id, float3 No)
{
    return normalize(mul(instanceProps[id].objectToWorldNormal, float4(No, 0.0f)).xyz);
}
float3 WorldToObjectNrm(uint id, float3 Nw)
{
    float3x3 MT = transpose((float3x3)instanceProps[id].objectToWorld);
    return normalize(mul(MT, Nw));
}

// ═══════════════════════════════════════════════════════════════════
//  Stores — RWByteAddressBuffer (raygen only)
// ═══════════════════════════════════════════════════════════════════

void gb_store_instID(RWByteAddressBuffer buf, uint px, uint instID)
{
    buf.Store(gb_addr(GB_PLANE_INSTID, px), instID);
}

void gb_store_matFlags(RWByteAddressBuffer buf, uint px, uint matID, bool isEmitter)
{
    buf.Store(gb_addr(GB_PLANE_MATFLAGS, px),
              (matID << 2) | (isEmitter ? GB_FLAG_EMITTER : 0u));
}

void gb_store_normal(RWByteAddressBuffer buf, uint px, float3 n_world, uint instID)
{
    float3 n_obj = (instID < 0xFFFFFFFEu) ? WorldToObjectNrm(instID, n_world) : n_world;
    buf.Store(gb_addr(GB_PLANE_NORMAL, px), PackNormal(n_obj));
}

void gb_store_localPos(RWByteAddressBuffer buf, uint px, float3 worldPos, uint instID)
{
    float3 lp = (instID < 0xFFFFFFFEu) ? WorldToObjectPos(instID, worldPos) : worldPos;
    buf.Store(gb_addr(GB_PLANE_POSX, px), asuint(lp.x));
    buf.Store(gb_addr(GB_PLANE_POSY, px), asuint(lp.y));
    buf.Store(gb_addr(GB_PLANE_POSZ, px), asuint(lp.z));
}

void gb_store_Kd(RWByteAddressBuffer buf, uint px, float3 Kd)
{
    buf.Store(gb_addr(GB_PLANE_KD, px), PackRGB9E5(Kd));
}

void gb_store_PrPm(RWByteAddressBuffer buf, uint px, float Pr, float Pm)
{
    buf.Store(gb_addr(GB_PLANE_PRPM, px), PackFloat2x16(Pr, Pm));
}

// Bulk store for sky/miss (sentinel values)
void gb_store_sky(RWByteAddressBuffer buf, uint px)
{
    buf.Store(gb_addr(GB_PLANE_INSTID, px),   0xFFFFFFFFu);
    buf.Store(gb_addr(GB_PLANE_MATFLAGS, px), GB_FLAG_EMITTER | GB_FLAG_SKY);
    buf.Store(gb_addr(GB_PLANE_NORMAL, px),   0u);
    buf.Store(gb_addr(GB_PLANE_POSX, px),     0u);
    buf.Store(gb_addr(GB_PLANE_POSY, px),     0u);
    buf.Store(gb_addr(GB_PLANE_POSZ, px),     0u);
    buf.Store(gb_addr(GB_PLANE_KD, px),       0u);
    buf.Store(gb_addr(GB_PLANE_PRPM, px),     0u);
}

// ═══════════════════════════════════════════════════════════════════
//  Loads — ByteAddressBuffer (SRV, merge/shading passes)
// ═══════════════════════════════════════════════════════════════════

uint gb_load_instID(ByteAddressBuffer buf, uint px)
{
    return buf.Load(gb_addr(GB_PLANE_INSTID, px));
}

bool gb_load_isEmitter(ByteAddressBuffer buf, uint px)
{
    return (buf.Load(gb_addr(GB_PLANE_MATFLAGS, px)) & GB_FLAG_EMITTER) != 0u;
}

bool gb_load_isSky(ByteAddressBuffer buf, uint px)
{
    return (buf.Load(gb_addr(GB_PLANE_MATFLAGS, px)) & GB_FLAG_SKY) != 0u;
}

uint gb_load_matID(ByteAddressBuffer buf, uint px)
{
    return buf.Load(gb_addr(GB_PLANE_MATFLAGS, px)) >> 2;
}

float3 gb_load_normal_world(ByteAddressBuffer buf, uint px, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(gb_addr(GB_PLANE_NORMAL, px)));
    return (instID < 0xFFFFFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

float3 gb_load_localPos(ByteAddressBuffer buf, uint px)
{
    return float3(
        asfloat(buf.Load(gb_addr(GB_PLANE_POSX, px))),
        asfloat(buf.Load(gb_addr(GB_PLANE_POSY, px))),
        asfloat(buf.Load(gb_addr(GB_PLANE_POSZ, px))));
}

// World-space position: 1 matrix multiply from cached instance transform
float3 gb_load_worldPos(ByteAddressBuffer buf, uint px, uint instID)
{
    float3 lp = gb_load_localPos(buf, px);
    return (instID < 0xFFFFFFFEu) ? ObjectToWorldPos(instID, lp) : lp;
}

float3 gb_load_Kd(ByteAddressBuffer buf, uint px)
{
    return UnpackRGB9E5(buf.Load(gb_addr(GB_PLANE_KD, px)));
}

float gb_load_Pr(ByteAddressBuffer buf, uint px)
{
    return f16tof32_custom(buf.Load(gb_addr(GB_PLANE_PRPM, px)) & 0xFFFFu);
}

float gb_load_Pm(ByteAddressBuffer buf, uint px)
{
    return f16tof32_custom(buf.Load(gb_addr(GB_PLANE_PRPM, px)) >> 16);
}

// ═══════════════════════════════════════════════════════════════════
//  Loads — RWByteAddressBuffer overloads (raygen pre-SRV, shading)
// ═══════════════════════════════════════════════════════════════════

uint gb_load_instID(RWByteAddressBuffer buf, uint px)
{
    return buf.Load(gb_addr(GB_PLANE_INSTID, px));
}

bool gb_load_isEmitter(RWByteAddressBuffer buf, uint px)
{
    return (buf.Load(gb_addr(GB_PLANE_MATFLAGS, px)) & GB_FLAG_EMITTER) != 0u;
}

bool gb_load_isSky(RWByteAddressBuffer buf, uint px)
{
    return (buf.Load(gb_addr(GB_PLANE_MATFLAGS, px)) & GB_FLAG_SKY) != 0u;
}

uint gb_load_matID(RWByteAddressBuffer buf, uint px)
{
    return buf.Load(gb_addr(GB_PLANE_MATFLAGS, px)) >> 2;
}

float3 gb_load_normal_world(RWByteAddressBuffer buf, uint px, uint instID)
{
    float3 raw = UnpackNormal(buf.Load(gb_addr(GB_PLANE_NORMAL, px)));
    return (instID < 0xFFFFFFFEu) ? ObjectToWorldNrm(instID, raw) : raw;
}

float3 gb_load_localPos(RWByteAddressBuffer buf, uint px)
{
    return float3(
        asfloat(buf.Load(gb_addr(GB_PLANE_POSX, px))),
        asfloat(buf.Load(gb_addr(GB_PLANE_POSY, px))),
        asfloat(buf.Load(gb_addr(GB_PLANE_POSZ, px))));
}

float3 gb_load_worldPos(RWByteAddressBuffer buf, uint px, uint instID)
{
    float3 lp = gb_load_localPos(buf, px);
    return (instID < 0xFFFFFFFEu) ? ObjectToWorldPos(instID, lp) : lp;
}

float3 gb_load_Kd(RWByteAddressBuffer buf, uint px)
{
    return UnpackRGB9E5(buf.Load(gb_addr(GB_PLANE_KD, px)));
}

float gb_load_Pr(RWByteAddressBuffer buf, uint px)
{
    return f16tof32_custom(buf.Load(gb_addr(GB_PLANE_PRPM, px)) & 0xFFFFu);
}

float gb_load_Pm(RWByteAddressBuffer buf, uint px)
{
    return f16tof32_custom(buf.Load(gb_addr(GB_PLANE_PRPM, px)) >> 16);
}

// ═══════════════════════════════════════════════════════════════════
//  Copy — shading pass swap (SRV source → UAV destination)
// ═══════════════════════════════════════════════════════════════════

void gb_copy(RWByteAddressBuffer dst, ByteAddressBuffer src, uint px)
{
    [unroll]
    for (uint p = 0; p < GB_NUM_PLANES; p++)
    {
        uint a = gb_addr(p, px);
        dst.Store(a, src.Load(a));
    }
}
