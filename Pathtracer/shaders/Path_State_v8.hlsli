// Per-pixel path vertex state scratch — written by raygen, overwritten by
// Pass_spat_gi_select_v8 later in the frame.
//
// Fields stored:
//   depth-1 vertex (set once at depth=1): x2_world, n2_world, uv, matID, objID, eta
//   v2 direction  (set once at depth=2): v2_world  (x3 → x2)
//   shadow ray   (set by raygen NEE on acceptance, traced at raygen end):
//                 origin (pre-offset), direction, distance; dist==0 means
//                 the currently-winning RIS sample doesn't need visibility
//                 (emitter hit / env / BSDF-sampled — visibility is implicit).
//
// tpost stays in registers since it's updated every bounce.
//
// SoA layout (56 B/pixel). Backing store is g_pathStateBuffer (u10), which
// is sized to the max of this layout and the spatial-select AoS layout.
//   plane 0 (PACK1,   16 B/pixel): x2.xyz (12 B) + n2_packed (4 B)
//   plane 1 (PACK2,   16 B/pixel): uv_packed + matID + objID + eta_asuint
//   plane 2 (V2,       4 B/pixel): v2_packed
//   plane 3 (SHADOW,  20 B/pixel): origin.xyz (12 B) + dir_packed (4 B) + dist (4 B)
static const uint PS_SZ_PACK1  = 16u;
static const uint PS_SZ_PACK2  = 16u;
static const uint PS_SZ_V2     =  4u;
static const uint PS_SZ_SHADOW = 20u;

static const uint PS_PLANE_PACK1  =  0u;
static const uint PS_PLANE_PACK2  = 16u;
static const uint PS_PLANE_V2     = 32u;
static const uint PS_PLANE_SHADOW = 36u;

// Tile-aligned pixel count (matches MapPixelID's 4×8 tile swizzle).
uint ps_numPx() { return ((IMG_W + 3u) / 4u) * ((IMG_H + 7u) / 8u) * 32u; }

uint ps_addr_pack1 (uint px) { return px * PS_SZ_PACK1; }
uint ps_addr_pack2 (uint px) { uint N = ps_numPx(); return N * PS_PLANE_PACK2  + px * PS_SZ_PACK2; }
uint ps_addr_v2    (uint px) { uint N = ps_numPx(); return N * PS_PLANE_V2     + px * PS_SZ_V2; }
uint ps_addr_shadow(uint px) { uint N = ps_numPx(); return N * PS_PLANE_SHADOW + px * PS_SZ_SHADOW; }


struct PathVertexState {
    float3 x2;
    float3 n2_s;
    float2 uv;
    uint   matID;
    uint   objID;
    float  eta;
    float3 v2;
};


void store_ps_depth1(RWByteAddressBuffer buf, uint pixelIdx,
                     float3 x2_world, float3 n2_world,
                     float2 uv, uint matID, uint objID, float eta)
{
    buf.Store4(ps_addr_pack1(pixelIdx), uint4(asuint(x2_world), PackNormal(n2_world)));
    buf.Store4(ps_addr_pack2(pixelIdx), uint4(PackFloat2x16(uv.x, uv.y),
                                              matID, objID, asuint(eta)));
}

void store_ps_v2(RWByteAddressBuffer buf, uint pixelIdx, float3 v2_world)
{
    buf.Store(ps_addr_v2(pixelIdx), PackNormal(v2_world));
}


PathVertexState load_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    PathVertexState s;

    const uint4 p1 = buf.Load4(ps_addr_pack1(pixelIdx));
    s.x2   = asfloat(p1.xyz);
    s.n2_s = UnpackNormal(p1.w);

    const uint4 p2 = buf.Load4(ps_addr_pack2(pixelIdx));
    UnpackFloat2x16(p2.x, s.uv.x, s.uv.y);
    s.matID = p2.y;
    s.objID = p2.z;
    s.eta   = asfloat(p2.w);

    s.v2 = UnpackNormal(buf.Load(ps_addr_v2(pixelIdx)));
    return s;
}

// Seed the PathVertexState slot with safe sentinel defaults before the path
// loop. Prevents load_ps from ever returning prior-frame spatial-pass scratch
// data — the raygen path-loop pass-through continue (matNi ≤ 1+EPSILON) and
// other early-break cases can skip store_ps_depth1, leaving the slot stale.
// matID = MATID_ENV_MISS is the standard sentinel used by shift/temporal to
// bypass material / BSDF access, so a leaked default doesn't hit driver.
// Also zeroes the shadow-ray dist marker so the end-of-raygen visibility
// test skips pixels where no NEE candidate was ever accepted.
void init_ps(RWByteAddressBuffer buf, uint pixelIdx)
{
    buf.Store4(ps_addr_pack1(pixelIdx),
               uint4(asuint(float3(0, 0, 0)), PackNormal(float3(0, 1, 0))));
    buf.Store4(ps_addr_pack2(pixelIdx),
               uint4(PackFloat2x16(0.0f, 0.0f),
                     MATID_ENV_MISS, MATID_ENV_MISS, asuint(1.0f)));
    buf.Store(ps_addr_v2(pixelIdx), PackNormal(float3(0, 1, 0)));
    buf.Store(ps_addr_shadow(pixelIdx) + 16u, 0u);  // dist = 0 → no shadow test
}

// Deferred-shadow-ray helpers. Used by raygen NEE to postpone the visibility
// test until the final RIS winner is known. dist > 0 means "this NEE sample
// is the current RIS winner; at raygen end, trace from origin along dir for
// tMax=dist and zero the sample's W if occluded." dist == 0 means no test.
struct ShadowRayInfo {
    float3 origin;  // already offset via offset_ray
    float3 dir;
    float  dist;
};

void store_shadow_ray(RWByteAddressBuffer buf, uint pixelIdx,
                      float3 origin, float3 dir, float dist)
{
    const uint addr = ps_addr_shadow(pixelIdx);
    buf.Store3(addr,        asuint(origin));
    buf.Store (addr + 12u,  PackNormal(dir));
    buf.Store (addr + 16u,  asuint(dist));
}

// Mark the slot as "no shadow test needed" without touching the origin/dir
// (they're unread when dist == 0, so writing just the dist word is enough).
void clear_shadow_ray(RWByteAddressBuffer buf, uint pixelIdx)
{
    buf.Store(ps_addr_shadow(pixelIdx) + 16u, 0u);
}

ShadowRayInfo load_shadow_ray(RWByteAddressBuffer buf, uint pixelIdx)
{
    const uint addr = ps_addr_shadow(pixelIdx);
    ShadowRayInfo s;
    s.origin = asfloat(buf.Load3(addr));
    s.dir    = UnpackNormal(buf.Load(addr + 12u));
    s.dist   = asfloat(buf.Load(addr + 16u));
    return s;
}
