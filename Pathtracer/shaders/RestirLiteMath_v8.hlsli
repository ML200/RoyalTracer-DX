#pragma once
// Diffuse reuse (Bitterli 2022 confidence weighting): samples, reservoirs, links, target function and
// MIS terms. Pure functions only; the buffers and rays live in RestirLite_v8.hlsli.



static const uint  LITE_INF = 0xffffffffu;
static const uint  LITE_EMPTY = 0xfffffffeu;
static const uint  LITE_KIND_LIGHT = 0u;
static const uint  LITE_KIND_SURFACE = 1u;
static const uint  LITE_KIND_INF = 2u;
static const float LITE_RADIANCE_SCALE = 64.0f;
static const float LITE_TINT_STEPS = 127.0f;
static const float LITE_INV_PI = 0.31830988618f;

struct LiteSample
{
    float3 position;
    uint   instance;
    float3 radiance;
    float3 normal;
    uint   kind;
};

struct LiteReservoir
{
    LiteSample s;
    float  W;
    uint   M;
    float3 tint;
};

uint LiteAddress(uint px) { return px * LITE_RESERVOIR_BYTES; }
bool LiteHasSample(LiteSample s) { return s.instance != LITE_EMPTY; }

bool LiteSameSample(LiteSample a, LiteSample b)
{
    return a.instance == b.instance && all(a.position == b.position);
}

// Quantize radiance before storing compact spatial candidates.
float3 LiteQuantizeRadiance(float3 L)
{
    return UnpackRGB9E5(PackRGB9E5(max(L, 0.0f) / LITE_RADIANCE_SCALE)) * LITE_RADIANCE_SCALE;
}

uint LitePackMeta(uint M, uint kind, float3 tint)
{
    const uint3 t = (uint3)round(saturate(tint) * LITE_TINT_STEPS);
    return min(M, 255u) | ((kind & 3u) << 8u) | (t.x << 10u) | (t.y << 17u) | (t.z << 24u);
}

LiteReservoir LiteEmpty(uint M)
{
    LiteReservoir r;
    r.s.position = 0.0f;
    r.s.instance = LITE_EMPTY;
    r.s.radiance = 0.0f;
    r.s.normal = 0.0f;
    r.s.kind = 0u;
    r.W = 0.0f;
    r.M = M;
    r.tint = 0.0f;
    return r;
}

LiteReservoir LiteLoad(RWByteAddressBuffer buf, uint px)
{
    const uint a = LiteAddress(px);
    const uint4 w0 = buf.Load4(a);
    const uint4 w1 = buf.Load4(a + 16u);
    LiteReservoir r;
    r.s.position = asfloat(w0.xyz);
    r.s.instance = w0.w;
    r.s.radiance = UnpackRGB9E5(w1.x) * LITE_RADIANCE_SCALE;
    r.s.normal = UnpackNormal(w1.y);
    r.W = asfloat(w1.z);
    r.M = w1.w & 255u;
    r.s.kind = (w1.w >> 8u) & 3u;
    r.tint = float3((w1.w >> 10u) & 127u, (w1.w >> 17u) & 127u, (w1.w >> 24u) & 127u) / LITE_TINT_STEPS;
    return r;
}

void LiteStore(RWByteAddressBuffer buf, uint px, LiteReservoir r)
{
    const uint a = LiteAddress(px);
    buf.Store4(a, uint4(asuint(r.s.position), r.s.instance));
    buf.Store4(a + 16u, uint4(PackRGB9E5(r.s.radiance / LITE_RADIANCE_SCALE),
        PackNormal(r.s.normal), asuint(r.W), LitePackMeta(r.M, r.s.kind, r.tint)));
}


LiteSample LiteSampleDirection(float3 direction, float3 radiance)
{
    LiteSample s;
    s.instance = LITE_INF;
    s.position = direction;
    s.normal = 0.0f;
    s.radiance = LiteQuantizeRadiance(radiance);
    s.kind = LITE_KIND_INF;
    return s;
}

struct LiteReceiver
{
    float3 x;
    float3 n;
    float3 albedo;
};


struct LiteLink
{
    float3 dir;
    float  dist;
    float  cosX;
    float  cosY;
    float  geom;
};

LiteLink LiteLinkFrom(float dist, float cosX, float cosY, bool infinite)
{
    LiteLink l;
    l.dir = 0.0f;
    l.dist = dist;
    l.cosX = cosX;
    l.cosY = infinite ? 1.0f : cosY;
    l.geom = infinite ? cosX : cosX * cosY / max(dist * dist, 1e-12f);
    return l;
}

// Evaluate geometry terms between receiver and reused sample.
LiteLink LiteConnect(LiteReceiver r, LiteSample s, float3 yWorld, float3 nyWorld)
{
    LiteLink l;
    if (s.instance == LITE_INF)
    {
        l.dir = s.position;
        l.dist = RAY_TMAX_PLANET;
        l.cosX = dot(r.n, l.dir);
        l.cosY = 1.0f;
        l.geom = l.cosX;
        return l;
    }
    const float3 d = yWorld - r.x;
    const float d2 = dot(d, d);
    l.dist = sqrt(d2);
    l.dir = d / max(l.dist, 1e-8f);
    l.cosX = dot(r.n, l.dir);
    l.cosY = dot(nyWorld, -l.dir);
    l.geom = l.cosX * l.cosY / max(d2, 1e-12f);
    return l;
}

bool LiteLinkValid(LiteLink l)
{
    return l.cosX > 1e-6f && l.cosY > 1e-6f && l.dist > 1e-5f;
}

float LiteTarget(float3 albedo, LiteSample s, LiteLink l, float3 yWorld, float visibility)
{
    if (!LiteLinkValid(l) || !(visibility > 0.0f)) return 0.0f;
    return Luma(albedo * s.radiance) * LITE_INV_PI * l.geom * visibility;
}


// Bitterli 2022 pairwise MIS.

// Compute the partner reservoir MIS correction.
float LiteMisPartner(float Mi, float pii, float Mc, float pci, float O, float Msum)
{
    const float num = O * pii;
    const float den = num + Mc * pci;
    return den > 0.0f ? (Mi / Msum) * (num / den) : 0.0f;
}

float LiteMisCanonicalTerm(float Mi, float pcc, float Mc, float pic, float O, float Msum)
{
    const float num = Mc * pcc;
    const float den = num + O * pic;
    return den > 0.0f ? (Mi / Msum) * (num / den) : 0.0f;
}

float LiteSanitizeWeight(float W)
{
    if (!(W > 0.0f) || isinf(W)) return 0.0f;
    return W;
}
