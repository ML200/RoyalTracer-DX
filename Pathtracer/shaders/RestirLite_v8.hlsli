#pragma once
// Diffuse reuse against the scene: world-space samples, visibility, the reuse tables in the cache
// buffer, the park state in the path-state buffer and the per-pixel reservoirs.
#include "Sharc_v8.hlsli"
#include "RestirLiteMath_v8.hlsli"

#define LITE_ENABLED    ((rs_flags & LITE_FLAG_ENABLED) != 0u)
#define LITE_SPATIAL    ((rs_flags & LITE_FLAG_SPATIAL) != 0u)
#define LITE_UNSHADOWED ((rs_flags & LITE_FLAG_UNSHADOWED) != 0u)
#define LITE_DEBUG      ((rs_flags & LITE_FLAG_DEBUG) != 0u)

float3 LiteWorldPosition(LiteSample s)
{
    return s.instance == LITE_INF ? s.position : ObjectToWorldPos(s.instance, s.position);
}

float3 LiteWorldNormal(LiteSample s)
{
    return s.instance == LITE_INF ? s.normal : ObjectToWorldNrm(s.instance, s.normal);
}

// Store surface samples in object space with a packed normal.
LiteSample LiteSampleSurface(uint instance, float3 worldPos, float3 worldNormalFacing, float3 radiance, uint kind)
{
    LiteSample s;
    s.instance = instance;
    s.position = WorldToObjectPos(instance, worldPos);
    s.normal = WorldToObjectNrm(instance, worldNormalFacing);
    s.radiance = LiteQuantizeRadiance(radiance);
    s.kind = kind;
    return s;
}
LiteReceiver LiteReceiverFromSD(SDRecord sd)
{
    LiteReceiver r;
    r.x = sd.x1;
    r.n = sd.n1_s;
    r.albedo = sd.Kd;
    return r;
}
float3 LiteVisibility(LiteReceiver r, LiteSample s, LiteLink l, float3 yWorld, float3 nyWorld)
{
    if (s.instance == LITE_INF)
        return VisibilityTransmittance(r.x, r.n, r.x + l.dir * RAY_TMAX_PLANET, -l.dir);
    return VisibilityTransmittance(r.x, r.n, yWorld, nyWorld);
}

bool LiteSimilar(LiteReceiver a, LiteReceiver b, float camDistA, float camDistB)
{
    if (dot(a.n, b.n) < lite_normalSimCos) return false;
    if (abs(dot(b.x - a.x, a.n)) > lite_planeDist * camDistA) return false;
    if (abs(dot(a.x - b.x, b.n)) > lite_planeDist * camDistB) return false;
    return true;
}

float3 LiteExactBroad(SDRecord sd, float2 iors, float3 dir)
{
    const float3 view = normalize(InitOrigin() - sd.x1);
    const SamplingP sp = CalculateStrategyProbabilities(sd.matID, view, sd.n1_s,
        (half)iors.x, (half)iors.y, sd.Kd, (half)sd.Pr, (half)sd.Pm);
    if (!HasBroadShare(sp, (half)sd.Pr, (half)sd.Pm)) return 0.0f;
    return EvaluateLobePdf_COMBINED(sp, LOBE_BROAD, sd.matID, sd.n1_s, sd.n1_s, dir, view,
        sd.Kd, (half)sd.Pr, (half)sd.Pm, (half)iors.x, (half)iors.y).val;
}
uint LiteReuseSize(uint slot)
{
    return slot == 0u ? LITE_REUSE_SIZE0 : (slot == 1u ? LITE_REUSE_SIZE1 : LITE_REUSE_SIZE2);
}

uint LiteReuseBase(uint slot)
{
    uint texels = slot >= 1u ? LITE_REUSE_SIZE0 * LITE_REUSE_SIZE0 : 0u;
    texels += slot >= 2u ? LITE_REUSE_SIZE1 * LITE_REUSE_SIZE1 : 0u;
    return LITE_REUSE_OFFSET + texels * 4u;
}

uint LiteReusePacked(uint slot)
{
    return slot == 0u ? lite_reuse0 : (slot == 1u ? lite_reuse1 : lite_reuse2);
}

int2 LiteReuseDelta(uint2 pixel, uint slot)
{
    const uint packed = LiteReusePacked(slot);
    const int size = (int)LiteReuseSize(slot);
    const int2 offset = int2(packed & 255u, (packed >> 8u) & 255u);
    const uint flags = packed >> LITE_REUSE_FLAGS_SHIFT;
    int2 c = (int2(pixel) + offset) % size;
    if (flags & 4u) c = c.yx;
    if (flags & 1u) c.x = size - 1 - c.x;
    if (flags & 2u) c.y = size - 1 - c.y;
    const uint word = g_sharc.Load(LiteReuseBase(slot) + (uint)(c.y * size + c.x) * 4u);
    int2 d = int2((int)(word << 16u) >> 16, (int)word >> 16);
    if (flags & 1u) d.x = -d.x;
    if (flags & 2u) d.y = -d.y;
    if (flags & 4u) d = d.yx;
    return d;
}

uint LiteShiftAddress(uint px, uint slot) { return ps_plane(PS_LITE_SHIFT_PLANE, PS_LITE_SHIFT_BYTES, px) + slot * 12u; }
uint LiteShiftMaskAddress(uint px) { return ps_plane(PS_LITE_SHIFT_PLANE, PS_LITE_SHIFT_BYTES, px) + 36u; }
uint LiteShiftOwnAddress(uint px) { return ps_plane(PS_LITE_SHIFT_PLANE, PS_LITE_SHIFT_BYTES, px) + 40u; }

uint LiteParkAddress(uint px) { return ps_numPx() * PS_LITE_PARK_PLANE + px * PS_LITE_PARK_BYTES; }
uint LiteParkStateAddress(uint px) { return ps_numPx() * PS_LITE_STATE_PLANE + px * PS_LITE_STATE_BYTES; }

void LiteMarkEmpty(uint px)
{
    g_liteReservoirs.Store(LiteAddress(px) + 28u, 0u);
}

void LiteParkStore(uint px, float3 broadWeight, float pdf)
{
    g_pathStateBuffer.Store4(LiteParkAddress(px), uint4(asuint(broadWeight), asuint(pdf)));
}

void LiteParkLoad(uint px, out float3 broadWeight, out float pdf)
{
    const uint4 w = g_pathStateBuffer.Load4(LiteParkAddress(px));
    broadWeight = asfloat(w.xyz);
    pdf = asfloat(w.w);
}

void LiteParkPointStore(uint px, float3 positionObj, uint instance, uint normalPk, float pdf)
{
    g_pathStateBuffer.Store4(LiteParkAddress(px), uint4(asuint(positionObj), instance));
    g_pathStateBuffer.Store2(LiteParkStateAddress(px) + 8u, uint2(normalPk, asuint(pdf)));
}

void LiteParkPointLoad(uint px, out LiteSample s, out float pdf)
{
    const uint4 a = g_pathStateBuffer.Load4(LiteParkAddress(px));
    const uint2 b = g_pathStateBuffer.Load2(LiteParkStateAddress(px) + 8u);
    s.position = asfloat(a.xyz);
    s.instance = a.w;
    s.normal = UnpackNormal(b.x);
    s.radiance = 0.0f;
    s.kind = LITE_KIND_SURFACE;
    pdf = asfloat(b.y);
}

// Apply weighted reservoir replacement to one compact candidate.
void LiteAddCandidate(uint px, LiteSample c, float3 tint, float phat, float w, inout uint seed)
{
    if (!(w > 0.0f) || !(phat > 0.0f) || isinf(w)) return;
    const uint2 state = g_pathStateBuffer.Load2(LiteParkStateAddress(px));
    float phatSel = asfloat(state.x);
    const float wsum = asfloat(state.y) + w;
    if (RandomFloatPCG(seed) * wsum < w)
    {
        phatSel = phat;
        LiteReservoir r;
        r.s = c;
        r.W = 0.0f;
        r.M = 1u;
        r.tint = tint;
        LiteStore(g_liteReservoirs, px, r);
    }
    g_liteReservoirs.Store(LiteAddress(px) + 24u, asuint(wsum / phatSel));
    g_pathStateBuffer.Store2(LiteParkStateAddress(px), uint2(asuint(phatSel), asuint(wsum)));
}

void LiteCandidate(uint px, float3 albedo, LiteSample c, float3 yWorld, LiteLink l, float3 visT,
    float misWeight, float pdfSolidAngle, float invN, inout uint seed)
{
    const float vis = Luma(visT);
    const float phat = LiteTarget(albedo, c, l, yWorld, vis);
    if (!(phat > 0.0f) || !(pdfSolidAngle > 0.0f)) return;
    const float w = misWeight * invN * Luma(albedo * c.radiance) * LITE_INV_PI * l.cosX * vis / pdfSolidAngle;
    LiteAddCandidate(px, c, visT, phat, w, seed);
}

void LiteCandidatePoint(uint px, float3 radiance, float invN, inout uint seed)
{
    LiteSample s;
    float pdf;
    LiteParkPointLoad(px, s, pdf);
    s.radiance = LiteQuantizeRadiance(radiance);
    const LiteReceiver r = LiteReceiverFromSD(load_SD(g_sample_current, px));
    const float3 y = LiteWorldPosition(s);
    const float3 ny = LiteWorldNormal(s);
    LiteCandidate(px, r.albedo, s, y, LiteConnect(r, s, y, ny), (float3)1.0f, 1.0f, pdf, invN, seed);
}

struct LiteGen
{
    LiteSample s;
    float3 tint;
    float  phatSel;
    float  wsum;
};

LiteGen LiteGenEmpty()
{
    LiteGen g;
    g.s = LiteEmpty(0u).s;
    g.tint = 0.0f;
    g.phatSel = 0.0f;
    g.wsum = 0.0f;
    return g;
}

LiteGen LiteGenLoad(uint px)
{
    const LiteReservoir r = LiteLoad(g_liteReservoirs, px);
    const uint2 state = g_pathStateBuffer.Load2(LiteParkStateAddress(px));
    LiteGen g;
    g.s = r.s;
    g.tint = r.tint;
    g.phatSel = asfloat(state.x);
    g.wsum = asfloat(state.y);
    return g;
}

void LiteGenCandidate(inout LiteGen g, float3 albedo, LiteSample c, float3 yWorld, LiteLink l, float3 visT,
    float misWeight, float pdfSolidAngle, float invN, inout uint seed)
{
    const float vis = Luma(visT);
    const float phat = LiteTarget(albedo, c, l, yWorld, vis);
    if (!(phat > 0.0f) || !(pdfSolidAngle > 0.0f)) return;
    const float w = misWeight * invN * Luma(albedo * c.radiance) * LITE_INV_PI * l.cosX * vis / pdfSolidAngle;
    if (!(w > 0.0f) || isinf(w)) return;
    g.wsum += w;
    if (RandomFloatPCG(seed) * g.wsum < w)
    {
        g.s = c;
        g.tint = visT;
        g.phatSel = phat;
    }
}

// Commit generated candidates after the full pixel estimate is known.
void LiteGenCommit(uint px, LiteGen g)
{
    LiteReservoir r;
    r.s = g.s;
    r.W = (g.wsum > 0.0f && g.phatSel > 0.0f) ? g.wsum / g.phatSel : 0.0f;
    r.M = 1u;
    r.tint = g.tint;
    LiteStore(g_liteReservoirs, px, r);
    g_pathStateBuffer.Store2(LiteParkStateAddress(px), uint2(asuint(g.phatSel), asuint(g.wsum)));
}
