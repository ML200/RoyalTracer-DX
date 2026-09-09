//====================================
//RESTIR LITE: RESAMPLED BROAD SHARE AT THE PRIMARY VERTEX
//====================================
// The regular path tracer splits the primary vertex's BROAD share (the
// diffuse lobe and a rough GGX lobe, LOBE_BROAD) off the radiance sum.
// Everything that share sees along one path is a candidate for
// the pixel's reservoir: a light-tree NEE point, the sun, an emitter or the
// sky the scatter ray found, or the cache-terminated secondary vertex (its
// broad-share outgoing radiance from SHaRC). A candidate is a point (or a
// direction) with a radiance, 32 bytes, so the reservoir passes carry no
// path state at all. The remaining lobes, cache misses and everything past
// the secondary vertex stay on the plain tracer.
//
// The secondary vertex. Every broad-share scatter ray at x1 yields one candidate:
// the point x2 with its outgoing radiance toward x1. On a cache hit that is
// the record (the vertex's direct light included, so nothing follows for the
// suffix); otherwise the tracer's continuation from x2 (NEE, sun,
// deeper bounces and cache hits) accumulates into the candidate's radiance
// along the path. The sample is then a point with an unbiased radiance
// estimate, as in ReSTIR GI, and whether the cache answered only changes its
// variance. x1's glossy lobes keep their share of the same path on the
// tracer, and on a cache hit the layers above x2's broad share are dropped
// for that share (their Fresnel remainder, a few percent).
//
// Target function p_hat_j(y) = lum(albedo_j * L(y)) / pi * G_j(y) * V_j(y):
// a Lambertian proxy (the exact gated lobe shades the winner), the
// geometric factor in area measure for points and solid angle for
// directions, and the receiver's visibility of the point. Visibility is in
// every target of every stage, so the supports of all strategies agree and
// the estimator is exact: one shadow ray per partner in the spatial shift.
// The A/B option LITE_FLAG_UNSHADOWED (off by default) drops V from the
// reuse targets and traces only the winner; generation still drops occluded
// NEE candidates, so the reuse MIS then over-credits a neighbour in a thin
// band along its own shadow edges (the classic visibility-reuse compromise,
// ~0.2 ms cheaper).
//
// Spatial reuse: the paired reuse of self-inverting delta tables (Lin,
// Kettunen, Wyman 2026): a partner's partner is the pixel itself, so each
// pixel evaluates and stores its own target of every partner's sample once,
// and the merge reads both directions of every pair. Defensive pairwise MIS
// with confidence weights (Bitterli 2022). Reuse is spatial only; no compact
// reservoir or resampled radiance is carried into the next frame.
//
// Buffers. g_Reservoirs_current holds this frame's candidate reservoir
// (written by Pass_pt, read by the spatial passes). Partner evaluations
// live in the path-state buffer's first 48 bytes per pixel (unused under the
// regular tracer), the reuse tables in the SHaRC allocation.
#ifndef RESTIR_LITE_V8_HLSLI
#define RESTIR_LITE_V8_HLSLI
#include "Sharc_v8.hlsli"

//====================================
//ROOT CONSTANT ALIASES AND FLAGS
//====================================
// Slots of the deprecated reservoir pipeline, repurposed while the regular
// path tracer owns the frame (Renderer.cpp packs them from ReSTIRSettings).
#define lite_spatMcap  rs_spatCountMax  // slot 5: spatial confidence cap
#define lite_spatSlots rs_spatCountMin  // slot 6: partners per pixel (0..3)
#define lite_reuse0    rs_spatRadMax    // slots 7, 8, 12: per-frame reuse table transforms
#define lite_reuse1    rs_spatRadMin
#define lite_reuse2    rs_spatTries
#define LITE_ENABLED    ((rs_flags & LITE_FLAG_ENABLED) != 0u)
#define LITE_SPATIAL    ((rs_flags & LITE_FLAG_SPATIAL) != 0u)
#define LITE_UNSHADOWED ((rs_flags & LITE_FLAG_UNSHADOWED) != 0u)
#define LITE_DEBUG      ((rs_flags & LITE_FLAG_DEBUG) != 0u)

static const uint  LITE_INF = 0xffffffffu;   // instance of a direction sample (sky, sun)
static const uint  LITE_EMPTY = 0xfffffffeu; // no sample
static const uint  LITE_KIND_LIGHT = 0u;     // point on an emitter (NEE or scatter hit)
static const uint  LITE_KIND_SURFACE = 1u;   // secondary vertex, cached or traced radiance
static const uint  LITE_KIND_INF = 2u;       // sky or sun direction
static const uint  LITE_RESERVOIR_BYTES = 32u;
static const float LITE_RADIANCE_SCALE = 64.0f;       // RGB9E5 range 2^16 -> 4e6
static const float LITE_TINT_STEPS = 127.0f;
static const float LITE_INV_PI = 0.31830988618f;

//====================================
//SAMPLE AND RESERVOIR
//====================================
// Positions and normals of surface samples are OBJECT space of `instance`
// (Sample_Data_v8.hlsli conventions): they survive floating-origin snaps
// and follow moving objects. A direction sample keeps the world direction.
// The normal of a surface sample faces the receiver that generated it, so a
// receiver on the other side sees a zero geometric factor: an emitter's or
// a cache record's outgoing radiance is one-sided.
struct LiteSample
{
    float3 position;
    uint   instance;
    float3 radiance;   // toward the receiver (view-independent for the kinds cached)
    float3 normal;
    uint   kind;
};

// tint: the OWNING pixel's visibility transmittance of the sample (thin
// glass tints), so its target and final shading never trace it again.
struct LiteReservoir
{
    LiteSample s;
    float  W;
    uint   M;
    float3 tint;
};

uint LiteAddress(uint px) { return px * LITE_RESERVOIR_BYTES; }
bool LiteHasSample(LiteSample s) { return s.instance != LITE_EMPTY; }

// The same point (bit-identical after reuse): its visibility from a given
// receiver is whatever that receiver already established, so evaluating it
// again needs no ray.
bool LiteSameSample(LiteSample a, LiteSample b)
{
    return a.instance == b.instance && all(a.position == b.position);
}

// Quantize once at generation, so every target evaluated later sees exactly
// the stored radiance (RGB9E5 under a 1/64 scale: a 4e6 range, and every
// channel within 1/1024 of the brightest one; the estimator converges to the
// integral of the stored value, so nothing statistical depends on this).
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

#ifndef SHARC_TEST
float3 LiteWorldPosition(LiteSample s)
{
    return s.instance == LITE_INF ? s.position : ObjectToWorldPos(s.instance, s.position);
}

float3 LiteWorldNormal(LiteSample s)
{
    return s.instance == LITE_INF ? s.normal : ObjectToWorldNrm(s.instance, s.normal);
}

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
#endif

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

//====================================
//RECEIVER, LINK AND TARGET
//====================================
struct LiteReceiver
{
    float3 x;
    float3 n;       // shading normal
    float3 albedo;  // texture-resolved Kd of the G-buffer
};

#ifndef SHARC_TEST
LiteReceiver LiteReceiverFromSD(SDRecord sd)
{
    LiteReceiver r;
    r.x = sd.x1;
    r.n = sd.n1_s;
    r.albedo = sd.Kd;
    return r;
}
#endif

// Geometry between a receiver and a sample. geom is the measure conversion
// the target uses: cos_x cos_y / d^2 for points (area measure), cos_x for
// directions (solid angle). Both sides of a pair evaluate this identically.
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

// Area-measure (or solid-angle) target of the receiver for the sample.
float LiteTarget(float3 albedo, LiteSample s, LiteLink l, float3 yWorld, float visibility)
{
    if (!LiteLinkValid(l) || !(visibility > 0.0f)) return 0.0f;
    return Luma(albedo * s.radiance) * LITE_INV_PI * l.geom * visibility;
}

#ifndef SHARC_TEST
float3 LiteVisibility(LiteReceiver r, LiteSample s, LiteLink l, float3 yWorld, float3 nyWorld)
{
    if (s.instance == LITE_INF)
        return VisibilityTransmittance(r.x, r.n, r.x + l.dir * RAY_TMAX_PLANET, -l.dir);
    return VisibilityTransmittance(r.x, r.n, yWorld, nyWorld);
}

// Symmetric receiver similarity (normal cone, plane distance as a fraction
// of the camera distance on both sides), so both pixels of a pair agree.
bool LiteSimilar(LiteReceiver a, LiteReceiver b, float camDistA, float camDistB)
{
    if (dot(a.n, b.n) < temp_normalSimCos) return false;
    if (abs(dot(b.x - a.x, a.n)) > temp_planeDist * camDistA) return false;
    if (abs(dot(a.x - b.x, b.n)) > temp_planeDist * camDistB) return false;
    return true;
}

// Exact gated broad share of the primary surface toward `dir`: the diffuse
// lobe and a broad GGX lobe under the transmittance of the layers above them,
// the same evaluation the tracer uses for its share (LOBE_BROAD in
// EvaluateAndPdf_COMBINED_L).
float3 LiteExactBroad(SDRecord sd, float2 iors, float3 dir)
{
    const float3 view = normalize(InitOrigin() - sd.x1);
    const SamplingP sp = CalculateStrategyProbabilities(sd.matID, view, sd.n1_s,
        (half)iors.x, (half)iors.y, sd.Kd, (half)sd.Pm);
    if (!HasBroadShare(sp, (half)sd.Pr, (half)sd.Pm)) return 0.0f;
    return EvaluateLobePdf_COMBINED(sp, LOBE_BROAD, sd.matID, sd.n1_s, sd.n1_s, dir, view,
        sd.Kd, (half)sd.Pr, (half)sd.Pm, (half)iors.x, (half)iors.y).val;
}
#endif

//====================================
//RESAMPLING MIS
//====================================
// Defensive pairwise MIS (Bitterli 2022) for a canonical c against partners
// i with confidences M. O = sum of the partners' confidences, Msum = O + Mc.
// The partner weight compares its own target with the canonical's at its
// sample; the canonical weight is Mc / Msum plus one term per partner. The
// weights of all strategies sum to one for every sample the canonical can
// produce, which is every sample with a nonzero contribution here.
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

// Point reservoirs use area measure: W contains the inverse area PDF and
// grows with distance squared / receiver-facing cosine. The legacy PSS
// reuse clamp is in a different measure and would darken valid distant or
// grazing samples, even when reuse is off. Preserve every positive finite
// contribution weight.
float LiteSanitizeWeight(float W)
{
    if (!(W > 0.0f) || isinf(W)) return 0.0f;
    return W;
}

//====================================
//PAIRED REUSE TABLES
//====================================
#ifndef SHARC_TEST
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

// Partner offset of `pixel` in table `slot`: the table is looked up under
// the frame's wrap offset and flip/transpose, and the delta is transformed
// back, which keeps the pairing self-inverting (the partner's delta returns
// here) while the pair pattern changes every frame.
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

// Partner evaluations (Pass_lite_shift -> Pass_lite_merge): per pixel, per
// slot {target of the partner's sample, visibility tint, partner pixel},
// then the valid-slot mask and the pixel's own target of its own sample.
// 48 of the 52 bytes per pixel of path-state planes unused under PT.
static const uint LITE_SHIFT_BYTES = 48u;
uint LiteShiftAddress(uint px, uint slot) { return px * LITE_SHIFT_BYTES + slot * 12u; }
uint LiteShiftMaskAddress(uint px) { return px * LITE_SHIFT_BYTES + 36u; }
uint LiteShiftOwnAddress(uint px) { return px * LITE_SHIFT_BYTES + 40u; }

//====================================
//GENERATION (Pass_pt)
//====================================
// The candidate reservoir lives in memory while the path runs, so nothing
// of it is live across a trace. Plane PACK1 of the path-state buffer parks
// the primary's broad-share scatter weight and pdf across the trace, plane
// PACK2 the running target of the selected candidate and the weight sum.
uint LiteParkAddress(uint px) { return px * 16u; }
uint LiteParkStateAddress(uint px) { return ps_numPx() * 16u + px * 16u; }

// A pixel without a broad share: M = 0 in the meta word is all any reader
// tests before touching the rest of the entry.
void LiteMarkEmpty(uint px)
{
    g_Reservoirs_current.Store(LiteAddress(px) + 28u, 0u);
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

// Once the scatter ray has found x2, the same bytes park the pending
// candidate point instead (object-space position and instance in PACK1, the
// normal facing x1 and the scatter pdf behind the running target/weight sum
// in PACK2). Its radiance accumulates along the path in registers, and the
// candidate is added when the path ends (LiteCandidatePoint).
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

// Weighted reservoir sampling against the reservoir in memory. w is the
// full resampling weight (MIS weight, 1/N and target over pdf), phat the
// generation target of the candidate.
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
        LiteStore(g_Reservoirs_current, px, r);
    }
    g_Reservoirs_current.Store(LiteAddress(px) + 24u, asuint(wsum / phatSel));
    g_pathStateBuffer.Store2(LiteParkStateAddress(px), uint2(asuint(phatSel), asuint(wsum)));
}

// One candidate seen by the primary's diffuse lobe. pdfSolidAngle is the
// density the candidate was drawn with; for a cache vertex the gate that
// also scales the target cancels out of the weight. misWeight is the
// balance-heuristic weight against the other technique that could have
// produced the same point (NEE vs scatter), invN the initial-sample share.
void LiteCandidate(uint px, float3 albedo, LiteSample c, float3 yWorld, LiteLink l, float3 visT,
    float misWeight, float pdfSolidAngle, float invN, inout uint seed)
{
    const float vis = Luma(visT);
    const float phat = LiteTarget(albedo, c, l, yWorld, vis);
    if (!(phat > 0.0f) || !(pdfSolidAngle > 0.0f)) return;
    const float w = misWeight * invN * Luma(albedo * c.radiance) * LITE_INV_PI * l.cosX * vis / pdfSolidAngle;
    LiteAddCandidate(px, c, visT, phat, w, seed);
}

// The pending secondary-vertex candidate at the end of its path: the parked
// point with the outgoing radiance its diffuse suffix accumulated, produced
// by the scatter technique alone (MIS weight one) at the parked pdf, seen
// through this pixel's own geometry.
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

// Register-resident reservoir for the candidates generated before the trace
// (the NEE samples), committed to memory once; the post-trace candidate then
// resamples against memory (LiteCandidate).
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

// Later initial samples of a pixel continue the reservoir in memory.
LiteGen LiteGenLoad(uint px)
{
    const LiteReservoir r = LiteLoad(g_Reservoirs_current, px);
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

void LiteGenCommit(uint px, LiteGen g)
{
    LiteReservoir r;
    r.s = g.s;
    r.W = (g.wsum > 0.0f && g.phatSel > 0.0f) ? g.wsum / g.phatSel : 0.0f;
    r.M = 1u;
    r.tint = g.tint;
    LiteStore(g_Reservoirs_current, px, r);
    g_pathStateBuffer.Store2(LiteParkStateAddress(px), uint2(asuint(g.phatSel), asuint(g.wsum)));
}
#endif // SHARC_TEST

#endif
