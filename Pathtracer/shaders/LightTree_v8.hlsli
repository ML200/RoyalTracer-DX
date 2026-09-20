#include "LightTreeTrail.h"
#include "LightTreeDecode.hlsli"

Buffer<uint> gLT_TriToBLAS       : register(t16);
Buffer<uint2> gLT_TriBitTrail    : register(t17);

Buffer<uint2> gLT_BLASBitTrail   : register(t18);

static const uint  LT_SENTINEL = 0xFFFFFFFFu;
static const float LT_PI = 3.14159265358979323846;

uint LT_SlotOfInstance(uint inst)
{
    return inst == LT_SENTINEL ? LT_SENTINEL : instanceProps[inst].lightSlot;
}

inline uint LT_PickAndRescale(float w0, float w1, float w2, float w3, uint n, float xi_in,
                              out float p_chosen, out float xi_out)
{
    float sum = 0.0;
    if (n > 0u) sum += max(w0, 0.0);
    if (n > 1u) sum += max(w1, 0.0);
    if (n > 2u) sum += max(w2, 0.0);
    if (n > 3u) sum += max(w3, 0.0);

    if (sum <= 0.0) {
        uint idx = min((uint)floor(xi_in * n), n-1);
        p_chosen = 1.0 / float(n);
        float start = float(idx) / float(n);
        float width = 1.0 / float(n);
        xi_out = (xi_in - start) / width;
        return idx;
    }
    float target = xi_in * sum;
    float accum  = 0.0;
    uint  idx    = n-1;
    float wsel   = n == 1u ? max(w0, 0.0) : (n == 2u ? max(w1, 0.0) : (n == 3u ? max(w2, 0.0) : max(w3, 0.0)));
    bool  found  = false;
    {
        const float wi = max(w0, 0.0); const float next = accum + wi;
        if (!found && n > 0u) { if (target < next) { idx = 0u; wsel = wi; found = true; } else accum = next; }
    }
    {
        const float wi = max(w1, 0.0); const float next = accum + wi;
        if (!found && n > 1u) { if (target < next) { idx = 1u; wsel = wi; found = true; } else accum = next; }
    }
    {
        const float wi = max(w2, 0.0); const float next = accum + wi;
        if (!found && n > 2u) { if (target < next) { idx = 2u; wsel = wi; found = true; } else accum = next; }
    }
    {
        const float wi = max(w3, 0.0); const float next = accum + wi;
        if (!found && n > 3u) { if (target < next) { idx = 3u; wsel = wi; found = true; } else accum = next; }
    }
    p_chosen = wsel / sum;
    xi_out = (wsel > 0.0) ? ((target - accum) / wsel) : 0.0;
    xi_out = clamp(xi_out, 0.0, ONE_MINUS_EPSILON);
    return idx;
}

struct LTLeaf { uint triFirst; uint triCount; uint nodeIndex; };

// ---------------------------------------------------------------------------------------------
// Receiver: the shading point with the parts of its BSDF the node importance accounts for.
// Positions and directions are in the space of the tree level being traversed.
// ---------------------------------------------------------------------------------------------
struct LT_Receiver
{
    float3 x;        // position
    float3 n;        // unit shading normal
    float3 i;        // unit direction towards the previous vertex (the view direction)
    float3 t;        // unit tangent of the anisotropy frame, orthogonal to n
    float  wd;       // reflectance of the diffuse lobes over pi
    float  wg;       // reflectance of the broad glossy lobes; zero skips the glossy term
    float2 u;        // GGX alpha^2 / (1 - alpha^2) per tangent axis, the NDF's covariance form
    float  kInv;     // 1 / (4 (i.n)^2): light variance to halfvector variance at the NDF peak
    float  lambdaR;  // sharpness of the SG that bounds the reflection lobe
};

float3 LT_Orthonormal(float3 n, float3 t)
{
    t -= n * dot(n, t);
    const float l2 = dot(t, t);
    if (l2 > 1e-8f) return t * rsqrt(l2);
    const float3 a = abs(n.x) < 0.9f ? float3(1, 0, 0) : float3(0, 1, 0);
    return normalize(cross(a, n));
}

LT_Receiver LT_MakeReceiver(float3 x, float3 n, float3 view, float3 tangent, float wd, float wg, float ax, float ay)
{
    LT_Receiver R;
    R.x = x; R.n = n; R.i = view; R.t = LT_Orthonormal(n, tangent);
    R.wd = wd * (1.0f / LT_PI); R.wg = wg;
    const float2 a2 = clamp(float2(ax * ax, ay * ay), 1e-6f, 0.999f);
    R.u = a2 / (1.0f - a2);
    const float cosI = max(dot(view, n), 1e-3f);
    R.kInv = 0.25f / (cosI * cosI);
    const float amax2 = max(a2.x, a2.y);
    R.lambdaR = (1.0f - amax2) / (2.0f * amax2);
    return R;
}
// A receiver without a material: the diffuse importance only (the learned cuts' prior).
LT_Receiver LT_DiffuseReceiver(float3 x, float3 n)
{
    return LT_MakeReceiver(x, n, n, float3(0.0f, 0.0f, 1.0f), 1.0f, 0.0f, 1.0f, 1.0f);
}
// Packed receiver (16 bytes): view direction, lobe weights, roughnesses and tangent. The light
// sample of a vertex and the MIS weights of the emitter hits from it both unpack this form,
// so their densities agree bit for bit.
uint4 LT_PackReceiver(float3 view, float3 tangent, float wd, float wg, float ax, float ay)
{
    return uint4(PackNormal(view), PackFloat2x16(wd, wg), PackFloat2x16(ax, ay), PackNormal(tangent));
}
LT_Receiver LT_UnpackReceiver(uint4 p, float3 x, float3 n)
{
    float wd, wg, ax, ay;
    UnpackFloat2x16(p.y, wd, wg);
    UnpackFloat2x16(p.z, ax, ay);
    return LT_MakeReceiver(x, n, UnpackNormal(p.x), UnpackNormal(p.w), wd, wg, ax, ay);
}

float3 LT_LocalReceiverNormal(float3x4 worldToLocal,float3 normal) {
    float3 a=worldToLocal[0].xyz,b=worldToLocal[1].xyz,c=worldToLocal[2].xyz;
    float3 cof0=cross(b,c),cof1=cross(c,a),cof2=cross(a,b);
    float3 pullback=float3(dot(cof0,normal),dot(cof1,normal),dot(cof2,normal));
    return normalize(pullback*(dot(a,cof0)<0.0f?-1.0f:1.0f));
}
// Move the receiver into the space of a mesh tree.
LT_Receiver LT_ReceiverToLocal(LT_Receiver R, float3x4 worldToLocal)
{
    const float3x3 M = (float3x3)worldToLocal;
    LT_Receiver L = R;
    L.x = mul(worldToLocal, float4(R.x, 1.0f));
    L.n = LT_LocalReceiverNormal(worldToLocal, R.n);
    const float3 i = mul(M, R.i);
    const float i2 = dot(i, i);
    L.i = i2 > 1e-20f ? i * rsqrt(i2) : L.n;
    L.t = LT_Orthonormal(L.n, mul(M, R.t));
    const float cosI = max(dot(L.i, L.n), 1e-3f);
    L.kInv = 0.25f / (cosI * cosI);
    return L;
}

// ---------------------------------------------------------------------------------------------
// Spherical Gaussian lighting (Tokuyoshi, Ikeda, Kulkarni, Harada 2024).
// ---------------------------------------------------------------------------------------------
float LT_Erfc(float x)
{
    // Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
    const float ax = abs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    const float r = p * exp(-ax * ax);
    return x >= 0.0f ? r : 2.0f - r;
}
// (1 - e^-l) / l.
float LT_Expm1Ratio(float l)
{
    return l < 1e-2f ? 1.0f - l * (0.5f - l * (1.0f / 6.0f - l * (1.0f / 24.0f))) : (1.0f - exp(-l)) / l;
}
// Inverse variance alpha^2 of the planar Gaussian that stands in for an SG of sharpness l in
// the product integral with the clamped cosine: alpha^2 = l^2 / (2 l + c(l)), with c fitted
// against the exact integral (worst RMS error 3e-4 over the normalized integral).
float LT_PlanarAlphaSq(float l)
{
    const float c = (8.64288f + 0.185745f * l * l) / (1.0f + l * (0.133794f + 0.0828773f * l));
    return l / (2.0f + c / l);
}
// Normalized product integral of the planar Gaussian and max(v, 0) with the Gaussian centred at
// v = m; the interpolation weight of Eq. 7 is the difference of two of these.
float LT_PlanarG(float m, float a)
{
    return exp(-a * a * m * m) / (2.0f * sqrt(LT_PI) * a) + 0.5f * m * LT_Erfc(-a * m);
}
// Peak value of a vMF lobe: kappa / (2 pi (1 - e^-2 kappa)).
float LT_VmfPeak(float kappa)
{
    return kappa < 1e-3f ? (1.0f + kappa) * (0.25f / LT_PI) : kappa / (2.0f * LT_PI * (1.0f - exp(-2.0f * kappa)));
}

// Importance of a light cluster for the receiver: the SG light the cluster forms at the
// receiver (the product of its spatial Gaussian and its emission vMF), integrated against the
// clamped cosine (Eq. 7) and against the microfacet reflection lobe with the NDF filtered by
// the light (Eq. 12). Positive whenever the cluster can light the receiver; zero only when its
// bounding sphere lies below the receiver's horizon or behind every emitter.
float LT_NodeImportance(LT_Receiver R, float3 mean, float variance, float3 axis, float kappa,
    float power, float radius, float cosThetaO)
{
    if (!(power > 0.0f)) return 0.0f;
    const float3 dv = mean - R.x;
    if (dot(dv, R.n) + radius <= 0.0f) return 0.0f;
    const float  d2 = max(dot(dv, dv), 1e-24f);
    const float  d  = sqrt(d2);
    const float3 o  = dv / d;
    const float  c  = dot(o, axis);
    {
        const float sinU = min(radius / d, 1.0f);
        const float cosU = sqrt(1.0f - sinU * sinU);
        const float sinO = sqrt(saturate(1.0f - cosThetaO * cosThetaO));
        if (cosThetaO * cosU - sinO * sinU > 0.0f && c > sinO * cosU + cosThetaO * sinU) return 0.0f;
    }
    // The bound-based variance takes over as the mean sinks below the horizon.
    const float th  = saturate(-dot(o, R.n));
    const float var = max(lerp(variance, 0.5f * radius * radius, th), max(d2 * 1e-15f, 1e-30f));
    const float lambdaS = clamp(d2 / var, 1e-8f, 1e15f);
    // The SG light: axis, sharpness and the exponent of its amplitude, kept free of the
    // cancellation between the two sharpnesses.
    const float  q     = kappa / lambdaS;
    const float  ratio = sqrt(max(1.0f + q * q - 2.0f * q * c, 1e-12f));
    const float  lambda = max(lambdaS * ratio, 1e-8f);
    const float3 xiRaw = o - q * axis;
    const float  xi2   = dot(xiRaw, xiRaw);
    const float3 xi    = xi2 > 1e-12f ? xiRaw * rsqrt(xi2) : o;
    const float  expo  = lambdaS * (q * q - 2.0f * q * c) / (ratio + 1.0f) - kappa;
    const float  A     = power * LT_VmfPeak(kappa) * exp(expo) / (var * lambda);
    const float  E1 = exp(-lambda);
    const float  m1 = LT_Expm1Ratio(lambda);
    float bracket;
    {
        const float m = dot(xi, R.n);
        const float a = sqrt(LT_PlanarAlphaSq(lambda));
        const float t = clamp(LT_PlanarG(m, a) - LT_PlanarG(-1.0f, a), 1e-5f, 1.0f);
        bracket = R.wd * ((1.0f - m1) * t + E1 * (m1 - E1) * (1.0f - t));
    }
    if (R.wg > 0.0f)
    {
        // The light's variance filters the NDF in halfvector space; the lobe is evaluated at the
        // halfvector of the light axis and weighted by the share of the widened reflection lobe
        // above the horizon.
        const float2 u  = R.u + 2.0f * R.kInv / lambda;
        const float2 a2 = u / (1.0f + u);
        const float3 b  = cross(R.n, R.t);
        float3 h = R.i + xi;
        const float h2 = dot(h, h);
        h = h2 > 1e-12f ? h * rsqrt(h2) : R.n;
        if (dot(h, R.n) < 0.0f) h = -h;
        const float3 m3 = float3(dot(h, R.t), dot(h, b), dot(h, R.n));
        const float  P  = a2.x * a2.y;
        const float  den = m3.x * m3.x * a2.y + m3.y * m3.y * a2.x + m3.z * m3.z * P;
        const float  D  = P * sqrt(P) / (LT_PI * den * den);
        const float3 i3 = float3(dot(R.i, R.t), dot(R.i, b), dot(R.i, R.n));
        const float  rho = D / (4.0f * sqrt(i3.x * i3.x * a2.x + i3.y * i3.y * a2.y + i3.z * i3.z));
        const float3 wr = 2.0f * dot(R.i, R.n) * R.n - R.i;
        const float3 pv = R.lambdaR * wr + lambda * xi;
        const float  lb = max(length(pv), 1e-8f);
        const float  cb = dot(pv, R.n) / lb;
        const float  Eb = exp(-lb);
        const float  sb = 0.5f * LT_Erfc(-sqrt(LT_PlanarAlphaSq(lb)) * cb);
        const float  V  = max((Eb + (1.0f - Eb) * sb) / (1.0f + Eb), 1e-5f);
        bracket += R.wg * V * rho * m1 * (1.0f + E1);
    }
    return A * bracket;
}

struct LTNodeCommon
{
    float3 mean; float variance; float3 axis; float kappa;
    float power; float radius; float cosThetaO;
    uint4 topology;
};
LTNodeCommon LT_FromTLAS(LightTLASNodeGpu a)
{
    LTNodeCommon c;
    c.mean = a.mean; c.variance = a.variance; c.axis = a.axis; c.kappa = a.kappa;
    c.power = a.power; c.radius = a.radius; c.cosThetaO = a.cosTheta_o;
    c.topology = uint4(a.firstChild, a.childCount, a.slot, 0u);
    return c;
}
LTNodeCommon LT_FromBLAS(LightBLASNodeGpu b)
{
    LTNodeCommon c;
    c.mean = b.mean; c.variance = b.variance; c.axis = b.axis; c.kappa = b.kappa;
    c.power = b.power; c.radius = b.radius; c.cosThetaO = b.cosTheta_o;
    c.topology = uint4(b.firstChild, b.childCount, b.triFirst, b.triCount);
    return c;
}
LTNodeCommon LT_LoadChild(uint phase, uint nodeOffset, uint index, LT_BlasFrame frame)
{
    if (phase == 0u) return LT_FromTLAS(LT_LoadTLAS(index));
    return LT_FromBLAS(LT_LoadBLAS(nodeOffset, index, frame));
}
float LT_Importance(LT_Receiver R, LTNodeCommon C)
{
    return LT_NodeImportance(R, C.mean, C.variance, C.axis, C.kappa, C.power, C.radius, C.cosThetaO);
}
inline float LT_NodeImportance_TLAS(LightTLASNodeGpu node, LT_Receiver R) { return LT_Importance(R, LT_FromTLAS(node)); }
inline float LT_NodeImportance_BLAS(LightBLASNodeGpu node, LT_Receiver R, float3x4 worldToLocal) {
    return LT_Importance(LT_ReceiverToLocal(R, worldToLocal), LT_FromBLAS(node));
}

// Choose among the children of t (t.x first child, t.y count) by importance; returns the
// chosen child's topology and rescales the random number.
uint4 LT_PickChild(LT_Receiver R, uint phase, uint nodeOffset, LT_BlasFrame frame, uint4 t,
    inout float xi, out float p, out uint idx)
{
    const uint count = min(t.y, 4u);
    float w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;
    uint4 t0 = 0u, t1 = 0u, t2 = 0u, t3 = 0u;
    LTNodeCommon C = LT_LoadChild(phase, nodeOffset, t.x, frame);
    [loop] for (uint i = 0u; i < count; ++i)
    {
        LTNodeCommon Cn = C;
        if (i + 1u < count) Cn = LT_LoadChild(phase, nodeOffset, t.x + i + 1u, frame);
        const float wi = max(LT_Importance(R, C), 0.0);
        if (i == 0u)      { w0 = wi; t0 = C.topology; }
        else if (i == 1u) { w1 = wi; t1 = C.topology; }
        else if (i == 2u) { w2 = wi; t2 = C.topology; }
        else              { w3 = wi; t3 = C.topology; }
        C = Cn;
    }
    float xiNext;
    idx = LT_PickAndRescale(w0, w1, w2, w3, count, xi, p, xiNext);
    xi = xiNext;
    return idx == 0u ? t0 : (idx == 1u ? t1 : (idx == 2u ? t2 : t3));
}
// Probability of child childIdx among the children of t, with its topology.
float LT_ChildProbability(LT_Receiver R, uint phase, uint nodeOffset, LT_BlasFrame frame, uint4 t,
    uint childIdx, out uint4 tc)
{
    const uint count = min(t.y, 4u);
    float sum = 0.0, wc = 0.0; tc = 0u;
    LTNodeCommon C = LT_LoadChild(phase, nodeOffset, t.x, frame);
    [loop] for (uint i = 0u; i < count; ++i)
    {
        LTNodeCommon Cn = C;
        if (i + 1u < count) Cn = LT_LoadChild(phase, nodeOffset, t.x + i + 1u, frame);
        const float wi = max(LT_Importance(R, C), 0.0);
        sum += wi;
        if (i == childIdx) { wc = wi; tc = C.topology; }
        C = Cn;
    }
    return sum > 0.0f ? wc / sum : 1.0f / float(t.y);
}

// Descend the top level, then the mesh tree, while accumulating the exact branch PDF. The
// receiver moves into the mesh's space once the top level has picked it.
bool LT_Descend(LT_Receiver R, float xiT, float xiB, uint startNode, uint startSlot,
    out uint slotOut, out uint instOut, out LTLeaf leaf, out float pdfT, out float pdfB)
{
    pdfT = 1.0f; pdfB = 1.0f;
    slotOut = startSlot; instOut = LT_SENTINEL;
    leaf.triFirst = 0u; leaf.triCount = 0u; leaf.nodeIndex = 0u;
    LT_BlasFrame frame = (LT_BlasFrame)0;
    uint node = startNode;
    if (startSlot == LT_SENTINEL)
    {
        const LightTLASNodeGpu Nroot = LT_LoadTLAS(startNode);
        uint4 t = uint4(Nroot.firstChild, Nroot.childCount, Nroot.slot, 0u);
        float xi = xiT;
        [loop] for (uint iter = 0u; t.y != 0u; ++iter)
        {
            if (iter == LT_TRAIL_MAX_DEPTH) { pdfT = 0.0f; return false; }
            float p; uint idx;
            t = LT_PickChild(R, 0u, 0u, frame, t, xi, p, idx);
            pdfT *= p;
        }
        slotOut = t.z; node = 0u;
    }
    if (slotOut == LT_SENTINEL) return false;
    const LightSlotGpu S = gLT_Slot[slotOut];
    const LT_Receiver Rp = LT_ReceiverToLocal(R, S.worldToLocal);
    const uint nodeOffset = S.nodeOffset;
    instOut = S.instanceID;
    frame = LT_LoadBlasFrame(nodeOffset);
    const LightBLASNodeGpu Broot = LT_LoadBLAS(nodeOffset, node, frame);
    uint4 t = uint4(Broot.firstChild, Broot.childCount, Broot.triFirst, Broot.triCount);
    float xi = xiB;
    [loop] for (uint iter = 0u; t.y != 0u; ++iter)
    {
        if (iter == LT_TRAIL_MAX_DEPTH) { pdfB = 0.0f; return false; }
        float p; uint idx;
        const uint first = t.x;
        t = LT_PickChild(Rp, 1u, nodeOffset, frame, t, xi, p, idx);
        pdfB *= p;
        node = first + idx;
    }
    leaf.triFirst = t.z; leaf.triCount = t.w; leaf.nodeIndex = node;
    return true;
}

// Sample leaf triangles by emitted weight, with a uniform zero-power fallback.
uint LT_SampleLeafTriangle_Stratified(LTLeaf leaf, float xi, out float pdfLeaf)
{
    const uint base = leaf.triFirst;

    if (leaf.triCount <= 1u) {
        pdfLeaf = 1.0f;
        return gLT_LeafTriIndex[base];
    }

    const uint count = leaf.triCount;
    float sumW = 0.0f;
    [loop] for (uint i = 0u; i < count; ++i) {
        const uint tj = gLT_LeafTriIndex[base + i];
        sumW += max(g_EmissiveTriangles[tj].weight, 0.0f);
    }

    if (sumW <= 0.0f) {
        uint k = min((uint)floor(xi * count), count - 1u);
        pdfLeaf = 1.0f / (float)count;
        return gLT_LeafTriIndex[base + k];
    }

    const float target = xi * sumW;
    float accum = 0.0f;
    uint  sel   = count - 1u;
    float selW  = 0.0f;
    [loop] for (uint k = 0u; k < count; ++k) {
        const uint  tk = gLT_LeafTriIndex[base + k];
        const float w  = max(g_EmissiveTriangles[tk].weight, 0.0f);
        const float next = accum + w;
        if (target < next) { sel = k; selW = w; break; }
        accum = next;
    }
    pdfLeaf = selW / sumW;
    return gLT_LeafTriIndex[base + sel];
}

LT_Sample LT_SampleSubtree(LT_Receiver R, inout uint rng, uint startNode, uint startSlot, out uint slotOut)
{
    LT_Sample empty; empty.id = LT_SENTINEL; empty.inst = LT_SENTINEL; empty.pdf = 0.0f; empty.learningToken = 0u;
    slotOut = LT_SENTINEL;
    if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) != 0u) return empty;

    float xiT = RandomFloatSingle(rng);
    float xiB = RandomFloatSingle(rng);

    float pdfT, pdfB, pdfL;
    uint slot, inst; LTLeaf leaf;
    const bool descended = LT_Descend(R, xiT, xiB, startNode, startSlot, slot, inst, leaf, pdfT, pdfB);
    if (!descended || !(pdfT > 0.0f) || !(pdfB > 0.0f) || leaf.triCount == 0u || inst == LT_SENTINEL)
        return empty;

    float xiL = (leaf.triCount > 1u) ? RandomFloatSingle(rng) : 0.0f;
    uint  tri = LT_SampleLeafTriangle_Stratified(leaf, xiL, pdfL);

    LT_Sample s; s.id = tri; s.inst = inst; s.pdf = pdfT * pdfB * pdfL; s.learningToken = 0u;
    slotOut = slot;
    return s;
}
LT_Sample LT_SampleSubtree(LT_Receiver R, inout uint rng, uint startNode=0u, uint startSlot=LT_SENTINEL)
{
    uint ignored;
    return LT_SampleSubtree(R, rng, startNode, startSlot, ignored);
}

// Replay stored trails to evaluate the matching subtree PDF.
float LT_PdfSubtree(LT_Receiver R, uint triIndex, uint slot, uint startNode=0u, uint startSlot=LT_SENTINEL, uint startDepth=0u)
{
    if (triIndex == LT_SENTINEL || slot == LT_SENTINEL) return 0.0f;

    float pdf = 1.0f;
    LT_BlasFrame frame = (LT_BlasFrame)0;
    uint node = startNode;
    if (startSlot == LT_SENTINEL)
    {
        const uint2 trail = gLT_BLASBitTrail[slot];
        const LightTLASNodeGpu Nroot = LT_LoadTLAS(startNode);
        uint4 t = uint4(Nroot.firstChild, Nroot.childCount, Nroot.slot, 0u);
        uint iter = startDepth;
        [loop] while (t.y != 0u)
        {
            if (iter == LT_TRAIL_MAX_DEPTH) return 0.0f;
            const uint childIdx = LT_TrailChild(trail, iter);
            if (childIdx >= t.y) return 0.0f;
            ++iter;
            uint4 tc;
            pdf *= LT_ChildProbability(R, 0u, 0u, frame, t, childIdx, tc);
            t = tc;
        }
        node = 0u;
    }
    const LightSlotGpu S = gLT_Slot[slot];
    const LT_Receiver Rp = LT_ReceiverToLocal(R, S.worldToLocal);
    const uint nodeOffset = S.nodeOffset;
    frame = LT_LoadBlasFrame(nodeOffset);
    const LightBLASNodeGpu Broot = LT_LoadBLAS(nodeOffset, node, frame);
    uint4 t = uint4(Broot.firstChild, Broot.childCount, Broot.triFirst, Broot.triCount);
    const uint2 trail = gLT_TriBitTrail[triIndex];
    uint iter = startSlot == LT_SENTINEL ? 0u : startDepth;
    [loop] while (t.y != 0u)
    {
        if (iter == LT_TRAIL_MAX_DEPTH) return 0.0f;
        const uint childIdx = LT_TrailChild(trail, iter);
        if (childIdx >= t.y) return 0.0f;
        ++iter;
        uint4 tc;
        pdf *= LT_ChildProbability(Rp, 1u, nodeOffset, frame, t, childIdx, tc);
        t = tc;
    }
    if (t.w <= 1u) return pdf;

    const uint count    = t.w;
    const uint leafBase = t.z;
    float sumW = 0.0f;
    float myW  = 0.0f;
    [loop] for (uint j = 0u; j < count; ++j) {
        const uint tj = gLT_LeafTriIndex[leafBase + j];
        const float w = max(g_EmissiveTriangles[tj].weight, 0.0f);
        sumW += w;
        if (tj == triIndex) myW = w;
    }
    const float pdfLeaf = (sumW > 0.0f) ? (myW / sumW) : (1.0f / (float)count);
    return pdf * pdfLeaf;
}
float LT_PdfSubtree(float3 x, float3 n, uint triIndex, uint slot, uint startNode=0u, uint startSlot=LT_SENTINEL, uint startDepth=0u)
{
    return LT_PdfSubtree(LT_DiffuseReceiver(x, n), triIndex, slot, startNode, startSlot, startDepth);
}
#include "LightTreeLearning_v8.hlsli"

inline float LT_TriangleArea(uint tri, uint objID)
{
    float3 A = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].x, 1));
    float3 B = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].y, 1));
    float3 C = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].z, 1));
    return 0.5 * length(cross(B - A, C - A));
}

// Convert triangle selection probability into an area-measure PDF.
float LT_Pdf_LightTree_Area(LT_Receiver R, uint tri, uint objID, bool useLearning=true)
{
    float p_select = LT_PdfSelectTriangle(R, tri, objID, useLearning);
    float area     = max(1e-10, LT_TriangleArea(tri, objID));
    return p_select / area;
}
float LT_Pdf_LightTree_Area(float3 x, float3 n, uint tri, uint objID, bool useLearning=true)
{
    return LT_Pdf_LightTree_Area(LT_DiffuseReceiver(x, n), tri, objID, useLearning);
}

struct LT_LightSampleResult
{
    float3 position;
    float3 normal;
    float3 emission;
    float  pdfSolidAngle;
    uint   triIndex;
    uint   objID;
};

// Convert a selected triangle into a world-space point and solid-angle PDF.
LT_LightSampleResult LT_SamplePointOnLightTree(float3 refPos, LT_Sample treeSample, inout uint rng)
{
    LT_LightSampleResult result = (LT_LightSampleResult)0;
    result.triIndex = treeSample.id;
    result.objID    = treeSample.inst;
    if (treeSample.id == LT_SENTINEL || treeSample.inst == LT_SENTINEL || !(treeSample.pdf > 0.0f)) return result;

    LightTriangle triData = g_EmissiveTriangles[result.triIndex];
    result.emission = triData.emission * GLOBAL_EMISSION_STRENGTH;

    float3x4 worldMat = instanceProps[result.objID].objectToWorld;
    float3 v0 = mul(worldMat, float4(triData.x, 1.0));
    float3 v1 = mul(worldMat, float4(triData.y, 1.0));
    float3 v2 = mul(worldMat, float4(triData.z, 1.0));

    float r1 = RandomFloatSingle(rng);
    float r2 = RandomFloatSingle(rng);
    float sqrtR1 = sqrt(r1);
    float u = 1.0f - sqrtR1;
    float v = r2 * sqrtR1;

    result.position = (1.0f - u - v) * v0 + u * v1 + v * v2;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 crossP = cross(e1, e2);
    float area2 = length(crossP);

    float3 normalW = mul((float3x3)instanceProps[result.objID].objectToWorldNormal,
                        cross(triData.y - triData.x, triData.z - triData.x));
    result.normal = normalW * rsqrt(max(dot(normalW, normalW), 1e-20f));
    float area = 0.5f * area2;

    float3 toLight = result.position - refPos;
    float distSq   = dot(toLight, toLight);
    float dist     = sqrt(distSq);

    float cosLight = max(dot(result.normal, -toLight / dist), 0.0f);

    float pdfArea = treeSample.pdf / max(area, 1e-10f);

    // Convert the selected area PDF to solid angle.
    if (cosLight > 1e-6f) {
        result.pdfSolidAngle = pdfArea * distSq / cosLight;
    } else {
        result.pdfSolidAngle = 0.0f;
    }

    return result;
}

LT_LightSampleResult LT_SamplePointOnLight(float3 refPos, float3 refNormal, inout uint rng)
{
    const LT_Sample treeSample = LT_SampleLight(LT_DiffuseReceiver(refPos, refNormal), rng);
    return LT_SamplePointOnLightTree(refPos, treeSample, rng);
}
