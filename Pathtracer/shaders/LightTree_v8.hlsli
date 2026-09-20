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

// Bound each node by receiver, orientation, distance, and emitted power.
inline float LT_NodeImportance_Common(
    float3 x, float3 n,
    float3 bmin, float3 bmax,
    float3 axis, float cosTheta_o, float sinTheta_o,
    float power)
{

    const float3 maxCorner = float3(
        (n.x >= 0.0f) ? bmax.x : bmin.x,
        (n.y >= 0.0f) ? bmax.y : bmin.y,
        (n.z >= 0.0f) ? bmax.z : bmin.z);
    if (dot(maxCorner - x, n) <= 0.0f) return 0.0f;

    const float3 c        = 0.5 * (bmin + bmax);
    const float3 e        = 0.5 * (bmax - bmin);
    const float  R        = sqrt(dot(e, e));
    const float3 toCenter = c - x;
    const float  d2       = dot(toCenter, toCenter);
    const float  distSq   = max(max(d2, R * R), 1e-12);

    if(d2<=R*R) return max(0.0f,power)/distSq;
    const float d=sqrt(d2);
    const float3 toCenterN=toCenter/max(d,1e-20f);
    const float sinThetaU=min(R/d,1.0f);
    const float cosThetaU=sqrt(max(1.0f-sinThetaU*sinThetaU,0.0f));

    const float ci = dot(n, toCenterN);
    float cos_i_prime;
    if (ci >= cosThetaU) {
        cos_i_prime = 1.0f;
    } else {
        const float si = sqrt(max(1.0f - ci * ci, 0.0f));
        cos_i_prime = ci * cosThetaU + si * sinThetaU;
    }

    const float cosTheta = clamp(dot(axis, -toCenterN), -1.0f, 1.0f);
    const float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));

    const float cos_T = cosTheta_o * cosThetaU - sinTheta_o * sinThetaU;
    const float sin_T = sinTheta_o * cosThetaU + cosTheta_o * sinThetaU;

    float cos_theta_prime;
    if (sin_T <= 0.0f) {

        cos_theta_prime = 1.0f;
    } else if (cosTheta >= cos_T) {

        cos_theta_prime = 1.0f;
    } else {

        cos_theta_prime = max(cosTheta * cos_T + sinTheta * sin_T, 0.0f);
    }

    return max(0.0f,cos_i_prime) * max(0.0f,power) * cos_theta_prime / distSq;
}

float3 LT_LocalReceiverNormal(float3x4 worldToLocal,float3 normal) {
    float3 a=worldToLocal[0].xyz,b=worldToLocal[1].xyz,c=worldToLocal[2].xyz;
    float3 cof0=cross(b,c),cof1=cross(c,a),cof2=cross(a,b);
    float3 pullback=float3(dot(cof0,normal),dot(cof1,normal),dot(cof2,normal));
    return normalize(pullback*(dot(a,cof0)<0.0f?-1.0f:1.0f));
}
inline float LT_NodeImportance_TLAS(LightTLASNodeGpu node,float3 x,float3 norm) {
    return LT_NodeImportance_Common(x,norm,node.bmin,node.bmax,node.axis,node.cosTheta_o,node.sinTheta_o,node.power);
}
inline float LT_NodeImportance_BLAS(LightBLASNodeGpu node,float3 x,float3 norm,float3x4 worldToLocal) {
    float3 localX=mul(worldToLocal,float4(x,1));
    float3 localN=LT_LocalReceiverNormal(worldToLocal,norm);
    return LT_NodeImportance_Common(localX,localN,node.bmin,node.bmax,node.axis,node.cosTheta_o,node.sinTheta_o,node.power);
}

struct LTNodeCommon
{
    float3 bmin; float3 bmax; float3 axis;
    float cosTheta_o; float sinTheta_o; float power;
    uint4 topology;
};
LTNodeCommon LT_LoadChild(uint phase, uint nodeOffset, uint index, LT_BlasFrame frame)
{
    LTNodeCommon c;
    if (phase == 0u)
    {
        const LightTLASNodeGpu a = LT_LoadTLAS(index);
        c.bmin = a.bmin; c.bmax = a.bmax; c.axis = a.axis;
        c.cosTheta_o = a.cosTheta_o; c.sinTheta_o = a.sinTheta_o; c.power = a.power;
        c.topology = uint4(a.firstChild, a.childCount, a.slot, 0u);
    }
    else
    {
        const LightBLASNodeGpu b = LT_LoadBLAS(nodeOffset, index, frame);
        c.bmin = b.bmin; c.bmax = b.bmax; c.axis = b.axis;
        c.cosTheta_o = b.cosTheta_o; c.sinTheta_o = b.sinTheta_o; c.power = b.power;
        c.topology = uint4(b.firstChild, b.childCount, b.triFirst, b.triCount);
    }
    return c;
}

// Descend TLAS then BLAS while accumulating the exact branch PDF.
bool LT_Descend(float3 x, float3 n, float xiT, float xiB, uint startNode, uint startSlot,
    out uint slotOut, out uint instOut, out LTLeaf leaf, out float pdfT, out float pdfB)
{
    pdfT = 1.0f; pdfB = 1.0f;
    slotOut = startSlot; instOut = LT_SENTINEL;
    leaf.triFirst = 0u; leaf.triCount = 0u; leaf.nodeIndex = 0u;
    uint   phase = startSlot == LT_SENTINEL ? 0u : 1u;
    bool   enter = true;
    uint   iter = 0u;
    float3 xP = x, nP = n;
    uint   nodeOffset = 0u;
    LT_BlasFrame frame = (LT_BlasFrame)0;
    uint   node = 0u;
    float  xi = xiT;
    uint4  t = 0u;
    [loop] for (;;)
    {
        if (enter)
        {
            enter = false; iter = 0u;
            if (phase == 0u)
            {
                const LightTLASNodeGpu Nroot = LT_LoadTLAS(startNode);
                t = uint4(Nroot.firstChild, Nroot.childCount, Nroot.slot, 0u);
                xi = xiT;
            }
            else
            {
                if (slotOut == LT_SENTINEL) return false;

                const LightSlotGpu S = gLT_Slot[slotOut];
                const float3x4 W2L = S.worldToLocal;
                xP = mul(W2L, float4(x, 1.0));
                nP = LT_LocalReceiverNormal(W2L, n);
                nodeOffset = S.nodeOffset; instOut = S.instanceID;
                frame = LT_LoadBlasFrame(nodeOffset);
                node = startSlot == LT_SENTINEL ? 0u : startNode;
                const LightBLASNodeGpu Nroot = LT_LoadBLAS(nodeOffset, node, frame);
                t = uint4(Nroot.firstChild, Nroot.childCount, Nroot.triFirst, Nroot.triCount);
                xi = xiB;
            }
        }
        if (t.y == 0u)
        {

            if (phase == 1u) { leaf.triFirst = t.z; leaf.triCount = t.w; leaf.nodeIndex = node; return true; }
            slotOut = t.z; phase = 1u; enter = true; continue;
        }

        if (iter == LT_TRAIL_MAX_DEPTH) { if (phase == 0u) pdfT = 0.0f; else pdfB = 0.0f; return false; }
        ++iter;

        const uint count = min(t.y, 4u);
        float w0 = 0.0, w1 = 0.0, w2 = 0.0, w3 = 0.0;
        uint4 t0 = 0u, t1 = 0u, t2 = 0u, t3 = 0u;
        LTNodeCommon C = LT_LoadChild(phase, nodeOffset, t.x, frame);
        [loop] for (uint i = 0u; i < count; ++i)
        {
            LTNodeCommon Cn = C;
            if (i + 1u < count) Cn = LT_LoadChild(phase, nodeOffset, t.x + i + 1u, frame);
            const float wi = max(LT_NodeImportance_Common(xP, nP, C.bmin, C.bmax, C.axis, C.cosTheta_o, C.sinTheta_o, C.power), 0.0);
            if (i == 0u)      { w0 = wi; t0 = C.topology; }
            else if (i == 1u) { w1 = wi; t1 = C.topology; }
            else if (i == 2u) { w2 = wi; t2 = C.topology; }
            else              { w3 = wi; t3 = C.topology; }
            C = Cn;
        }

        float p, xi_next;
        const uint idx = LT_PickAndRescale(w0, w1, w2, w3, count, xi, p, xi_next);
        if (phase == 0u) pdfT *= p; else pdfB *= p;
        node = t.x + idx;
        t = idx == 0u ? t0 : (idx == 1u ? t1 : (idx == 2u ? t2 : t3));
        xi = xi_next;
    }
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

LT_Sample LT_SampleSubtree(float3 worldPos, float3 worldNormal, inout uint rng, uint startNode, uint startSlot, out uint slotOut)
{
    LT_Sample empty; empty.id = LT_SENTINEL; empty.inst = LT_SENTINEL; empty.pdf = 0.0f; empty.learningToken = 0u;
    slotOut = LT_SENTINEL;
    if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) != 0u) return empty;

    float xiT = RandomFloatSingle(rng);
    float xiB = RandomFloatSingle(rng);

    float pdfT, pdfB, pdfL;
    uint slot, inst; LTLeaf leaf;
    const bool descended = LT_Descend(worldPos, worldNormal, xiT, xiB, startNode, startSlot, slot, inst, leaf, pdfT, pdfB);
    if (!descended || !(pdfT > 0.0f) || !(pdfB > 0.0f) || leaf.triCount == 0u || inst == LT_SENTINEL)
        return empty;

    float xiL = (leaf.triCount > 1u) ? RandomFloatSingle(rng) : 0.0f;
    uint  tri = LT_SampleLeafTriangle_Stratified(leaf, xiL, pdfL);

    LT_Sample s; s.id = tri; s.inst = inst; s.pdf = pdfT * pdfB * pdfL; s.learningToken = 0u;
    slotOut = slot;
    return s;
}
LT_Sample LT_SampleSubtree(float3 worldPos, float3 worldNormal, inout uint rng, uint startNode=0u, uint startSlot=LT_SENTINEL)
{
    uint ignored;
    return LT_SampleSubtree(worldPos, worldNormal, rng, startNode, startSlot, ignored);
}

// Replay stored trails to evaluate the matching subtree PDF.
float LT_PdfSubtree(float3 x, float3 n, uint triIndex, uint slot, uint startNode=0u, uint startSlot=LT_SENTINEL, uint startDepth=0u)
{
    if (triIndex == LT_SENTINEL || slot == LT_SENTINEL) return 0.0f;

    const uint2 slotTrail = gLT_BLASBitTrail[slot];
    const uint2 triTrail  = gLT_TriBitTrail[triIndex];

    float  pdfTLAS = 1.0f, pdfBLAS = 1.0f;
    uint   phase = startSlot == LT_SENTINEL ? 0u : 1u;
    bool   enter = true;
    uint   iter = 0u;
    float3 xP = x, nP = n;
    uint   nodeOffset = 0u;
    LT_BlasFrame frame = (LT_BlasFrame)0;
    uint2  trail = 0u;
    uint4  t = 0u;
    [loop] for (;;)
    {
        if (enter)
        {
            enter = false;
            if (phase == 0u)
            {
                const LightTLASNodeGpu Nroot = LT_LoadTLAS(startNode);
                t = uint4(Nroot.firstChild, Nroot.childCount, Nroot.slot, 0u);
                trail = slotTrail; iter = startDepth;
            }
            else
            {
                const LightSlotGpu S = gLT_Slot[slot];
                const float3x4 W2L = S.worldToLocal;
                xP = mul(W2L, float4(x, 1.0));
                nP = LT_LocalReceiverNormal(W2L, n);
                nodeOffset = S.nodeOffset;
                frame = LT_LoadBlasFrame(nodeOffset);
                const LightBLASNodeGpu Nroot = LT_LoadBLAS(nodeOffset, startSlot == LT_SENTINEL ? 0u : startNode, frame);
                t = uint4(Nroot.firstChild, Nroot.childCount, Nroot.triFirst, Nroot.triCount);
                trail = triTrail; iter = startSlot == LT_SENTINEL ? 0u : startDepth;
            }
        }
        if (t.y == 0u)
        {
            if (phase == 0u) { phase = 1u; enter = true; continue; }

            if (t.w <= 1u) {
                return pdfTLAS * pdfBLAS;
            }

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
            return pdfTLAS * pdfBLAS * pdfLeaf;
        }

        if (iter == LT_TRAIL_MAX_DEPTH) return 0.0f;
        const uint childIdx = LT_TrailChild(trail, iter);
        if (childIdx >= t.y) return 0.0f;
        ++iter;

        const uint count = min(t.y, 4u);
        float sum = 0.0; float wc = 0.0; uint4 tc = 0u;
        LTNodeCommon C = LT_LoadChild(phase, nodeOffset, t.x, frame);
        [loop] for (uint i = 0u; i < count; ++i)
        {
            LTNodeCommon Cn = C;
            if (i + 1u < count) Cn = LT_LoadChild(phase, nodeOffset, t.x + i + 1u, frame);
            const float wi = max(LT_NodeImportance_Common(xP, nP, C.bmin, C.bmax, C.axis, C.cosTheta_o, C.sinTheta_o, C.power), 0.0);
            sum += wi;
            if (i == childIdx) { wc = wi; tc = C.topology; }
            C = Cn;
        }

        const float p = (sum > 0.0f) ? (wc / sum) : (1.0f / float(t.y));
        if (phase == 0u) pdfTLAS *= p; else pdfBLAS *= p;
        t = tc;
    }
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
float LT_Pdf_LightTree_Area(float3 x, float3 n, uint tri, uint objID, bool useLearning=true)
{
    float p_select = LT_PdfSelectTriangle(x, n, tri, objID, useLearning);
    float area     = max(1e-10, LT_TriangleArea(tri, objID));
    return p_select / area;
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
    const LT_Sample treeSample = LT_SampleLight(refPos, refNormal, rng);
    return LT_SamplePointOnLightTree(refPos, treeSample, rng);
}
