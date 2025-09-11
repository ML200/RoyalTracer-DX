#ifndef LT_HAVE_LEAF_ALIAS
#define LT_HAVE_LEAF_ALIAS 0
#endif

// ---------------------------------------------------------------------------

Buffer<uint> gLT_TriToBLAS       : register(t16);
Buffer<uint> gLT_TriToLeafOffset : register(t17);
Buffer<uint> gLT_BLASToItem      : register(t18);

// ============================================================================
// CONSTANTS / HELPERS
// ============================================================================
static const uint  LT_SENTINEL = 0xFFFFFFFFu;
static const float LT_PI = 3.14159265358979323846;

inline float3 LT_AabbCenter(float3 mn, float3 mx) { return 0.5*(mn+mx); }
inline float  LT_AabbRadius(float3 mn, float3 mx) { float3 e = 0.5*(mx-mn); return length(e); }
inline float  LT_SafeAcos(float x){ return acos(clamp(x, -1.0, 1.0)); }
inline float  LT_SafeAsin(float x){ return asin(clamp(x, -1.0, 1.0)); }

struct LTLeaf { uint triFirst; uint triCount; uint nodeIndex; };

// ============================================================================
// CONTY–KULLA NODE IMPORTANCE
// ============================================================================
inline float LT_NodeImportance_Common(
    float3 x, float3 n,
    float3 bmin, float3 bmax,
    float3 axis, float theta_o, float theta_e,
    float power, bool isGlobalLight, float /*offset_threshold*/)
{
    const float3 c = 0.5 * (bmin + bmax);
    const float3 e = 0.5 * (bmax - bmin);
    const float  R = length(e);

    float3 v  = x - c;
    float  d0 = length(v);
    float3 dir_to_point = (d0 > 0.0) ? (v / d0) : float3(0,0,1);

    float theta_u = LT_SafeAsin(saturate(R / max(d0, 1e-3)));

    float theta   = LT_SafeAcos(dot(axis, dir_to_point));
    float theta_p = max(theta - theta_o - theta_u, 0.0);
    if (theta_p >= theta_e) return 0.0;
    float orientTerm = cos(theta_p);

    float ci    = dot(-dir_to_point, n);
    float sin_u = sin(theta_u);
    if (ci <= -sin_u) return 0.0;

    float theta_i_p = max(LT_SafeAcos(ci) - theta_u, 0.0);
    float incidTerm = max(cos(theta_i_p), 0.0);

    float d    = max(d0, 0.5 * R);
    float invd = isGlobalLight ? 1.0 : rcp(d);
    float geom = isGlobalLight ? 1.0 : (invd * invd);

    return power * orientTerm * incidTerm * geom;
}

inline float LT_NodeImportance_TLAS(LightTLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.theta_o, n.theta_e, n.power, false, 0.01f);
}
inline float LT_NodeImportance_BLAS(LightBLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.theta_o, n.theta_e, n.power, false, 0.01f);
}

inline float LT_BranchProb(float IL, float IR)
{
    float sum = IL + IR;
    float prob = (sum > 0.0f) ? (IL / sum) : 0.5f;
    return clamp(prob, 0.1f, 0.9f);
}

// ============================================================================
// STOCHASTIC DESCENT
// ============================================================================
uint LT_DescendTLAS(float3 x, float3 n, inout uint rng, out float pdfTLAS)
{
    pdfTLAS = 1.0;
    uint node = 0;
    [loop] for (;;)
    {
        LightTLASNodeGpu N = gLT_TLAS[node];
        if (N.blasIndex != LT_SENTINEL) return N.blasIndex;

        LightTLASNodeGpu L = gLT_TLAS[N.left];
        LightTLASNodeGpu R = gLT_TLAS[N.right];

        float wL = max(LT_NodeImportance_TLAS(L,x,n), 0.0);
        float wR = max(LT_NodeImportance_TLAS(R,x,n), 0.0);
        float pL = LT_BranchProb(wL, wR);

        bool chooseL = (RandomFloatSingle(rng) < pL);
        pdfTLAS *= chooseL ? pL : (1.0 - pL);
        node    = chooseL ? N.left : N.right;
    }
}

LTLeaf LT_DescendBLAS(float3 x, float3 n, uint blasIndex, inout uint rng, out float pdfBLAS)
{
    pdfBLAS = 1.0;
    BlasRangeGpu R = gLT_Range[blasIndex];
    uint node = 0;

    [loop] for (;;)
    {
        LightBLASNodeGpu N = gLT_BLAS[R.nodeOffset + node];
        if (N.left == LT_SENTINEL && N.right == LT_SENTINEL)
        {
            LTLeaf outL; outL.triFirst = N.triFirst; outL.triCount = N.triCount; outL.nodeIndex = node;
            return outL;
        }

        LightBLASNodeGpu L = gLT_BLAS[R.nodeOffset + N.left];
        LightBLASNodeGpu Rc= gLT_BLAS[R.nodeOffset + N.right];

        float wL = max(LT_NodeImportance_BLAS(L,x,n), 0.0);
        float wR = max(LT_NodeImportance_BLAS(Rc,x,n), 0.0);
        float pL = LT_BranchProb(wL, wR);

        bool chooseL = (RandomFloatSingle(rng) < pL);
        pdfBLAS *= chooseL ? pL : (1.0 - pL);
        node    = chooseL ? N.left : N.right;
    }
}

// ============================================================================
// LEAF TRIANGLE SAMPLING
// ============================================================================
inline float LT_LeafLocalWeight(float3 x, float3 n, uint objID, uint triIndex)
{
    const float eps = 1e-3f;

    float3 A = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[triIndex].x, 1)).xyz;
    float3 B = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[triIndex].y, 1)).xyz;
    float3 C = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[triIndex].z, 1)).xyz;

    float3 triCentroid = (A + B + C) * (1.0f / 3.0f);
    float3 Nlight   = normalize(cross(B - A, C - A));
    float3 v        = x - triCentroid;
    float  d        = max(length(v), eps);
    float3 wiL      = v / d;

    float w0 = max(g_EmissiveTriangles[triIndex].weight, 0.0f);

    float cosLight  = saturate(dot(Nlight, wiL));
    float cosSurf   = saturate(dot(n, -wiL));

    return w0 * cosLight * cosSurf * rcp(d);
}

uint LT_SampleLeafTriangle(float3 x, float3 n, uint blasIndex, LTLeaf leaf, inout uint rng, out float pdfLeaf)
{
    BlasRangeGpu R   = gLT_Range[blasIndex];
    uint base        = R.triIndexOffset + leaf.triFirst;
    uint count       = max(leaf.triCount, 1u);

#if LT_HAVE_LEAF_ALIAS
    float sumW = max(gLT_BLAS[R.nodeOffset + leaf.nodeIndex].power, 1e-20);

    float u0 = RandomFloatSingle(rng);
    float u1 = RandomFloatSingle(rng);
    uint  i0 = min((uint)floor(u0 * count), count - 1u);
    uint  idx= base + i0;

    float q = gLT_LeafAliasProb[idx];
    uint  a = gLT_LeafAliasIdx [idx];
    uint  local    = (u1 < q) ? i0 : a;
    uint  triIndex = gLT_LeafTriIndex[base + local];

    float w = max(g_EmissiveTriangles[triIndex].weight, 0.0f);
    pdfLeaf = w / sumW;
    return triIndex;

#else
    uint objID = gLT_BLASToItem[blasIndex];

    float sumW = 0.0f;
    [loop] for (uint k = 0; k < count; ++k)
    {
        uint tri = gLT_LeafTriIndex[base + k];
        sumW += max(LT_LeafLocalWeight(x, n, objID, tri), 0.0f);
    }

    if (sumW <= 0.0f)
    {
        uint  k       = min((uint)floor(RandomFloatSingle(rng) * count), count - 1u);
        uint  triZero = gLT_LeafTriIndex[base + k];
        pdfLeaf = 1.0f / count;
        return triZero;
    }

    float  u       = RandomFloatSingle(rng) * sumW;
    float  acc     = 0.0f;
    uint   triPick = gLT_LeafTriIndex[base + (count - 1u)];
    float  wPick   = 0.0f;

    [loop] for (uint k = 0; k < count; ++k)
    {
        uint  tri = gLT_LeafTriIndex[base + k];
        float w   = max(LT_LeafLocalWeight(x, n, objID, tri), 0.0f);
        acc      += w;

        if (u <= acc)
        {
            triPick = tri;
            wPick   = w;
            break;
        }
    }

    pdfLeaf = (wPick > 0.0f) ? (wPick / sumW) : (1.0f / count);
    return triPick;
#endif
}

// ============================================================================
// TOP-LEVEL SAMPLER
// ============================================================================
LT_Sample LT_SampleLight(float3 worldPos, float3 worldNormal, inout uint rng)
{
    float pdfT, pdfB, pdfL;
    uint   blas = LT_DescendTLAS(worldPos, worldNormal, rng, pdfT);
    LTLeaf leaf = LT_DescendBLAS(worldPos, worldNormal, blas, rng, pdfB);
    uint   tri  = LT_SampleLeafTriangle(worldPos, worldNormal, blas, leaf, rng, pdfL);
    LT_Sample s; s.id = tri; s.pdf = pdfT * pdfB * pdfL;
    return s;
}

// ============================================================================
// PDF
// ============================================================================
float LT_PdfSelectTriangle(float3 x, float3 n, uint triIndex)
{
    uint blas = gLT_TriToBLAS[triIndex];
    if (blas == LT_SENTINEL) return 0.0f;

    uint item = gLT_BLASToItem[blas];

    float pdfTLAS = 1.0f;
    uint  tnode   = 0;

    [loop] for (;;)
    {
        LightTLASNodeGpu N = gLT_TLAS[tnode];
        if (N.blasIndex != LT_SENTINEL)
            break;

        LightTLASNodeGpu L = gLT_TLAS[N.left];
        LightTLASNodeGpu R = gLT_TLAS[N.right];

        float wL = max(LT_NodeImportance_TLAS(L, x, n), 0.0f);
        float wR = max(LT_NodeImportance_TLAS(R, x, n), 0.0f);
        float pL = LT_BranchProb(wL, wR);

        bool inL = (item >= L.itemFirst) && (item < (L.itemFirst + L.itemCount));
        pdfTLAS *= inL ? pL : (1.0f - pL);
        tnode    = inL ? N.left : N.right;
    }

    BlasRangeGpu Rng = gLT_Range[blas];
    float pdfBLAS = 1.0f;
    uint  bnode   = 0;

    uint localIdx = gLT_TriToLeafOffset[triIndex];

    [loop] for (;;)
    {
        LightBLASNodeGpu N = gLT_BLAS[Rng.nodeOffset + bnode];

        if (N.left == LT_SENTINEL && N.right == LT_SENTINEL)
        {
#if LT_HAVE_LEAF_ALIAS
            float sumW    = max(gLT_BLAS[Rng.nodeOffset + bnode].power, 1e-20f);
            float wTri    = max(g_EmissiveTriangles[triIndex].weight, 0.0f);
            float pdfLeaf = (sumW > 0.0f) ? (wTri / sumW) : 0.0f;
            return pdfTLAS * pdfBLAS * pdfLeaf;
#else
            uint  base  = Rng.triIndexOffset + N.triFirst;
            uint  count = max(N.triCount, 1u);
            uint  objID = gLT_BLASToItem[blas];

            float sumW    = 0.0f;
            float wTriSel = 0.0f;

            [loop] for (uint k = 0; k < count; ++k)
            {
                uint  t = gLT_LeafTriIndex[base + k];
                float w = max(LT_LeafLocalWeight(x, n, objID, t), 0.0f);
                sumW += w;
                if (t == triIndex) wTriSel = w;
            }

            float pdfLeaf = (sumW > 0.0f) ? (wTriSel / sumW) : (1.0f / count);
            return pdfTLAS * pdfBLAS * pdfLeaf;
#endif
        }

        LightBLASNodeGpu NL = gLT_BLAS[Rng.nodeOffset + N.left];
        LightBLASNodeGpu NR = gLT_BLAS[Rng.nodeOffset + N.right];

        float wL = max(LT_NodeImportance_BLAS(NL, x, n), 0.0f);
        float wR = max(LT_NodeImportance_BLAS(NR, x, n), 0.0f);
        float pL = LT_BranchProb(wL, wR);

        bool inL = (localIdx >= NL.triFirst) && (localIdx < (NL.triFirst + NL.triCount));
        pdfBLAS *= inL ? pL : (1.0f - pL);
        bnode    = inL ? N.left : N.right;
    }
}

inline float LT_TriangleArea(uint tri, uint objID)
{
    float3 A = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].x, 1)).xyz;
    float3 B = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].y, 1)).xyz;
    float3 C = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].z, 1)).xyz;
    return 0.5 * length(cross(B - A, C - A));
}

inline float LT_Pdf_LightTree_Area(float3 x, float3 n, uint tri, uint objID)
{
    float p_select = LT_PdfSelectTriangle(x, n, tri);
    float area     = max(1e-20, LT_TriangleArea(tri, objID));
    return p_select / area;
}
