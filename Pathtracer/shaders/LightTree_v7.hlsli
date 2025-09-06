#ifndef LT_HAVE_LEAF_ALIAS
#define LT_HAVE_LEAF_ALIAS 1
#endif

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


// ============================================================================
// CONTY–KULLA NODE IMPORTANCE
// ============================================================================
inline float LT_NodeImportance_Common(float3 x, float3 n,
                                      float3 bmin, float3 bmax,
                                      float3 axis, float theta_o, float theta_e,
                                      float power, bool isGlobalLight, float offset_threshold)
{
    float3 c  = 0.5 * (bmin + bmax);
    float3 v  = x - c;
    float3 dir_to_point = normalize(v);

    float  R  = length(0.5 * (bmax - bmin));
    float  d  = max(length(v), R);
    float  theta_u = LT_SafeAsin(saturate(R / d));

    float cosAxis = dot(axis, dir_to_point);
    //cosAxis = abs(cosAxis); //Not double sided!
    float theta   = LT_SafeAcos(cosAxis);

    float theta_prime = theta - theta_o - theta_u;
    if (theta_prime >= theta_e) return 0.0;

    float gate = 1.0f;

    float ang   = max(0.0f, cos(max(theta_prime, 0.0f)));
    float invd2 = isGlobalLight ? 1.0f : (1.0f / (d * d));
    return power * ang * invd2 * gate;
}


inline float LT_NodeImportance_TLAS(LightTLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.theta_o, n.theta_e, n.power, /*isGlobal*/false, 0.0001f);
}
inline float LT_NodeImportance_BLAS(LightBLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.theta_o, n.theta_e, n.power, false, 0.0001f);
}

inline float LT_BranchProb(float wL, float wR, float PL, float PR)
{
    // Regular case
    float sum = wL + wR;
    if (sum > 1e-12) return wL / sum;

    // Tie: fall back to subtree energy
    float PE = PL + PR;
    return (PE > 0.0) ? (PL / PE) : 0.5;
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
        float sum = wL + wR;
        float pL = LT_BranchProb(wL, wR, L.power, R.power);

        bool chooseL = (RandomFloatSingle(rng) < pL);
        pdfTLAS *= chooseL ? pL : (1.0 - pL);
        node    = chooseL ? N.left : N.right;
    }
}

struct LTLeaf { uint triFirst; uint triCount; uint nodeIndex; };

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
        float sum = wL + wR;
        float pL = LT_BranchProb(wL, wR, L.power, Rc.power);

        bool chooseL = (RandomFloatSingle(rng) < pL);
        pdfBLAS *= chooseL ? pL : (1.0 - pL);
        node    = chooseL ? N.left : N.right;
    }
}

// ============================================================================
// LEAF TRIANGLE SAMPLING
// ============================================================================
uint LT_SampleLeafTriangle(uint blasIndex, LTLeaf leaf, inout uint rng, out float pdfLeaf)
{
    BlasRangeGpu R = gLT_Range[blasIndex];
    uint base  = R.triIndexOffset + leaf.triFirst;
    uint count = max(leaf.triCount, 1u);
    float sumW = max(gLT_BLAS[R.nodeOffset + leaf.nodeIndex].power, 1e-20);

#if LT_HAVE_LEAF_ALIAS
    float u0 = RandomFloatSingle(rng);
    float u1 = RandomFloatSingle(rng);
    uint  i0 = min((uint)floor(u0 * count), count - 1u);
    uint  idx= base + i0;

    float q = gLT_LeafAliasProb[idx];
    uint  a = gLT_LeafAliasIdx [idx];
    uint  local = (u1 < q) ? i0 : a;
    uint  triIndex = gLT_LeafTriIndex[base + local];

    float w = max(g_EmissiveTriangles[triIndex].weight, 0.0);
    pdfLeaf = w / sumW;
    return triIndex;
#else
    float u = RandomFloatSingle(rng) * sumW;
    float acc = 0.0;
    [loop] for (uint k=0; k<count; ++k) {
        uint  triIndex = gLT_LeafTriIndex[base + k];
        float w = max(g_EmissiveTriangles[triIndex].weight, 0.0);
        acc += w;
        if (u <= acc || k == count-1u) {
            pdfLeaf = w / sumW;
            return triIndex;
        }
    }
    pdfLeaf = 1.0 / count;
    return gLT_LeafTriIndex[base];
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
    uint   tri  = LT_SampleLeafTriangle(blas, leaf, rng, pdfL);
    LT_Sample s; s.id = tri; s.pdf = pdfT * pdfB * pdfL;
    return s;
}


// ============================================================================
// PDF of different sampling strat
// ============================================================================
float LT_PdfSelectTriangle(float3 x, float3 n, uint triIndex)
{
    // Which BLAS and which local offset within its leafTriList?
    uint blas   = gLT_TriToBLAS[triIndex];
    if (blas == LT_SENTINEL) return 0.0;   // not in the tree (shouldn't happen)
    uint item   = gLT_BLASToItem[blas];
    uint localT = gLT_TriToLeafOffset[triIndex];

    // TLAS path prob to this BLAS
    float pdfTLAS = 1.0;
    uint tnode = 0;
    [loop] for (;;)
    {
        LightTLASNodeGpu N = gLT_TLAS[tnode];
        if (N.blasIndex != LT_SENTINEL)
            break; // at TLAS leaf (exactly one BLAS under it)

        LightTLASNodeGpu L = gLT_TLAS[N.left];
        LightTLASNodeGpu R = gLT_TLAS[N.right];

        float wL = max(LT_NodeImportance_TLAS(L, x, n), 0.0);
        float wR = max(LT_NodeImportance_TLAS(R, x, n), 0.0);
        float pL = LT_BranchProb(wL, wR, L.power, R.power);

        bool inL = (item >= L.itemFirst) && (item < (L.itemFirst + L.itemCount));
        pdfTLAS *= inL ? pL : (1.0 - pL);
        tnode    = inL ? N.left : N.right;
    }

    // BLAS path prob to the specific leaf that contains triIndex
    BlasRangeGpu Rng = gLT_Range[blas];
    float pdfBLAS = 1.0;
    uint bnode = 0;

    [loop] for (;;)
    {
        LightBLASNodeGpu N = gLT_BLAS[Rng.nodeOffset + bnode];

        // leaf?
        if (N.left == LT_SENTINEL && N.right == LT_SENTINEL)
        {
            // within-leaf probability (alias or linear give same pdf)
            float sumW = max(gLT_BLAS[Rng.nodeOffset + bnode].power, 1e-20);
            float wTri = max(g_EmissiveTriangles[triIndex].weight, 0.0);
            float pdfLeaf = (sumW > 0.0) ? (wTri / sumW) : 0.0;
            return pdfTLAS * pdfBLAS * pdfLeaf;
        }

        LightBLASNodeGpu L = gLT_BLAS[Rng.nodeOffset + N.left];
        LightBLASNodeGpu Rc= gLT_BLAS[Rng.nodeOffset + N.right];

        float wL = max(LT_NodeImportance_BLAS(L, x, n), 0.0);
        float wR = max(LT_NodeImportance_BLAS(Rc, x, n), 0.0);
        float pL = LT_BranchProb(wL, wR, L.power, Rc.power);

        // membership via local triangle ranges inside this BLAS
        bool inL = (localT >= L.triFirst) && (localT < (L.triFirst + L.triCount));
        pdfBLAS *= inL ? pL : (1.0 - pL);
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