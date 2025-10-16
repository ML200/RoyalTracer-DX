#ifndef LT_LEAF_SAMPLER_MODE
#define LT_LEAF_SAMPLER_MODE 1
#endif

Buffer<uint> gLT_TriToBLAS       : register(t16);
Buffer<uint> gLT_TriToLeafOffset : register(t17);
Buffer<uint> gLT_BLASToItem      : register(t18);

// CONSTANTS / HELPERS
static const uint  LT_SENTINEL = 0xFFFFFFFFu;
static const float LT_PI = 3.14159265358979323846;

inline float3 LT_AabbCenter(float3 mn, float3 mx) { return 0.5*(mn+mx); }
inline float  LT_AabbRadius(float3 mn, float3 mx) { float3 e = 0.5*(mx-mn); return length(e); }
inline float  LT_SafeAcos(float x){ return acos(clamp(x, -1.0, 1.0)); }
inline float  LT_SafeAsin(float x){ return asin(clamp(x, -1.0, 1.0)); }

// Categorical pick (for 4-ary tree)
inline uint LT_CategoricalPick(in float w[4], uint n, float u, out float pChosen)
{
    float sum = 0.0;
    [unroll] for (uint i=0;i<4;i++) if (i<n) sum += max(w[i], 0.0);
    if (sum > 0.0) {
        float acc = 0.0;
        [unroll] for (uint i=0;i<4;i++){
            if (i>=n) break;
            float p = w[i] / sum;
            acc += p;
            if (u <= acc) { pChosen = p; return i; }
        }
        pChosen = w[n-1]/sum; return n-1; // numeric tail
    } else {
        uint idx = min((uint)floor(u * n), n-1);
        pChosen = 1.0 / float(n);
        return idx;
    }
}

inline uint LT_PickAndRescale(in float w[4], uint n, float xi_in,
                              out float p_chosen, out float xi_out)
{
    float sum = 0.0;
    [unroll] for (uint i=0;i<4;i++) if (i<n) sum += max(w[i], 0.0);

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
    [unroll] for (uint i=0;i<4;i++){
        if (i>=n) break;
        float wi = max(w[i], 0.0);
        float next = accum + wi;
        if (target < next) { idx = i; break; }
        accum = next;
    }
    float wi = max(w[idx], 0.0);
    p_chosen = wi / sum;
    xi_out = (wi > 0.0) ? saturate((target - accum) / wi) : 0.0;
    return idx;
}



struct LTLeaf { uint triFirst; uint triCount; uint nodeIndex; };

// Custom angle importance factor based on angle between normal and corner direction
// Idea:
// - the more corners are behind the surface, the less volume is relevant -> downweight based on number
// - angle between normal and corner direction can be interpreted as cheap lambertian approx of the node -> downweight if all corners and the center have a small angle
inline float LT_AabbVertexVisibilityWeight(float3 x, float3 n, float3 bmin, float3 bmax)
{
    float3 lo = min(bmin, bmax);
    float3 hi = max(bmin, bmax);
    float3 c  = 0.5 * (lo + hi);

    float3 P[9] = {
        float3(lo.x,lo.y,lo.z), float3(hi.x,lo.y,lo.z),
        float3(lo.x,hi.y,lo.z), float3(hi.x,hi.y,lo.z),
        float3(lo.x,lo.y,hi.z), float3(hi.x,lo.y,hi.z),
        float3(lo.x,hi.y,hi.z), float3(hi.x,hi.y,hi.z),
        c
    };
    const float angleFloorMin = 0.1;

    float maxFrontCos = 0.0;   // best visible corner
    uint  behindCount = 0u;

    [unroll] for (uint i = 0; i < 9; ++i)
    {
        float3 v   = P[i] - x;
        float  l2  = max(dot(v, v), EPSILON);
        float3 dir = v * rsqrt(l2);

        float cosT = dot(n, dir);
        if (cosT > 0.0) {
            maxFrontCos = max(maxFrontCos, cosT);
        } else {
            behindCount++;
        }
    }

    // no front-facing samples?
    if (maxFrontCos <= 0.0) return 0.0;

    float angleFactor = max(maxFrontCos, angleFloorMin);

    // behind penalty
    float denom   = (behindCount > 0u) ? float(behindCount) : 1.0;
    float penalty = 1.0 / denom;

    return angleFactor * penalty;
}


// CONTY–KULLA NODE IMPORTANCE
inline float LT_NodeImportance_Common(
    float3 x, float3 n,
    float3 bmin, float3 bmax,
    float3 axis, float theta_o, float theta_e,
    float power, bool isGlobalLight, float)
{
    const float3 c = 0.5 * (bmin + bmax);
    const float3 e = 0.5 * (bmax - bmin);
    const float  R = length(e);

    float3 v  = x - c;
    float  d0 = length(v);
    float3 dir_to_point = (d0 > 0.0) ? (v / d0) : float3(0,0,1);

    float theta     = LT_SafeAcos(dot(axis, dir_to_point));
    float theta_u   = LT_SafeAsin(saturate(R / max(d0, 1e-6)));
    float theta_p   = max(theta - theta_o - theta_u, 0.0);
    if (theta_p >= theta_e) return 0.0;         // outside emission lobe
    float orientTerm = cos(theta_p);            // conservative emitter cosine

    float ci    = dot(-dir_to_point, n);
    float sin_u = sin(theta_u);
    if (ci <= -sin_u) return 0.0;

    float d2   = isGlobalLight ? 1.0 : (d0*d0 + R*R); // conservative near field clamp
    float geom = isGlobalLight ? 1.0 : rcp(d2);

    float visCorners = LT_AabbVertexVisibilityWeight(x, n, bmin, bmax);

    return power * geom * orientTerm * visCorners;
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
    return clamp(prob, 0.0001f, 0.9999f);
}

// STOCHASTIC DESCENT
uint LT_DescendTLAS_Stratified(float3 x, float3 n, inout float xi, out float pdfTLAS)
{
    pdfTLAS = 1.0;
    uint node = 0;

    [loop] for (;;)
    {
        LightTLASNodeGpu N = gLT_TLAS[node];
        if (N.childCount == 0) {
            return N.blasIndex; // leaf
        }

        float w[4];
        [unroll] for (uint i=0;i<4;i++){
            if (i < N.childCount) {
                LightTLASNodeGpu C = gLT_TLAS[N.firstChild + i];
                w[i] = max(LT_NodeImportance_TLAS(C, x, n), 0.0);
            } else {
                w[i] = 0.0;
            }
        }

        float p, xi_next;
        uint  idx = LT_PickAndRescale(w, N.childCount, xi, p, xi_next);
        pdfTLAS *= p;
        node = N.firstChild + idx;
        xi   = xi_next; // rescaled for the next level
    }
}

LTLeaf LT_DescendBLAS_Stratified(float3 x, float3 n, uint blasIndex, inout float xi, out float pdfBLAS)
{
    pdfBLAS = 1.0;
    BlasRangeGpu R = gLT_Range[blasIndex];
    uint node = 0;

    [loop] for (;;)
    {
        LightBLASNodeGpu N = gLT_BLAS[R.nodeOffset + node];
        if (N.childCount == 0) {
            LTLeaf L; L.triFirst = N.triFirst; L.triCount = N.triCount; L.nodeIndex = node;
            return L;
        }

        float w[4];
        [unroll] for (uint i=0;i<4;i++){
            if (i < N.childCount) {
                LightBLASNodeGpu C = gLT_BLAS[R.nodeOffset + (N.firstChild + i)];
                w[i] = max(LT_NodeImportance_BLAS(C, x, n), 0.0);
            } else {
                w[i] = 0.0;
            }
        }

        float p, xi_next;
        uint  idx = LT_PickAndRescale(w, N.childCount, xi, p, xi_next);
        pdfBLAS *= max(p, 1e-20);
        node = N.firstChild + idx;
        xi   = xi_next;
    }
}

// LEAF TRIANGLE SAMPLING
uint LT_SampleLeafTriangle_Stratified(float3 x, float3 n,
                                      uint blasIndex, LTLeaf leaf,
                                      inout float xi, out float pdfLeaf)
{
    BlasRangeGpu R   = gLT_Range[blasIndex];
    uint base        = R.triIndexOffset + leaf.triFirst;
    uint count       = max(leaf.triCount, 1u);

#if LT_LEAF_SAMPLER_MODE == 1
    // Uniform
    uint  k        = min((uint)floor(xi * count), count - 1u);
    uint  triIndex = gLT_LeafTriIndex[base + k];

    pdfLeaf = 1.0 / (float)count;

    // rescale xi into chosen subinterval
    float start = (float)k / (float)count;
    float width = 1.0 / (float)count;
    xi = saturate((xi - start) / width);
    return triIndex;

#else
    // Alias
    LightBLASNodeGpu node = gLT_BLAS[R.nodeOffset + leaf.nodeIndex];
    float sumW = max(node.power, 1e-20);

    float scaled = xi * count;
    uint  i0     = min((uint)floor(scaled), count - 1u);
    float r      = frac(scaled);

    uint  idx = base + i0;
    float q   = gLT_LeafAliasProb[idx];
    uint  a   = gLT_LeafAliasIdx [idx];

    const bool takePrimary = (r < q);
    uint  local    = takePrimary ? i0 : a;
    uint  triIndex = gLT_LeafTriIndex[base + local];

    float w = max(g_EmissiveTriangles[triIndex].weight, 0.0f);
    pdfLeaf = (sumW > 0.0f) ? (w / sumW) : 0.0f;

    // rescale xi within chosen branch
    xi = takePrimary ? (r / max(q, 1e-20f))
                     : ((r - q) / max(1.0f - q, 1e-20f));
    xi = saturate(xi);
    return triIndex;
#endif
}

// TOP-LEVEL SAMPLER
LT_Sample LT_SampleLight(float3 worldPos, float3 worldNormal, inout uint rng)
{
    // Draw ONE random number and reuse/rescale it through TLAS -> BLAS -> Leaf
    float xi = RandomFloatSingle(rng);

    float pdfT, pdfB, pdfL;

    uint   blas = LT_DescendTLAS_Stratified(worldPos, worldNormal, xi, pdfT);
    LTLeaf leaf = LT_DescendBLAS_Stratified(worldPos, worldNormal, blas, xi, pdfB);
    uint   tri  = LT_SampleLeafTriangle_Stratified(worldPos, worldNormal, blas, leaf, xi, pdfL);

    LT_Sample s; s.id = tri; s.pdf = pdfT * pdfB * pdfL;
    return s;
}

// Path Guiding sampler
LT_Path_Sample LT_SampleGuidedPath(float3 worldPos, float3 worldNormal, inout uint rng)
{
    // Draw ONE random number and reuse/rescale it through TLAS -> BLAS -> Leaf
    float xi = RandomFloatSingle(rng);

    float pdfT, pdfB, ...;

    uint   blas = LT_DescendTLAS_Stratified(worldPos, worldNormal, xi, pdfT);
    LTLeaf leaf = LT_DescendBLAS_Stratified(worldPos, worldNormal, blas, xi, pdfB);

    LT_Path_Sample s; s.dir = tri; s.pdf = pdfT * pdfB * ...;
    return s;
}

// PDF
float LT_PdfSelectTriangle(float3 x, float3 n, uint triIndex)
{
    uint blas = gLT_TriToBLAS[triIndex];
    if (blas == LT_SENTINEL) return 0.0f;

    uint item = gLT_BLASToItem[blas];

    // TLAS path probability
    float pdfTLAS = 1.0f;
    uint  tnode   = 0;

    [loop] for (;;)
    {
        LightTLASNodeGpu N = gLT_TLAS[tnode];
        if (N.childCount == 0) break;

        float w[4]; float sum=0.0;
        int childHit = -1;

        [unroll] for (uint i=0;i<4;i++){
            if (i >= N.childCount) break;
            LightTLASNodeGpu C = gLT_TLAS[N.firstChild + i];
            w[i] = max(LT_NodeImportance_TLAS(C, x, n), 0.0);
            sum += w[i];
            bool inChild = (item >= C.itemFirst) && (item < (C.itemFirst + C.itemCount));
            if (inChild) childHit = (int)i;
        }

        if (childHit < 0) {
            // shouldn’t happen; uniform fallback
            pdfTLAS *= 1.0 / float(N.childCount);
            tnode = N.firstChild; // pick first for determinism
            continue;
        }

        float p = (sum > 0.0) ? (w[childHit] / sum) : (1.0 / float(N.childCount));
        pdfTLAS *= p;
        tnode = N.firstChild + (uint)childHit;
    }

    // BLAS path probability
    BlasRangeGpu Rng = gLT_Range[blas];
    float pdfBLAS = 1.0f;
    uint  bnode   = 0;

    uint localIdx = gLT_TriToLeafOffset[triIndex];

    [loop] for (;;)
    {
        LightBLASNodeGpu N = gLT_BLAS[Rng.nodeOffset + bnode];

        if (N.childCount == 0)
        {
#if LT_LEAF_SAMPLER_MODE == 1
            // UNIFORM pdf
            uint  count   = max(N.triCount, 1u);
            float pdfLeaf = 1.0f / (float)count;
            return pdfTLAS * pdfBLAS * pdfLeaf;
#else
            // ALIAS pdf
            float sumW    = max(N.power, 1e-20f);
            float wTri    = max(g_EmissiveTriangles[triIndex].weight, 0.0f);
            float pdfLeaf = (sumW > 0.0f) ? (wTri / sumW) : 0.0f;
            return pdfTLAS * pdfBLAS * pdfLeaf;
#endif
        }

        // Find which child contains localIdx
        float w[4]; float sum = 0.0;
        int childHit = -1;

        [unroll] for (uint i=0;i<4;i++){
            if (i >= N.childCount) break;
            LightBLASNodeGpu C = gLT_BLAS[Rng.nodeOffset + (N.firstChild + i)];
            w[i] = max(LT_NodeImportance_BLAS(C, x, n), 0.0);
            sum += w[i];
            bool inChild = (localIdx >= C.triFirst) && (localIdx < (C.triFirst + C.triCount));
            if (inChild) childHit = (int)i;
        }

        if (childHit < 0) {
            // Uniform fallback if not found
            pdfBLAS *= 1.0 / float(N.childCount);
            bnode = N.firstChild;
            continue;
        }

        float p = (sum > 0.0) ? (w[childHit] / sum) : (1.0 / float(N.childCount));
        pdfBLAS *= p;
        bnode = N.firstChild + (uint)childHit;
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
    float area     = max(1e-10, LT_TriangleArea(tri, objID));
    return p_select / area;
}
