//====================================================================
//LIGHT TREE LOOKUP BUFFERS
//====================================================================
Buffer<uint> gLT_TriToBLAS       : register(t16);
Buffer<uint> gLT_TriToLeafOffset : register(t17);
Buffer<uint> gLT_BLASToItem      : register(t18);

//====================================================================
//CONSTANTS AND HELPERS
//====================================================================
static const uint  LT_SENTINEL = 0xFFFFFFFFu;
static const float LT_PI = 3.14159265358979323846;

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
    xi_out = (wi > 0.0) ? ((target - accum) / wi) : 0.0;
    xi_out = clamp(xi_out, 0.0, ONE_MINUS_EPSILON);
    return idx;
}



struct LTLeaf { uint triFirst; uint triCount; uint nodeIndex; };

//====================================================================
//TRIG-FREE NODE IMPORTANCE
//====================================================================
//Precomputed cosTheta_o / cosTheta_e in nodes.
inline float LT_NodeImportance_Common(
    float3 x, float3 n,
    float3 bmin, float3 bmax,
    float3 axis, float cosTheta_o, float cosTheta_e,
    float power)
{
    const float3 c = 0.5 * (bmin + bmax);
    const float3 e = 0.5 * (bmax - bmin);
    const float  R2 = dot(e, e);

    float3 v  = x - c;
    float  d2 = dot(v, v);
    float  invD = rsqrt(max(d2, 1e-12));
    float  d0 = d2 * invD;
    float3 dir = v * invD;

    float  R    = sqrt(R2);
    float  sinU = saturate(R * invD);
    float  cosU = sqrt(max(1.0 - sinU * sinU, 0.0));

    float cosTheta  = dot(axis, dir);
    float sinTheta  = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    float sinTheta_o = sqrt(max(1.0 - cosTheta_o * cosTheta_o, 0.0));

    float cosA = cosTheta * cosTheta_o + sinTheta * sinTheta_o;
    float sinA = sinTheta * cosTheta_o - cosTheta * sinTheta_o;
    float cosFull = cosA * cosU + sinA * sinU;

    float cosOuterBound = cosTheta_o * cosU - sinTheta_o * sinU;
    float sinOuterBound = sinTheta_o * cosU + cosTheta_o * sinU;
    bool  insideCone = (sinOuterBound <= 0.0) || (cosTheta >= cosOuterBound);

    float orientTerm = insideCone ? 1.0 : max(cosFull, 0.0);

    float ci = dot(-dir, n);
    float cosReceiver = saturate(ci + sinU);

    float geom = rcp(d2 + R2);

    //Apply a small relative floor to the receiver term and orient term
    //instead of hard-cutting to 0 when the cluster is just past the
    //angular horizon (ci <= -sinU) or past the emissive cone
    //(cosFull <= cosTheta_e). Pixels whose shading normals sit right at
    //the cutoff would otherwise see the cluster's pdf drop from a small
    //positive value to exactly 0, and any sample drawn in that transition
    //band produces wi = p_hat/pdf fireflies. The floor keeps the pdf
    //continuous and is consistent on both sampling and PDF sides, so
    //RIS/MIS remain unbiased. Samples genuinely behind the receiver still
    //get rejected at the cosSurf gate in NEE, the floor just wastes a
    //tiny fraction of samples on them instead of producing spikes near
    //the boundary.
    const float kFloor = 0.01;
    cosReceiver = max(cosReceiver, kFloor);
    orientTerm  = max(orientTerm,  kFloor);

    return power * geom * orientTerm * cosReceiver;
}

inline float LT_NodeImportance_TLAS(LightTLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.cosTheta_o, n.cosTheta_e, n.power);
}
inline float LT_NodeImportance_BLAS(LightBLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.cosTheta_o, n.cosTheta_e, n.power);
}

inline float LT_BranchProb(float IL, float IR)
{
    float sum = IL + IR;
    float prob = (sum > 0.0f) ? (IL / sum) : 0.5f;
    return clamp(prob, 0.0001f, 0.9999f);
}

//====================================================================
//STOCHASTIC DESCENT
//====================================================================
uint LT_DescendTLAS_Stratified(float3 x, float3 n, inout float xi, out float pdfTLAS)
{
    pdfTLAS = 1.0;
    uint node = 0;

    //Depth cap: a malformed tree, stale childCount or cyclic firstChild,
    //would otherwise hang the GPU. 64 is far above any realistic tree
    //depth, zero pdf on overflow invalidates the sample downstream.
    [loop] for (uint iter = 0u; iter < 64u; ++iter)
    {
        LightTLASNodeGpu N = gLT_TLAS[node];
        if (N.childCount == 0) {
            return N.blasIndex; //leaf
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
        xi   = xi_next; //rescaled for the next level
    }

    pdfTLAS = 0.0f;
    return 0u;
}

LTLeaf LT_DescendBLAS_Stratified(float3 x, float3 n, uint blasIndex, inout float xi, out float pdfBLAS)
{
    pdfBLAS = 1.0;
    BlasRangeGpu R = gLT_Range[blasIndex];

    //BLAS nodes store WORLD-SPACE bounds and cone axes, triangles are
    //transformed via instance.objectToWorld at build time, see
    //LightTree.h buildBLASes_SAOH. Query in world space, the worldToLocal
    //field is kept in the range struct for future refit support but is
    //not used on the current fully-rebuilt-at-load path.
    float3 xLocal = x;
    float3 nLocal = n;

    uint node = 0;

    //Depth cap, see LT_DescendTLAS_Stratified. Zero pdf on overflow
    //invalidates the sample, returning a zero-count leaf is a safe token
    //since LT_SampleLeafTriangle_Stratified clamps triCount via max(..., 1u).
    [loop] for (uint iter = 0u; iter < 64u; ++iter)
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
                w[i] = max(LT_NodeImportance_BLAS(C, xLocal, nLocal), 0.0);
            } else {
                w[i] = 0.0;
            }
        }

        float p, xi_next;
        uint  idx = LT_PickAndRescale(w, N.childCount, xi, p, xi_next);
        pdfBLAS *= p;
        node = N.firstChild + idx;
        xi   = xi_next;
    }

    pdfBLAS = 0.0f;
    LTLeaf L; L.triFirst = 0u; L.triCount = 0u; L.nodeIndex = 0u;
    return L;
}

//====================================================================
//LEAF TRIANGLE SAMPLING
//====================================================================
//Power-weighted selection within the leaf. A leaf can hold up to
//maxLeafTris triangles, LightTreeBuilder default = 16, picking them
//uniformly wastes samples on dim triangles when one dominates the leaf's
//emission, that was the dominant-light-still-noisy failure mode. Two-pass
//scan: first pass sums per-triangle weight, second pass picks
//proportionally. Matched by LT_PdfSelectTriangle's leaf case.
uint LT_SampleLeafTriangle_Stratified(float3 x, float3 n,
                                      uint blasIndex, LTLeaf leaf,
                                      inout float xi, out float pdfLeaf)
{
    BlasRangeGpu R   = gLT_Range[blasIndex];
    uint base        = R.triIndexOffset + leaf.triFirst;
    uint count       = max(leaf.triCount, 1u);

    //Pass 1: accumulate per-triangle weight, emission power.
    float sumW = 0.0f;
    [loop] for (uint i = 0u; i < count; ++i) {
        const uint tj = gLT_LeafTriIndex[base + i];
        sumW += max(g_EmissiveTriangles[tj].weight, 0.0f);
    }

    //Degenerate fallback: all weights zero -> uniform.
    if (sumW <= 0.0f) {
        uint k = min((uint)floor(xi * count), count - 1u);
        pdfLeaf = 1.0f / (float)count;
        float start = (float)k / (float)count;
        float width = 1.0f / (float)count;
        xi = saturate((xi - start) / width);
        return gLT_LeafTriIndex[base + k];
    }

    //Pass 2: pick the triangle whose cumulative weight crosses xi*sumW.
    const float target = xi * sumW;
    float accum = 0.0f;
    uint  sel = count - 1u;
    float selW = 0.0f;
    [loop] for (uint k = 0u; k < count; ++k) {
        const uint tk = gLT_LeafTriIndex[base + k];
        const float w = max(g_EmissiveTriangles[tk].weight, 0.0f);
        const float next = accum + w;
        if (target < next) { sel = k; selW = w; break; }
        accum = next;
    }

    pdfLeaf = selW / sumW;
    const uint triIndex = gLT_LeafTriIndex[base + sel];

    //Rescale xi into the chosen subinterval for the caller's next use.
    xi = (selW > 0.0f) ? saturate((target - accum) / selW) : 0.0f;
    return triIndex;
}

//====================================================================
//TOP-LEVEL SAMPLER
//====================================================================
LT_Sample LT_SampleLight(float3 worldPos, float3 worldNormal, inout uint rng)
{
    //Draw ONE random number and reuse/rescale it through TLAS -> BLAS -> Leaf
    float xiT = RandomFloatSingle(rng);
    float xiB = RandomFloatSingle(rng);
    float xiL = RandomFloatSingle(rng);

    float pdfT, pdfB, pdfL;

    uint   blas = LT_DescendTLAS_Stratified(worldPos, worldNormal, xiT, pdfT);
    LTLeaf leaf = LT_DescendBLAS_Stratified(worldPos, worldNormal, blas, xiB, pdfB);
    uint   tri  = LT_SampleLeafTriangle_Stratified(worldPos, worldNormal, blas, leaf, xiL, pdfL);

    LT_Sample s; s.id = tri; s.pdf = pdfT * pdfB * pdfL;
    return s;
}


//====================================================================
//PDF
//====================================================================
float LT_PdfSelectTriangle(float3 x, float3 n, uint triIndex)
{
    uint blas = gLT_TriToBLAS[triIndex];
    if (blas == LT_SENTINEL) return 0.0f;

    uint item = gLT_BLASToItem[blas];

    //TLAS path probability
    float pdfTLAS = 1.0f;
    uint  tnode   = 0;

    //Depth cap, see LT_DescendTLAS_Stratified. On overflow the sample
    //is invalidated by returning zero pdf.
    bool tlasLeafReached = false;
    [loop] for (uint iterT = 0u; iterT < 64u; ++iterT)
    {
        LightTLASNodeGpu N = gLT_TLAS[tnode];
        if (N.childCount == 0) { tlasLeafReached = true; break; }

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
            //shouldn't happen, uniform fallback
            pdfTLAS *= 1.0 / float(N.childCount);
            tnode = N.firstChild; //pick first for determinism
            continue;
        }

        float p = (sum > 0.0) ? (w[childHit] / sum) : (1.0 / float(N.childCount));
        pdfTLAS *= p;
        tnode = N.firstChild + (uint)childHit;
    }
    if (!tlasLeafReached) return 0.0f;

    //BLAS path probability. BLAS nodes are in world space, see
    //LT_DescendBLAS_Stratified for rationale. Query in world space.
    BlasRangeGpu Rng = gLT_Range[blas];
    float3 xLocal = x;
    float3 nLocal = n;

    float pdfBLAS = 1.0f;
    uint  bnode   = 0;

    uint localIdx = gLT_TriToLeafOffset[triIndex];

    //Depth cap, see LT_DescendTLAS_Stratified. If the loop exits without
    //finding a leaf, fall through to the trailing return 0.0f.
    [loop] for (uint iterB = 0u; iterB < 64u; ++iterB)
    {
        LightBLASNodeGpu N = gLT_BLAS[Rng.nodeOffset + bnode];

        if (N.childCount == 0)
        {
            //Matches LT_SampleLeafTriangle_Stratified: per-triangle power
            //weight. Sum weights over the leaf and return the target
            //triangle's share. Degenerate fallback mirrors sampling.
            const uint  count    = max(N.triCount, 1u);
            const uint  leafBase = Rng.triIndexOffset + N.triFirst;

            float sumW = 0.0f;
            float myW  = 0.0f;
            [loop] for (uint j = 0u; j < count; ++j) {
                const uint tj = gLT_LeafTriIndex[leafBase + j];
                const float w = max(g_EmissiveTriangles[tj].weight, 0.0f);
                sumW += w;
                if (tj == triIndex) myW = w;
            }

            const float pdfLeaf = (sumW > 0.0f) ? (myW / sumW)
                                                : (1.0f / (float)count);
            return pdfTLAS * pdfBLAS * pdfLeaf;
        }

        //Find which child contains localIdx
        float w[4]; float sum = 0.0;
        int childHit = -1;

        [unroll] for (uint i=0;i<4;i++){
            if (i >= N.childCount) break;
            LightBLASNodeGpu C = gLT_BLAS[Rng.nodeOffset + (N.firstChild + i)];
            w[i] = max(LT_NodeImportance_BLAS(C, xLocal, nLocal), 0.0);
            sum += w[i];
            bool inChild = (localIdx >= C.triFirst) && (localIdx < (C.triFirst + C.triCount));
            if (inChild) childHit = (int)i;
        }

        if (childHit < 0) {
            pdfBLAS *= 1.0 / float(N.childCount);
            bnode = N.firstChild;
            continue;
        }

        float p = (sum > 0.0) ? (w[childHit] / sum) : (1.0 / float(N.childCount));
        pdfBLAS *= p;
        bnode = N.firstChild + (uint)childHit;
    }

    return 0.0f;
}

inline float LT_TriangleArea(uint tri, uint objID)
{
    float3 A = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].x, 1)).xyz;
    float3 B = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].y, 1)).xyz;
    float3 C = mul(instanceProps[objID].objectToWorld, float4(g_EmissiveTriangles[tri].z, 1)).xyz;
    return 0.5 * length(cross(B - A, C - A));
}

float LT_Pdf_LightTree_Area(float3 x, float3 n, uint tri, uint objID)
{
    float p_select = LT_PdfSelectTriangle(x, n, tri);
    float area     = max(1e-10, LT_TriangleArea(tri, objID));
    return p_select / area;
}

//====================================================================
//LIGHT SAMPLE RESULT
//====================================================================
//Helper to select a light source, pick a point on it, return point and pdf.
struct LT_LightSampleResult
{
    float3 position;      //world space position on light
    float3 normal;        //geometric normal of the light triangle
    float3 emission;      //emissive color
    float  pdfSolidAngle; //PDF w.r.t Solid Angle
    uint   triIndex;      //selected global triangle index
    uint   objID;         //InstanceID of the light
};

//Requires: g_EmissiveTriangles, instanceProps
LT_LightSampleResult LT_SamplePointOnLight(float3 refPos, float3 refNormal, inout uint rng)
{
    LT_LightSampleResult result;

    //Pick a triangle from the tree
    LT_Sample treeSample = LT_SampleLight(refPos, refNormal, rng);
    result.triIndex = treeSample.id;

    //Fetch Triangle Data
    LightTriangle triData = g_EmissiveTriangles[result.triIndex];
    result.objID    = triData.instanceID;
    result.emission = triData.emission;

    //Transform Vertices
    float4x4 worldMat = instanceProps[result.objID].objectToWorld;
    float3 v0 = mul(worldMat, float4(triData.x, 1.0)).xyz;
    float3 v1 = mul(worldMat, float4(triData.y, 1.0)).xyz;
    float3 v2 = mul(worldMat, float4(triData.z, 1.0)).xyz;

    //Sample Point Uniformly
    float r1 = RandomFloatSingle(rng);
    float r2 = RandomFloatSingle(rng);
    float sqrtR1 = sqrt(r1);
    float u = 1.0f - sqrtR1;
    float v = r2 * sqrtR1;

    result.position = (1.0f - u - v) * v0 + u * v1 + v * v2;

    //Normal and Area
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 crossP = cross(e1, e2);
    float area2 = length(crossP);
    result.normal = crossP / area2; //normalize
    float area = 0.5f * area2;

    //Calculate Solid Angle PDF
    //PDF_SA = PDF_Area * (dist^2 / cosTheta_Light)
    float3 toLight = result.position - refPos;
    float distSq   = dot(toLight, toLight);
    float dist     = sqrt(distSq);

    //Check for valid geometry, avoid divide by zero
    float cosLight = max(dot(result.normal, -toLight / dist), 0.0f);

    float pdfArea = treeSample.pdf / max(area, 1e-10f);

    if (cosLight > 1e-6f) {
        result.pdfSolidAngle = pdfArea * distSq / cosLight;
    } else {
        result.pdfSolidAngle = 0.0f;
    }

    return result;
}
