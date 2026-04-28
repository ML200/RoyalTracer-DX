//====================================
//LIGHT TREE LOOKUP BUFFERS
//====================================
Buffer<uint> gLT_TriToBLAS       : register(t16);
Buffer<uint> gLT_TriToLeafOffset : register(t17);
Buffer<uint> gLT_BLASToItem      : register(t18);

//====================================
//CONSTANTS AND HELPERS
//====================================
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

//====================================
//TRIG-FREE NODE IMPORTANCE
//====================================
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

    //relative floor on receiver/orient, hard-cut at boundaries produces RIS fireflies
    //consistent on sampling and PDF sides, RIS/MIS stay unbiased
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

//====================================
//Q-SMOOTHED PER-NODE DESCENT PROBABILITY
//====================================
//gain=0 makes the helpers no-ops, so the descent collapses to the original importance pick
//q_i = s + (1-s)*a_i, normalized: balanced children -> near-uniform, dominated children -> stay biased
//removes cluster-boundary banding without splitting/reservoir register cost
static const float LT_SPLIT_GAIN = 1.0f;
static const float LT_SPLIT_MAX  = 0.5f;

//s rises with min/max child importance ratio, capped to keep dominated paths concentrated
inline float LT_SmoothS(float w[4], uint n)
{
    if (LT_SPLIT_GAIN <= 0.0f) return 0.0f;
    float lo = 1e30f;
    float hi = 0.0f;
    [unroll] for (uint i=0;i<4;i++) {
        if (i >= n) break;
        if (w[i] > 0.0f) {
            lo = min(lo, w[i]);
            hi = max(hi, w[i]);
        }
    }
    if (hi <= 0.0f || lo >= 1e29f) return 0.0f;
    float r = saturate(lo / hi);
    return clamp(r * LT_SPLIT_GAIN, 0.0f, LT_SPLIT_MAX);
}

//in-place: replaces importance weights with un-normalized q values
//LT_PickAndRescale normalizes by sum(w) so passing q yields p = q_i / sum(q), the smoothed descent prob
inline void LT_QSmoothInPlace(inout float w[4], uint n)
{
    if (LT_SPLIT_GAIN <= 0.0f) return;
    float s = LT_SmoothS(w, n);
    if (s <= 0.0f) return;

    float sumW = 0.0f;
    [unroll] for (uint i=0;i<4;i++) if (i<n) sumW += max(w[i], 0.0f);
    if (sumW <= 0.0f) return;

    [unroll] for (uint i=0;i<4;i++) {
        if (i < n && w[i] > 0.0f) {
            float a_i = w[i] / sumW;
            w[i] = s + (1.0f - s) * a_i;
        }
    }
}

//====================================
//STOCHASTIC DESCENT
//====================================
uint LT_DescendTLAS_Stratified(float3 x, float3 n, inout float xi, out float pdfTLAS)
{
    pdfTLAS = 1.0;
    uint node = 0;

    //depth cap, malformed tree could hang GPU, zero pdf invalidates sample
    [loop] for (uint iter = 0u; iter < 64u; ++iter)
    {
        LightTLASNodeGpu N = gLT_TLAS[node];
        if (N.childCount == 0) {
            return N.blasIndex;
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

        //smooth near-balanced clusters to remove visible cut-boundary banding
        LT_QSmoothInPlace(w, N.childCount);

        float p, xi_next;
        uint  idx = LT_PickAndRescale(w, N.childCount, xi, p, xi_next);
        pdfTLAS *= p;
        node = N.firstChild + idx;
        xi   = xi_next;
    }

    pdfTLAS = 0.0f;
    return 0u;
}

LTLeaf LT_DescendBLAS_Stratified(float3 x, float3 n, uint blasIndex, inout float xi, out float pdfBLAS)
{
    pdfBLAS = 1.0;
    BlasRangeGpu R = gLT_Range[blasIndex];

    //BLAS nodes in world space, triangles transformed via instance.objectToWorld at build
    //worldToLocal kept for future refit, unused on fully-rebuilt-at-load path
    float3 xLocal = x;
    float3 nLocal = n;

    uint node = 0;

    //depth cap, zero pdf on overflow, zero-count leaf is safe since LeafTriangle clamps
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

        //smooth near-balanced clusters to remove visible cut-boundary banding
        LT_QSmoothInPlace(w, N.childCount);

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

//====================================
//LEAF TRIANGLE SAMPLING
//====================================
//power-weighted within leaf, uniform picks waste samples on dim tris when one dominates
//2-pass, sum weights then pick proportionally, matches LT_PdfSelectTriangle's leaf
uint LT_SampleLeafTriangle_Stratified(float3 x, float3 n,
                                      uint blasIndex, LTLeaf leaf,
                                      inout float xi, out float pdfLeaf)
{
    BlasRangeGpu R   = gLT_Range[blasIndex];
    uint base        = R.triIndexOffset + leaf.triFirst;
    uint count       = max(leaf.triCount, 1u);

    //pass 1, sum per-tri weights
    float sumW = 0.0f;
    [loop] for (uint i = 0u; i < count; ++i) {
        const uint tj = gLT_LeafTriIndex[base + i];
        sumW += max(g_EmissiveTriangles[tj].weight, 0.0f);
    }

    //degenerate fallback, uniform
    if (sumW <= 0.0f) {
        uint k = min((uint)floor(xi * count), count - 1u);
        pdfLeaf = 1.0f / (float)count;
        float start = (float)k / (float)count;
        float width = 1.0f / (float)count;
        xi = saturate((xi - start) / width);
        return gLT_LeafTriIndex[base + k];
    }

    //pass 2, pick on cumulative cross
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

    xi = (selW > 0.0f) ? saturate((target - accum) / selW) : 0.0f;
    return triIndex;
}

//====================================
//TOP-LEVEL SAMPLER
//====================================
LT_Sample LT_SampleLight(float3 worldPos, float3 worldNormal, inout uint rng)
{
    //one random per stratum, reused/rescaled down TLAS -> BLAS -> Leaf
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


//====================================
//PDF
//====================================
float LT_PdfSelectTriangle(float3 x, float3 n, uint triIndex)
{
    uint blas = gLT_TriToBLAS[triIndex];
    if (blas == LT_SENTINEL) return 0.0f;

    uint item = gLT_BLASToItem[blas];

    //TLAS path probability
    float pdfTLAS = 1.0f;
    uint  tnode   = 0;

    //depth cap, zero pdf on overflow
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
            //uniform fallback for malformed itemFirst/itemCount, no q correction
            pdfTLAS *= 1.0 / float(N.childCount);
            tnode = N.firstChild;
            continue;
        }

        //mirror sampler: replace w with q in-place, then p = w[hit] / sum(w) is the smoothed prob
        LT_QSmoothInPlace(w, N.childCount);
        float sumQ = 0.0f;
        [unroll] for (uint i=0;i<4;i++) if (i<N.childCount) sumQ += max(w[i], 0.0f);
        float p = (sumQ > 0.0f && w[childHit] > 0.0f) ? (w[childHit] / sumQ) : 0.0f;
        pdfTLAS *= p;
        tnode = N.firstChild + (uint)childHit;
    }
    if (!tlasLeafReached) return 0.0f;

    //BLAS path probability, world space
    BlasRangeGpu Rng = gLT_Range[blas];
    float3 xLocal = x;
    float3 nLocal = n;

    float pdfBLAS = 1.0f;
    uint  bnode   = 0;

    uint localIdx = gLT_TriToLeafOffset[triIndex];

    [loop] for (uint iterB = 0u; iterB < 64u; ++iterB)
    {
        LightBLASNodeGpu N = gLT_BLAS[Rng.nodeOffset + bnode];

        if (N.childCount == 0)
        {
            //matches LT_SampleLeafTriangle_Stratified power-weight form
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

        LT_QSmoothInPlace(w, N.childCount);
        float sumQ = 0.0f;
        [unroll] for (uint i=0;i<4;i++) if (i<N.childCount) sumQ += max(w[i], 0.0f);
        float p = (sumQ > 0.0f && w[childHit] > 0.0f) ? (w[childHit] / sumQ) : 0.0f;
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

//====================================
//LIGHT SAMPLE RESULT
//====================================
struct LT_LightSampleResult
{
    float3 position;
    float3 normal;
    float3 emission;
    float  pdfSolidAngle;
    uint   triIndex;
    uint   objID;
};

LT_LightSampleResult LT_SamplePointOnLight(float3 refPos, float3 refNormal, inout uint rng)
{
    LT_LightSampleResult result;

    LT_Sample treeSample = LT_SampleLight(refPos, refNormal, rng);
    result.triIndex = treeSample.id;

    LightTriangle triData = g_EmissiveTriangles[result.triIndex];
    result.objID    = triData.instanceID;
    result.emission = triData.emission;

    float4x4 worldMat = instanceProps[result.objID].objectToWorld;
    float3 v0 = mul(worldMat, float4(triData.x, 1.0)).xyz;
    float3 v1 = mul(worldMat, float4(triData.y, 1.0)).xyz;
    float3 v2 = mul(worldMat, float4(triData.z, 1.0)).xyz;

    //uniform tri sample
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
    result.normal = crossP / area2;
    float area = 0.5f * area2;

    //PDF_SA = PDF_Area * dist^2 / cosTheta_Light
    float3 toLight = result.position - refPos;
    float distSq   = dot(toLight, toLight);
    float dist     = sqrt(distSq);

    float cosLight = max(dot(result.normal, -toLight / dist), 0.0f);

    float pdfArea = treeSample.pdf / max(area, 1e-10f);

    if (cosLight > 1e-6f) {
        result.pdfSolidAngle = pdfArea * distSq / cosLight;
    } else {
        result.pdfSolidAngle = 0.0f;
    }

    return result;
}
