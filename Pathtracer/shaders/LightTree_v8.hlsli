//====================================
//LIGHT TREE LOOKUP BUFFERS
//====================================
Buffer<uint> gLT_TriToBLAS       : register(t16);
Buffer<uint> gLT_TriBitTrail     : register(t17);  // 2-bits-per-level BLAS descent path per emissive triangle
Buffer<uint> gLT_BLASBitTrail    : register(t18);  // 2-bits-per-level TLAS descent path per BLAS

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
//importance bound from the ATS light-tree reference
//deliberately matches the reference quirks: theta_u=PI inside the bbox (not just the sphere),
//cos_theta clamped to [0,1] so back-facing clusters get the horizon-grazing factor instead of zero,
//distance² clamped to max(d², 2R) (units are mismatched in the reference but we copy as-is for parity),
//and no theta_e clip in the importance bound (theta_e only appears in the SAOH M_Omega build cost).
//sinTheta_o is precomputed at build time and passed in to skip a sqrt per call.
inline float LT_NodeImportance_Common(
    float3 x, float3 n,
    float3 bmin, float3 bmax,
    float3 axis, float cosTheta_o, float sinTheta_o,
    float power)
{
    //if even the corner that maximizes dot(n, .) is below the receiver horizon, the whole bbox is, return 0
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
    const float  distSq   = max(max(d2, 2.0 * R), 1e-12);

    //inside-AABB short-circuits both factors to 1: theta_u=PI forces sinThetaU=0, cosThetaU=-1,
    //which makes cos_T=-cosTheta_o, sin_T=-sinTheta_o<=0, so cos_theta_prime collapses to 1.
    //skips a normalize, two dot products, and the entire orientation chain.
    if (x.x >= bmin.x && x.x <= bmax.x &&
        x.y >= bmin.y && x.y <= bmax.y &&
        x.z >= bmin.z && x.z <= bmax.z)
    {
        return power / distSq;
    }

    const float  d         = sqrt(d2);
    const float3 toCenterN = toCenter / d;

    //inside-sphere -> theta_u = PI/2; else arcsin(R/d)
    float sinThetaU, cosThetaU;
    const float ratio = R / d;
    if (ratio >= 1.0f) {
        sinThetaU = 1.0f;
        cosThetaU = 0.0f;
    } else {
        sinThetaU = ratio;
        cosThetaU = sqrt(max(1.0f - sinThetaU * sinThetaU, 0.0f));
    }

    //receiver: cos(max(0, theta_i - theta_u))
    //reference does NOT clamp the result to >=0; the bbox-vs-normal early reject above is supposed to mask that
    const float ci = dot(n, toCenterN);
    float cos_i_prime;
    if (ci >= cosThetaU) {
        cos_i_prime = 1.0f;
    } else {
        const float si = sqrt(max(1.0f - ci * ci, 0.0f));
        cos_i_prime = ci * cosThetaU + si * sinThetaU;
    }

    //orientation: cos_theta clamped to [0,1] so back-facing clusters land on the horizon
    const float cosTheta = clamp(dot(axis, -toCenterN), 0.0f, 1.0f);
    const float sinTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f));

    //T = theta_o + theta_u
    const float cos_T = cosTheta_o * cosThetaU - sinTheta_o * sinThetaU;
    const float sin_T = sinTheta_o * cosThetaU + cosTheta_o * sinThetaU;

    float cos_theta_prime;
    if (sin_T <= 0.0f) {
        //T >= PI, the extended cone wraps the sphere
        cos_theta_prime = 1.0f;
    } else if (cosTheta >= cos_T) {
        //theta <= T, point inside the extended cone
        cos_theta_prime = 1.0f;
    } else {
        //cos(theta - T)
        cos_theta_prime = max(cosTheta * cos_T + sinTheta * sin_T, 0.0f);
    }

    return cos_i_prime * power * cos_theta_prime / distSq;
}

inline float LT_NodeImportance_TLAS(LightTLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.cosTheta_o, n.sinTheta_o, n.power);
}
inline float LT_NodeImportance_BLAS(LightBLASNodeGpu n, float3 x, float3 norm)
{
    return LT_NodeImportance_Common(x, norm, n.bmin, n.bmax, n.axis, n.cosTheta_o, n.sinTheta_o, n.power);
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

    //BLAS nodes live in this instance's OBJECT space (one instance per BLAS), so
    //map the query point/normal in via worldToLocal (inverse of the shifted
    //objectToWorld, refreshed each light-tree refit). This keeps the BLAS
    //descent invariant to the instance transform AND the floating-origin snap -
    //world-space BLAS bounds went stale on every 1 km origin shift, breaking
    //importance + pdf once the camera left spawn. For uniform scale this is the
    //same distribution as the old world-space build (the per-BLAS scale cancels
    //in the PickAndRescale normalization).
    const float3 xL = mul(R.worldToLocal, float4(x, 1.0)).xyz;
    const float3 nL = normalize(mul((float3x3)R.worldToLocal, n));

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
                w[i] = max(LT_NodeImportance_BLAS(C, xL, nL), 0.0);
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

//====================================
//LEAF TRIANGLE SAMPLING
//====================================
//power-weighted within leaf, uniform picks waste samples on dim tris when one dominates
//fast path for the common count==1 case (maxLeafTris=1 build setting)
uint LT_SampleLeafTriangle_Stratified(uint blasIndex, LTLeaf leaf, float xi, out float pdfLeaf)
{
    BlasRangeGpu R = gLT_Range[blasIndex];
    uint base      = R.triIndexOffset + leaf.triFirst;

    //single-tri leaf: zero-cost selection
    if (leaf.triCount <= 1u) {
        pdfLeaf = 1.0f;
        return gLT_LeafTriIndex[base];
    }

    //fallback: degenerate-build leaves with multiple tris, power-weighted CDF
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

//====================================
//TOP-LEVEL SAMPLER
//====================================
LT_Sample LT_SampleLight(float3 worldPos, float3 worldNormal, inout uint rng)
{
    //one random per stratum, reused/rescaled down TLAS -> BLAS -> Leaf
    float xiT = RandomFloatSingle(rng);
    float xiB = RandomFloatSingle(rng);

    float pdfT, pdfB, pdfL;

    uint   blas = LT_DescendTLAS_Stratified(worldPos, worldNormal, xiT, pdfT);
    LTLeaf leaf = LT_DescendBLAS_Stratified(worldPos, worldNormal, blas, xiB, pdfB);
    //leaf-level xi only consumed for the rare multi-tri leaf fallback
    float xiL = (leaf.triCount > 1u) ? RandomFloatSingle(rng) : 0.0f;
    uint  tri = LT_SampleLeafTriangle_Stratified(blas, leaf, xiL, pdfL);

    LT_Sample s; s.id = tri; s.pdf = pdfT * pdfB * pdfL;
    return s;
}


//====================================
//PDF
//====================================
//bit trails record which child to descend into at each level (2 bits per level).
//build emits one trail per BLAS (TLAS descent) and one per emissive triangle (BLAS descent),
//so the PDF skips the per-child range-check for free and just pulls the right child terrain of the trail.
float LT_PdfSelectTriangle(float3 x, float3 n, uint triIndex)
{
    uint blas = gLT_TriToBLAS[triIndex];
    if (blas == LT_SENTINEL) return 0.0f;

    uint blasTrail = gLT_BLASBitTrail[blas];
    uint triTrail  = gLT_TriBitTrail[triIndex];

    //TLAS path probability
    float pdfTLAS = 1.0f;
    uint  tnode   = 0;
    uint  tdepth  = 0;

    [loop] for (uint iterT = 0u; iterT < 64u; ++iterT)
    {
        LightTLASNodeGpu N = gLT_TLAS[tnode];
        if (N.childCount == 0) break;

        const uint childIdx = (blasTrail >> (2u * tdepth)) & 3u;

        float w[4]; float sum = 0.0;
        [unroll] for (uint i=0;i<4;i++){
            if (i < N.childCount) {
                LightTLASNodeGpu C = gLT_TLAS[N.firstChild + i];
                w[i] = max(LT_NodeImportance_TLAS(C, x, n), 0.0);
                sum += w[i];
            } else {
                w[i] = 0.0;
            }
        }

        //sum==0 mirrors LT_PickAndRescale's uniform fallback
        float p = (sum > 0.0f) ? (w[childIdx] / sum) : (1.0f / float(N.childCount));
        pdfTLAS *= p;
        tnode = N.firstChild + childIdx;
        tdepth++;
    }

    //BLAS path probability - same object-space mapping as the descent above,
    //so the pdf stays consistent with selection (unbiased) post origin-snap.
    BlasRangeGpu Rng = gLT_Range[blas];
    const float3 xL  = mul(Rng.worldToLocal, float4(x, 1.0)).xyz;
    const float3 nL  = normalize(mul((float3x3)Rng.worldToLocal, n));
    float pdfBLAS    = 1.0f;
    uint  bnode      = 0;
    uint  bdepth     = 0;

    [loop] for (uint iterB = 0u; iterB < 64u; ++iterB)
    {
        LightBLASNodeGpu N = gLT_BLAS[Rng.nodeOffset + bnode];

        if (N.childCount == 0)
        {
            //fast path: single-tri leaf is the common case with maxLeafTris=1
            if (N.triCount <= 1u) {
                return pdfTLAS * pdfBLAS;
            }

            //fallback: power-weighted, matches LT_SampleLeafTriangle_Stratified
            const uint count    = N.triCount;
            const uint leafBase = Rng.triIndexOffset + N.triFirst;
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

        const uint childIdx = (triTrail >> (2u * bdepth)) & 3u;

        float w[4]; float sum = 0.0;
        [unroll] for (uint i=0;i<4;i++){
            if (i < N.childCount) {
                LightBLASNodeGpu C = gLT_BLAS[Rng.nodeOffset + (N.firstChild + i)];
                w[i] = max(LT_NodeImportance_BLAS(C, xL, nL), 0.0);
                sum += w[i];
            } else {
                w[i] = 0.0;
            }
        }

        float p = (sum > 0.0f) ? (w[childIdx] / sum) : (1.0f / float(N.childCount));
        pdfBLAS *= p;
        bnode = N.firstChild + childIdx;
        bdepth++;
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
    result.emission = triData.emission * GLOBAL_EMISSION_STRENGTH;

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
