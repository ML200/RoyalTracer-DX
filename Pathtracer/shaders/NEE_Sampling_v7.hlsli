uint pickAlias(inout uint seed)
{
    float2 rnd;
    rnd.x = RandomFloatSingle(seed);
    rnd.y = RandomFloatSingle(seed);

    uint   N;
    uint   strideInBytes;
    g_AliasProb.GetDimensions(N, strideInBytes);
    uint i = (uint)(rnd.x * N);

    return (rnd.y < g_AliasProb[i]) ? i : g_AliasIdx[i];
}

// ---- wave-uniform helpers (lane 0 touches memory / RNG) ----
/*uint pickAliasWave(inout uint waveSeed)
{
    uint idx = 0;
    if (WaveIsFirstLane())
        idx = pickAlias(waveSeed);        // alias table uses waveSeed
    return WaveReadLaneFirst(idx);
}*/

//------------------------------------------
//  Wave-level alias-table pick (WAVE_CANDIDATES k variants)
//  – Each of the first k lanes generates
//    a light index with its local seed.
//  – Every lane then draws one of those k
//    indices at random.
//  – k should be a small power of two
//    (1, 2, 4, 8, 16, 32) for best warp occupancy.
//  – Powerful GPUs can use their full range for DI sampling, should be limited for GI at each vertex to ~4-8 tho
//------------------------------------------
/*uint pickAliasWave(inout uint waveSeed, inout uint2 threadSeed)
{
    // Safety: k shouldnt exceed the lane cound -> we cant have more samples than lanes]
    uint k = clamp(WAVE_CANDIDATES_DI, 1u, WaveGetLaneCount());

    const uint lane   = WaveGetLaneIndex();
    uint       idx_k  = 0;

    // Step 1: compute the light indices for the specified number of varienty. Importantly, use several lanes in parallel to maximise paralellism
    if (lane < k)
    {
        idx_k = pickAlias(threadSeed.x);
    }

    // Step 2: every lane chooses which candidate (0…k-1) to use randomly. This is unbiased as we drew the indices with replacement
    uint choice = (uint)(RandomFloatSingle(threadSeed.x) * k);

    // Step 3: fetch the chosen index from the generating lane
    uint idx = WaveReadLaneAt(idx_k, choice);

    return idx;
}*/

uint pickAliasWave(inout uint waveSeed, inout uint2 threadSeed)
{
    return pickAlias(threadSeed.x);;
}




// Sample a NEE sample
SampleReturn SampleNEE(
    SampleData sdata,
    inout uint waveSeed,
    inout uint2 threadSeed
) {
    // Pick a random emissive triangle
    uint lightIdx = pickAliasWave(waveSeed, threadSeed);

    // Generate barycentric coordinates early
    float xi1 = RandomFloatSingle(threadSeed.x);
    float xi2 = RandomFloatSingle(threadSeed.x);
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }
    float uu = 1.0f - xi1 - xi2;
    float vv = xi1;
    float ww = xi2;

    // Transform triangle vertices & compute sampled point immediately
    float3 x = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].x, 1.f)).xyz;
    float3 y = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].y, 1.f)).xyz;
    float3 z = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].z, 1.f)).xyz;

    float3 x2 = uu * x + vv * y + ww * z;

    // Compute sample direction and normalized vector
    float3 L = x2 - sdata.x1;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    float3 L_norm = L / dist;

    // Compute normal and area
    float3 edge1 = y - x;
    float3 edge2 = z - x;
    float3 normal = cross(edge1, edge2);
    float area = 0.5f * length(normal);
    float3 normal_l = normal / (2.0f * area + EPSILON); // = normalize(cross(...))

    // Flip normal if needed
    if (dot(normal_l, -L_norm) < 0.0f) {
        normal_l = -normal_l;
    }

    // Compute NEE PDF
    float pdf_l = g_EmissiveTriangles[lightIdx].weight / max(area, EPSILON);

    // Compute BSDF importance PDF
    float2 probs = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1);
    float pdf0 = BRDF_PDF(0, sdata.matID, sdata.n1, -L_norm, sdata.o);
    float pdf1 = BRDF_PDF(1, sdata.matID, sdata.n1, -L_norm, sdata.o);
    float pdf_b = (probs.x * pdf0 + probs.y * pdf1) * max(dot(normal_l, -L_norm), 0.0f) / dist2;

    // Pack results
    SampleReturn sreturn;
    sreturn.x2 = x2;
    sreturn.n2 = normal_l;
    sreturn.L2 = g_EmissiveTriangles[lightIdx].emission;
    sreturn.objID = g_EmissiveTriangles[lightIdx].instanceID;
    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}



// Sample a NEE sample
SampleReturn SampleNEE_gen(
    in float3 x1,
    in float3 n1,
    in uint matID1,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
) {
    // Pick a random emissive triangle
    uint lightIdx = pickAliasWave(waveSeed, threadSeed);

    // Generate barycentric coordinates early
    float xi1 = RandomFloatSingle(threadSeed.x);
    float xi2 = RandomFloatSingle(threadSeed.x);
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }
    float uu = 1.0f - xi1 - xi2;
    float vv = xi1;
    float ww = xi2;

    // Transform triangle vertices & compute sampled point immediately
    float3 x = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].x, 1.f)).xyz;
    float3 y = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].y, 1.f)).xyz;
    float3 z = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].z, 1.f)).xyz;

    float3 x2 = uu * x + vv * y + ww * z;

    // Compute sample direction and normalized vector
    float3 L = x2 - x1;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    float3 L_norm = L / dist;

    // Compute normal and area
    float3 edge1 = y - x;
    float3 edge2 = z - x;
    float3 normal = cross(edge1, edge2);
    float area = 0.5f * length(normal);
    float3 normal_l = normal / (2.0f * area + EPSILON); // = normalize(cross(...))

    // Flip normal if needed
    if (dot(normal_l, -L_norm) < 0.0f) {
        normal_l = -normal_l;
    }

    // Compute NEE PDF
    float pdf_l = g_EmissiveTriangles[lightIdx].weight / max(area, EPSILON);

    // Compute BSDF importance PDF
    float2 probs = CalculateStrategyProbabilities(matID1, o, n1);
    float pdf0 = BRDF_PDF(0, matID1, n1, -L_norm, o);
    float pdf1 = BRDF_PDF(1, matID1, n1, -L_norm, o);
    float pdf_b = (probs.x * pdf0 + probs.y * pdf1) * max(dot(normal_l, -L_norm), 0.0f) / dist2;

    // Pack results
    SampleReturn sreturn;
    sreturn.x2 = x2;
    sreturn.n2 = normal_l;
    sreturn.L2 = g_EmissiveTriangles[lightIdx].emission;
    sreturn.objID = g_EmissiveTriangles[lightIdx].instanceID;
    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}