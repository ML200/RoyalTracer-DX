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
){
    // Pick a random light id using alias table
    uint idx = pickAliasWave(waveSeed, threadSeed);
    //LightTriangle sampleLight = g_EmissiveTriangles[idx];

    // Calculate the current world coordinates of the triangle
    float4x4 conversionMatrix = instanceProps[g_EmissiveTriangles[idx].instanceID].objectToWorld;
    float3 x_v = mul(conversionMatrix, float4(g_EmissiveTriangles[idx].x, 1.f)).xyz;
    float3 y_v = mul(conversionMatrix, float4(g_EmissiveTriangles[idx].y, 1.f)).xyz;
    float3 z_v = mul(conversionMatrix, float4(g_EmissiveTriangles[idx].z, 1.f)).xyz;

    // Generate random barycentric coordinates
    float xi1 = RandomFloatSingle(threadSeed.x);
    float xi2 = RandomFloatSingle(threadSeed.x);
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }
    float uu = 1.0f - xi1 - xi2;
    float vv = xi1;
    float ww = xi2;
    float3 x2 = uu * x_v + vv * y_v + ww * z_v;

    // Get the sample direction and compute distance
    float3 L = x2 - sdata.x1;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    float3 L_norm = normalize(L);

    // Compute the light's surface normal from triangle geometry
    float3 edge1 = y_v - x_v;
    float3 edge2 = z_v - x_v;
    float3 cross_l = cross(edge1, edge2);
    float3 normal_l = normalize(cross_l);

    if(dot(normal_l, -L_norm) < 0.0f){
        normal_l = -normal_l;
    }

    float area_l = abs(length(cross_l) * 0.5f);
    float pdf_l = g_EmissiveTriangles[idx].weight / max(area_l, EPSILON);

    float2 probs = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1);
    float pdf0 = BRDF_PDF(0, sdata.matID, sdata.n1, -L_norm, sdata.o);
    float pdf1 = BRDF_PDF(1, sdata.matID, sdata.n1, -L_norm, sdata.o);
    float P1 = SafeMultiplyScalar(probs.x, pdf0);
    float P2 = SafeMultiplyScalar(probs.y, pdf1);
    float cos_light = dot(normal_l, -L_norm);
    float pdf_b = (P1 + P2) * cos_light / dist2;


    // Fill in the sample and return
    SampleReturn sreturn;
    sreturn.x2 = x2;
    sreturn.n2 = normal_l;
    sreturn.L2 = g_EmissiveTriangles[idx].emission;
    sreturn.objID = g_EmissiveTriangles[idx].instanceID;
    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}