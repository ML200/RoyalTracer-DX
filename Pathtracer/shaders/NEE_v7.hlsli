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
uint pickAliasWave(inout uint waveSeed)
{
    uint idx = 0;
    if (WaveIsFirstLane())
        idx = pickAlias(waveSeed);        // alias table uses waveSeed
    return WaveReadLaneFirst(idx);
}

LightTriangle LoadLightWave(uint idx)
{
    LightTriangle tri;
    if (WaveIsFirstLane()) tri = g_EmissiveTriangles[idx];

    tri.x          = WaveReadLaneFirst(tri.x);
    tri.y          = WaveReadLaneFirst(tri.y);
    tri.z          = WaveReadLaneFirst(tri.z);
    tri.weight     = WaveReadLaneFirst(tri.weight);
    tri.instanceID = WaveReadLaneFirst(tri.instanceID);
    return tri;
}

float4x4 LoadMatrixWave(uint instID)
{
    float4x4 M;
    if (WaveIsFirstLane()) M = instanceProps[instID].objectToWorld;
    [unroll] for (int r = 0; r < 4; ++r)
        M[r] = WaveReadLaneFirst(M[r]);
    return M;
}


// Sample a NEE sample
SampleReturn SampleNEE(
    SampleData sdata,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    // Pick a random light id using alias table
    uint idx = pickAliasWave(waveSeed);
    LightTriangle sampleLight = g_EmissiveTriangles[idx];

    // Calculate the current world coordinates of the triangle
    float4x4 conversionMatrix = instanceProps[sampleLight.instanceID].objectToWorld;
    float3 x_v = mul(conversionMatrix, float4(sampleLight.x, 1.f)).xyz;
    float3 y_v = mul(conversionMatrix, float4(sampleLight.y, 1.f)).xyz;
    float3 z_v = mul(conversionMatrix, float4(sampleLight.z, 1.f)).xyz;

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
    float pdf_l = sampleLight.weight / max(area_l, EPSILON);

    float2 probs = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1);
    float pdf0 = BRDF_PDF(0, sdata.matID, sdata.n1, -L_norm, sdata.o);
    float pdf1 = BRDF_PDF(1, sdata.matID, sdata.n1, -L_norm, sdata.o);
    float P1 = SafeMultiplyScalar(probs.x, pdf0);
    float P2 = SafeMultiplyScalar(probs.y, pdf1);
    float pdf_b = P1 + P2;

    // Fill in the sample and return
    SampleReturn sreturn;
    sreturn.x2 = x2;
    sreturn.n2 = normal_l;
    sreturn.objID = sampleLight.instanceID;
    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}