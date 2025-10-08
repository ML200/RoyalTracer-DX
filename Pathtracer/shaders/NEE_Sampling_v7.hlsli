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

uint pickAliasWave(inout uint waveSeed, inout uint2 threadSeed)
{
    return pickAlias(threadSeed.x);
}

// Sample NEE in area measrure
SampleReturn SampleNEE(
    SampleData sdata,
    inout uint  waveSeed,
    inout uint2 threadSeed)
{
    LT_Sample pick = LT_SampleLight(sdata.x1, sdata.n1, threadSeed.x);
    uint triID = pick.id;

    uint   inst = g_EmissiveTriangles[triID].instanceID;
    float3 X = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].x, 1)).xyz;
    float3 Y = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].y, 1)).xyz;
    float3 Z = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].z, 1)).xyz;

    float u = RandomFloatSingle(threadSeed.x);
    float v = RandomFloatSingle(threadSeed.x);
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    float3 x2 = (1.0f - u - v) * X + u * Y + v * Z;

    float3 e1 = Y - X, e2 = Z - X;
    float3 n  = cross(e1, e2);
    float  A  = 0.5f * length(n);
    float3 nL = n / max(2.0f*A, EPSILON);

    float3 L    = x2 - sdata.x1;
    float  d2   = dot(L, L);
    float3 Ldir = L * rsqrt(max(d2, EPSILON));

    if (dot(nL, -Ldir) < 0.0f) return (SampleReturn)0;

    float pdf_nee_area  = pick.pdf / max(A, EPSILON);

    /*float2 probs = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1);
    float pdf0   = BRDF_PDF(0, sdata.matID, sdata.n1, -Ldir, sdata.o);
    float pdf1   = BRDF_PDF(1, sdata.matID, sdata.n1, -Ldir, sdata.o);*/
    float pdf_bsdf_area = BRDF_PDF_COMBINED(sdata.matID, sdata.n1, -Ldir, sdata.o)
                        * max(dot(nL, -Ldir), 0.0f) / max(d2, EPSILON);

    // Pack
    SampleReturn sreturn;
    sreturn.x2       = x2;
    sreturn.n2       = nL;
    sreturn.L2       = g_EmissiveTriangles[triID].emission;
    sreturn.objID    = inst;
    sreturn.pdf_bsdf = pdf_bsdf_area;
    sreturn.pdf_nee  = pdf_nee_area;

    return sreturn;
}



// Sample a NEE sample (generalized for solid angle)
SampleReturn SampleNEE_gen(
    in float3 x1,
    in float3 n1,
    in uint matID1,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
) {
    // Pick a random emissive triangle
    LT_Sample pick = LT_SampleLight(x1, n1, threadSeed.x);
    uint lightIdx = pick.id;

    float xi1 = RandomFloatSingle(threadSeed.x);
    float xi2 = RandomFloatSingle(threadSeed.x);
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }
    float uu = 1.0f - xi1 - xi2;
    float vv = xi1;
    float ww = xi2;

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
    float3 normal_l = normal / (2.0f * area + EPSILON);

    if (dot(normal_l, -L_norm) < 0.0f) {
        return (SampleReturn)0;
    }

    // Compute NEE PDF (solid angle!)
    float pdf_l = pick.pdf / max(area, EPSILON) * dist2 / max(dot(normal_l, -L_norm), 0.0f);//g_EmissiveTriangles[lightIdx].weight / max(area, EPSILON) * dist2 / max(dot(normal_l, -L_norm), 0.0f);

    // Compute BSDF PDF (fro MIS)
    /*float2 probs = CalculateStrategyProbabilities(matID1, o, n1);
    float pdf0 = BRDF_PDF(0, matID1, n1, -L_norm, o);
    float pdf1 = BRDF_PDF(1, matID1, n1, -L_norm, o);*/
    float pdf_b = BRDF_PDF_COMBINED(matID1, n1, -L_norm, o);//(probs.x * pdf0 + probs.y * pdf1);

    SampleReturn sreturn;
    sreturn.x2 = x2;
    sreturn.n2 = normal_l;
    sreturn.L2 = g_EmissiveTriangles[lightIdx].emission;
    sreturn.objID = g_EmissiveTriangles[lightIdx].instanceID;
    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}

