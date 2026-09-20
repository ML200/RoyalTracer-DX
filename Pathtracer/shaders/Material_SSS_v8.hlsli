#pragma once

static const uint  SSS_MAX_STEPS  = 64u;
static const float SSS_MIN_RADIUS = 1e-3f;
static const float SSS_RR_FLOOR   = 0.05f;
static const float SSS_INV_PI     = 0.31830988618f;

struct SSSWalkResult {
    float3 entryDir;
    float3 firstScatterPos;
    float3 firstScatterDir;
    float3 wRest;
    float3 wTotal;
    float3 exitPos;
    float3 exitNormal;
    uint   nScatters;
    bool   valid;
};

inline void SSS_OrthoBasis(float3 n, out float3 t, out float3 b)
{
    const float s = (n.z >= 0.0f) ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Sample a local direction from the HG phase distribution.
inline float3 SampleHenyeyGreenstein(float3 wo, float g, inout uint seed)
{
    const float u1 = RandomFloatSingle(seed);
    const float u2 = RandomFloatSingle(seed);

    float cosT;
    if (abs(g) < 1e-3f) {
        cosT = 1.0f - 2.0f * u1;
    } else {
        const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cosT = (1.0f + g * g - s * s) / (2.0f * g);
    }

    const float sinT = sqrt(max(0.0f, 1.0f - cosT * cosT));
    const float phi  = 6.28318530718f * u2;

    float3 T, B;
    SSS_OrthoBasis(wo, T, B);
    return normalize(sinT * cos(phi) * T + sinT * sin(phi) * B + cosT * wo);
}

// Evaluate the Henyey-Greenstein phase function.
inline float EvaluatePhaseHG(float g, float cosTheta)
{
    const float gg    = g * g;
    const float denom = 1.0f + gg - 2.0f * g * cosTheta;
    return (1.0f - gg) / (4.0f * 3.14159265359f * max(denom * sqrt(max(denom, 1e-8f)), 1e-8f));
}

inline float3 SSS_ExitNormal(uint instID, uint primID, float2 bc, float3 dir)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];

    const float b1 = bc.x;
    const float b2 = bc.y;
    const float b0 = 1.0f - b1 - b2;

    float3 n_local = UnpackNormal_INT(BTriVertex[i0].packedNormal) * b0
                   + UnpackNormal_INT(BTriVertex[i1].packedNormal) * b1
                   + UnpackNormal_INT(BTriVertex[i2].packedNormal) * b2;

    if (dot(n_local, n_local) < 1e-20f)
        n_local = cross(BTriVertex[i1].vertex - BTriVertex[i0].vertex,
                        BTriVertex[i2].vertex - BTriVertex[i0].vertex);

    const float3x3 R = (float3x3)instanceProps[instID].objectToWorld;
    float3 nW = mul(R, n_local);
    nW *= rsqrt(max(dot(nW, nW), 1e-20f));

    return (dot(nW, dir) > 0.0f) ? nW : -nW;
}

inline SSSWalkResult SubsurfaceWalk(
    float3 entryPos, float3 entryNormal,
    uint matID, inout uint seed)
{
    SSSWalkResult r;
    r.entryDir        = -entryNormal;
    r.firstScatterPos = entryPos;
    r.firstScatterDir = -entryNormal;
    r.wRest           = float3(1, 1, 1);
    r.wTotal          = float3(1, 1, 1);
    r.exitPos         = entryPos;
    r.exitNormal      = entryNormal;
    r.nScatters       = 0u;
    r.valid           = false;

    const float3 albedo  = saturate(LoadSSSAlbedo(matID));
    const float  radius  = max(LoadSSSRadius(matID), SSS_MIN_RADIUS);
    const float  g       = LoadPhaseG(matID);
    const float  sigma_t = 1.0f / radius;

    float3 dir = -CosineUnitVectorInHemisphere(entryNormal, seed);
    r.entryDir = dir;

    float3 pos    = offset_ray(entryPos, -entryNormal);
    float3 wTotal = float3(1, 1, 1);
    float3 wRest  = float3(1, 1, 1);

    [loop]
    for (uint step = 0u; step < SSS_MAX_STEPS; ++step)
    {
        if (!IsRayValid(pos, dir, 10000.0f)) return r;

        const float u  = RandomFloatSingle(seed);
        const float dl = -log(max(1.0f - u, 1e-6f)) / sigma_t;

        RayDesc ray;
        ray.Origin    = pos;
        ray.Direction = dir;
        ray.TMin      = 0.0001f;
        ray.TMax      = dl;

        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {

            const float tHit  = q.CommittedRayT();
            const uint instID = q.CommittedInstanceID();
            const uint primID = FlatPrimID(instID, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());

            r.exitPos    = pos + dir * tHit;
            r.exitNormal = SSS_ExitNormal(instID, primID, q.CommittedTriangleBarycentrics(), dir);
            r.wRest      = wRest;
            r.wTotal     = wTotal;

            r.valid      = true;
            return r;
        }

        pos += dir * dl;
        const float3 newDir = SampleHenyeyGreenstein(dir, g, seed);

        if (r.nScatters == 0u) {
            r.firstScatterPos = pos;
            r.firstScatterDir = newDir;

            r.entryDir = normalize(r.firstScatterPos - entryPos);
        } else {
            wRest *= albedo;
        }
        wTotal *= albedo;
        r.nScatters++;
        dir = newDir;

        const float p = max(wTotal.x, max(wTotal.y, wTotal.z));
        if (p < SSS_RR_FLOOR)
        {
            if (RandomFloatSingle(seed) > p) return r;
            const float inv = 1.0f / max(p, 1e-4f);
            wTotal *= inv;
            wRest  *= inv;
        }
    }

    return r;
}

