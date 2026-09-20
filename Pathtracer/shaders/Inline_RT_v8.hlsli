struct [raypayload] TracePayload
{
    uint   flags  : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float  pdf    : read(caller, closesthit, miss) : write(caller, closesthit, miss);
    float  spread : read(caller, closesthit)       : write(caller, closesthit);
    uint   dirPk  : read(caller)                   : write(caller, closesthit);
    uint   nPk    : read(caller)                   : write(caller, closesthit);
    float3 color  : read(caller)                   : write(caller, closesthit, miss);
    uint   auxPk  : read(caller)                   : write(caller, closesthit, miss);
};

static const uint MEDIUM_INVALID = 0xFFFFFFFFu;

// Material context of a path vertex, shared by the passes that shade one.
struct HitContext {
    float3 hitPos;
    float3 hitNormal;
    uint   matID;
    uint   instID;
    bool   backface;
    half3  hitLocalKd;
    half   hitLocalPr;
    half   hitLocalPm;
    half2  iors;
    uint   mediumMatID;
    half3  absorptionTint;
};

struct HitInfo {
    float3 hitPos;
    float3 hitNormal;

    float3 rawNormal;
    float3 geometricNormal;
    bool   backface;
    uint   lightID;
    float2 uv;
};


static const float RTG_ORIGIN      = 1.0f / 32.0f;
static const float RTG_FLOAT_SCALE = 1.0f / 65536.0f;
static const float RTG_INT_SCALE   = 256.0f;

inline float3 offset_ray(float3 p, float3 n)
{
    int3 of_i = int3(RTG_INT_SCALE * n.x, RTG_INT_SCALE * n.y, RTG_INT_SCALE * n.z);
    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return float3(
        abs(p.x) < RTG_ORIGIN ? p.x + RTG_FLOAT_SCALE * n.x : p_i.x,
        abs(p.y) < RTG_ORIGIN ? p.y + RTG_FLOAT_SCALE * n.y : p_i.y,
        abs(p.z) < RTG_ORIGIN ? p.z + RTG_FLOAT_SCALE * n.z : p_i.z);
}

inline bool IsRayValid(float3 origin, float3 direction, float tMax)
{
    if (any(isnan(direction)) || any(isinf(direction))) return false;
    if (any(isnan(origin))    || any(isinf(origin)))    return false;
    const float d2 = dot(direction, direction);
    if (d2 < 0.25f || d2 > 4.0f) return false;
    if (tMax <= 1e-4f) return false;
    if (any(abs(origin) > 5.0e7f)) return false;
    return true;
}

inline bool AlphaCandidateOccludes(uint instID, uint primID, float2 bary)
{
    const uint matID = materialIDs[instanceProps[instID].materialBase + primID];
    const int  texID = LoadAlbedoTexID(matID);

    if (texID < 0) return true;

    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];

    const float2 uv0 = (float2)BTriVertex[i0].texCoord;
    const float2 uv1 = (float2)BTriVertex[i1].texCoord;
    const float2 uv2 = (float2)BTriVertex[i2].texCoord;

    const float  b0 = 1.0f - bary.x - bary.y;
    const float2 uv = uv0 * b0 + uv1 * bary.x + uv2 * bary.y;

    Texture2D<float4> tex = ResourceDescriptorHeap[texID];
    float alpha = SampleMaterialTex(tex, uv * LoadAlbedoUVScale(matID), 0).a;

    if (LoadInvertAlpha(matID)) alpha = 1.0f - alpha;

    return alpha >= LoadAlphaThreshold(matID);
}

inline bool IsVisible(float3 A, float3 nA, float3 B, float3 nB)
{
    const float3 link = B - A;
    const float3 oA = offset_ray(A, dot( link, nA) >= 0.0f ? nA : -nA);
    const float3 oB = offset_ray(B, dot(-link, nB) >= 0.0f ? nB : -nB);

    const float3 conn = oB - oA;

    if (dot(conn, link) <= 0.0f) return true;

    const float dist = length(conn);

    if (dist <= EPSILON) return true;

    const float3 direction = conn / dist;

    if (!IsRayValid(oA, direction, dist)) return false;

    RayDesc ray;
    ray.Origin    = oA;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = dist*0.998f;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    [loop]
    for (uint i = 0u; q.Proceed() && i < 128u; ++i)
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            const uint cInstID = q.CandidateInstanceID();
            const uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            if (AlphaCandidateOccludes(cInstID, cPrimID, q.CandidateTriangleBarycentrics()))
                q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

inline float3 CandidateGeoNormalW(uint instID, uint primID)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;
    const float3x3 N = (float3x3)instanceProps[instID].objectToWorldNormal;
    float3 nW = mul(N, cross(p1 - p0, p2 - p0));
    return nW * rsqrt(max(dot(nW, nW), 1e-20f));
}

inline float3 ThinGlassShadowTr(uint matID, uint instID, uint primID, float3 dir)
{
    const float3 nW = CandidateGeoNormalW(instID, primID);
    const float  Ni = LoadNi(matID);
    const float  F  = FresnelDielectric(-dir, nW, 1.0f, Ni).x;
    return (1.0f - F) * LoadTf(matID);
}

inline float3 VisibilityTransmittance(float3 A, float3 nA, float3 B, float3 nB)
{
    const float3 link = B - A;
    const float3 oA = offset_ray(A, dot( link, nA) >= 0.0f ? nA : -nA);
    const float3 oB = offset_ray(B, dot(-link, nB) >= 0.0f ? nB : -nB);

    const float3 conn = oB - oA;
    if (dot(conn, link) <= 0.0f) return 1.0.xxx;

    const float dist = length(conn);
    if (dist <= EPSILON) return 1.0.xxx;

    const float3 direction = conn / dist;
    if (!IsRayValid(oA, direction, dist)) return 0.0.xxx;

    RayDesc ray;
    ray.Origin    = oA;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = dist * 0.998f;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    float3 tr = 1.0.xxx;
    [loop]
    for (uint i = 0u; q.Proceed() && i < 128u; ++i)
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            const uint cInstID = q.CandidateInstanceID();
            const uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            const uint cMatID  = materialIDs[instanceProps[cInstID].materialBase + cPrimID];

            if (LoadIsThinGlass(cMatID))
            {

                tr *= ThinGlassShadowTr(cMatID, cInstID, cPrimID, direction);
            }
            else if (LoadKd_w(cMatID) < 1.0f - EPSILON)
            {

                const float3 nW = CandidateGeoNormalW(cInstID, cPrimID);
                tr *= 1.0f - FresnelDielectric(-direction, nW, 1.0f, LoadNi(cMatID)).x;
            }
            else if (AlphaCandidateOccludes(cInstID, cPrimID, q.CandidateTriangleBarycentrics()))
            {
                q.CommitNonOpaqueTriangleHit();
            }
        }
    }
    return (q.CommittedStatus() == COMMITTED_NOTHING) ? tr : 0.0.xxx;
}


inline float3 ClampNormalToViewAndReflection(float3 N, float3 V, float3 Ng, float epsView, float epsRefl)
{
    float3 Vn  = normalize(V);
    float3 Nn  = normalize(N);
    float3 NGn = normalize(Ng);

    float a = dot(Nn, Vn);
    if (a < epsView)
    {
        float3 Nperp = Nn - a * Vn;
        float  len2  = dot(Nperp, Nperp);

        float3 u_any;
        {
            float3 t = (abs(Vn.x) > 0.5f) ? float3(-Vn.y, Vn.x, 0.0f) : float3(0.0f, -Vn.z, Vn.y);
            u_any = normalize(cross(Vn, t));
        }
        float3 u = (len2 < 1e-12f) ? u_any : (Nperp * rsqrt(len2));

        float s_view = sqrt(saturate(1.0f - epsView * epsView));
        Nn = normalize(s_view * u + epsView * Vn);
    }

    if (dot(Nn, NGn) < 0.0f)
        Nn = normalize(Nn - 2.0f * dot(Nn, NGn) * NGn);

    {
        float3 R = reflect(-Vn, Nn);
        if (dot(R, NGn) >= epsRefl)
            return Nn;
    }

    float  c     = dot(Vn, NGn);
    float3 Ngp   = NGn - c * Vn;
    float  ngp2  = dot(Ngp, Ngp);
    float  m     = (ngp2 > 0.0f) ? rsqrt(ngp2) * ngp2 : 0.0f;
    float3 u;
    if (ngp2 > 1e-16f) {
        u = Ngp * rsqrt(ngp2);
    } else {
        float3 t = (abs(Vn.x) > 0.5f) ? float3(-Vn.y, Vn.x, 0.0f) : float3(0.0f, -Vn.z, Vn.y);
        u = normalize(cross(Vn, t));
    }

    a = saturate(dot(Nn, Vn));
    float thetaStart = acos(a);
    float thetaMax   = acos(saturate(epsView));

    float alpha = atan2(sqrt(saturate(1.0f - c*c)), c);
    float delta = acos(clamp(epsRefl, -1.0f, 1.0f));

    float L = 0.5f * (alpha - delta);
    float U = 0.5f * (alpha + delta);

    float Lc = max(0.0f, L);
    float Uc = min(thetaMax, U);

    float thetaTarget;
    if (Lc <= Uc)
    {
        thetaTarget = clamp(thetaStart, Lc, Uc);
    }
    else
    {
        thetaTarget = 0.0f;
    }

    float aT = cos(thetaTarget);
    float sT = sqrt(saturate(1.0f - aT * aT));
    float3 Nopt = aT * Vn + sT * u;

    if (dot(Nopt, NGn) < 0.0f)
        Nopt = normalize(Nopt - 2.0f * dot(Nopt, NGn) * NGn);
    else
        Nopt = normalize(Nopt);

    return Nopt;
}

// Sample and sanitize material albedo at the requested ray level.
float3 EvaluateAlbedo(uint matID, float2 uv, uint level)
{
    float3 albedo = LoadKd_rgb(matID);
    const int texID = LoadAlbedoTexID(matID);
    if (texID != -1)
    {
        float2 albedoUV = uv * LoadAlbedoUVScale(matID);
        Texture2D<float4> tex = ResourceDescriptorHeap[texID];
        albedo = SampleMaterialTex(tex, albedoUV, level).rgb;
    }
    return albedo;
}

// Fetch roughness and metalness with material-specific texture rules.
float2 EvaluatePBRProperties(uint matID, float2 uv, uint level)
{

    if (FORCE_DIFFUSE)
        return float2(1.0f, 0.0f);

    const float4 pbr4 = LoadPrPmPsPc(matID);
    float2 pbrProps = pbr4.xy;

    const int rmaID = LoadRmaTexID(matID);
    if (rmaID != -1)
    {
        float2 rmaUV = uv * LoadRmaUVScale(matID);
        Texture2D<float4> tex = ResourceDescriptorHeap[rmaID];
        float4 rmaSample = SampleMaterialTex(tex, rmaUV, level);

        pbrProps.x = rmaSample.g;
        pbrProps.y = rmaSample.b;
    }
    return pbrProps;
}

inline void RefetchMaterial(uint matID, float2 uv, out float3 localKd, out float localPr, out float localPm, uint level = 0)
{
    localKd = EvaluateAlbedo(matID, uv, level);
    float2 pbr = EvaluatePBRProperties(matID, uv, level);
    localPr = pbr.x;
    localPm = pbr.y;
}

inline dx::HitObject TraceRay_Custom(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF,
    uint lowHint = 0u,
    uint lowHintBits = 0u)
{

    TracePayload payload = (TracePayload)0;

    dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, rayFlags, instanceMask, 0, 1, 0, ray, payload);

    const uint hint = ((hitObj.IsHit() ? (0x40u | (hitObj.GetInstanceID() & 0x3Fu)) : 0u) << lowHintBits) | lowHint;
    dx::MaybeReorderThread(hitObj, hint, 7u + lowHintBits);
    return hitObj;
}

inline float3 EvalMissState(float3 rayDir, float3 sunDisk)
{
    return EvaluateSky(rayDir);
}

inline uint LightRecordOf(uint instID, uint primID)
{
    const uint base = instanceProps[instID].triToLightBase;
    if (base == 0xFFFFFFFFu) return 0xFFFFFFFFu;
    if (base & 0x80000000u)
    {
        const uint records = base & 0x7FFFFFFFu;
        const uint litOpaque = instanceProps[instID]._pad[0];
        const uint litAlpha  = instanceProps[instID]._pad[1];
        const uint opaque    = instanceProps[instID].opaqueTriCount;
        if (primID < litOpaque) return records + primID;
        if (primID >= opaque && primID < opaque + litAlpha) return records + litOpaque + (primID - opaque);
        return 0xFFFFFFFFu;
    }
    return gTriToLightId[base + primID];
}

// Interpolate hit attributes and derive the complete shading state.
HitInfo EvalSurfaceStateImpl(
    uint   instID,
    uint   primID,
    float2 bc2,
    float3 originOrDir,
    bool   viewIsDir,
    uint   level
)
{

    const uint baseI      = instanceProps[instID].indexBase;
    const uint baseM      = instanceProps[instID].materialBase;
    const uint materialID = materialIDs[baseM + primID];

    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];

    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;

    const float2 uv0 = (float2)BTriVertex[i0].texCoord;
    const float2 uv1 = (float2)BTriVertex[i1].texCoord;
    const float2 uv2 = (float2)BTriVertex[i2].texCoord;

    const uint pn0 = BTriVertex[i0].packedNormal;
    const uint pn1 = BTriVertex[i1].packedNormal;
    const uint pn2 = BTriVertex[i2].packedNormal;

    const float b1 = bc2.x;
    const float b2 = bc2.y;
    const float b0 = 1.0f - b1 - b2;

    const float3 p_local = p0 * b0 + p1 * b1 + p2 * b2;
    const float2 uv      = uv0 * b0 + uv1 * b1 + uv2 * b2;

    const float3 e1_local = p1 - p0;
    const float3 e2_local = p2 - p0;
    const float3 faceN_un = cross(e1_local, e2_local);
    const float  faceLen2 = dot(faceN_un, faceN_un);

    const float3 flatN_obj =
        (faceLen2 > 1e-20f) ? (faceN_un * rsqrt(faceLen2)) : float3(0.0f, 0.0f, 1.0f);

    float3 n_local;
    {
        float3 n0 = UnpackNormal_INT(pn0);
        float3 n1 = UnpackNormal_INT(pn1);
        float3 n2 = UnpackNormal_INT(pn2);

        {
            const float n0l2 = dot(n0, n0);
            if (n0l2 < EPSILON) n0 = flatN_obj;
            else {
                const float inv0 = rsqrt(n0l2);
                if (dot(n0 * inv0, flatN_obj) <= 0.4f) n0 = flatN_obj;
            }

            const float n1l2 = dot(n1, n1);
            if (n1l2 < EPSILON) n1 = flatN_obj;
            else {
                const float inv1 = rsqrt(n1l2);
                if (dot(n1 * inv1, flatN_obj) <= 0.4f) n1 = flatN_obj;
            }

            const float n2l2 = dot(n2, n2);
            if (n2l2 < EPSILON) n2 = flatN_obj;
            else {
                const float inv2 = rsqrt(n2l2);
                if (dot(n2 * inv2, flatN_obj) <= 0.4f) n2 = flatN_obj;
            }
        }

        n_local = n0 * b0 + n1 * b1 + n2 * b2;
        n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));

        if (dot(n_local, flatN_obj) < 0.0f)
        {
            n_local = n_local - 2.0f * dot(n_local, flatN_obj) * flatN_obj;
            n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));
        }
    }

    float3 posW;
    float3 normW;
    float3 geoNormW;
    float3 tangentW_geom;
    float3 bitangentW_geom;

    {
        const float3x4 M = instanceProps[instID].objectToWorld;
        const float3x3 R = (float3x3)M;
        const float3x3 N = (float3x3)instanceProps[instID].objectToWorldNormal;

        posW     = mul(M, float4(p_local, 1.0f));
        normW    = mul(N, n_local);
        normW   *= rsqrt(max(dot(normW, normW), 1e-20f));

        geoNormW = mul(N, flatN_obj);
        geoNormW *= rsqrt(max(dot(geoNormW, geoNormW), 1e-20f));

        const float2 dUV1 = uv1 - uv0;
        const float2 dUV2 = uv2 - uv0;

        const float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
        const float invDet = (abs(det) > 1e-8f) ? rcp(det) : 0.0f;

        const float3 tanO = (e1_local * dUV2.y - e2_local * dUV1.y) * invDet;
        const float3 bitanO = (e2_local * dUV1.x - e1_local * dUV2.x) * invDet;

        tangentW_geom = mul(R, tanO);
        tangentW_geom *= rsqrt(max(dot(tangentW_geom, tangentW_geom), 1e-20f));
        bitangentW_geom = mul(R, bitanO);
    }

    HitInfo hit = (HitInfo)0.0f;
    hit.uv = uv;

    const int normalTexID = LoadNormalTexID(materialID);
    [branch]
    if (normalTexID != -1)
    {
        const float2 normalUV = uv * LoadNormalUVScale(materialID);

        float3 tangentW = tangentW_geom - dot(tangentW_geom, normW) * normW;
        tangentW *= rsqrt(max(dot(tangentW, tangentW), 1e-20f));

        float3 bitangentW = cross(normW, tangentW);

        if (dot(bitangentW, bitangentW_geom) < 0.0f) bitangentW = -bitangentW;

        Texture2D<float4> nTex = ResourceDescriptorHeap[normalTexID];
        const float3 n_tan =
            SampleMaterialTex(nTex, normalUV, level).xyz * 2.0f - 1.0f;

        normW = n_tan.x * tangentW + n_tan.y * bitangentW + n_tan.z * normW;
        normW *= rsqrt(max(dot(normW, normW), 1e-20f));
    }

    float3 viewDir = viewIsDir ? originOrDir : (posW - originOrDir);
    viewDir *= rsqrt(max(dot(viewDir, viewDir), 1e-20f));

    const bool   isBackface      = (dot(viewDir, geoNormW) > 0.0f);
    const float3 geoNormOriented = isBackface ? -geoNormW : geoNormW;

    hit.hitPos    = posW;
    hit.hitNormal = isBackface ? -normW : normW;
    hit.rawNormal = hit.hitNormal;
    hit.geometricNormal = geoNormOriented;
    hit.backface  = isBackface;

    {
        const float3 Vw = -viewDir;
        hit.hitNormal = ClampNormalToViewAndReflection(hit.hitNormal, Vw, geoNormOriented, 0.005f, 0.02f);
    }

    const uint frontLightID = LightRecordOf(instID, primID);
    hit.lightID = isBackface ? 0xFFFFFFFFu : frontLightID;

    return hit;
}

// Evaluate a surface from a ray origin and barycentric hit.
HitInfo EvalSurfaceState(uint instID, uint primID, float2 bc2, float3 origin, uint level)
{
    return EvalSurfaceStateImpl(instID, primID, bc2, origin, false, level);
}

HitInfo EvalSurfaceStateDir(uint instID, uint primID, float2 bc2, float3 rayDir, uint level)
{
    return EvalSurfaceStateImpl(instID, primID, bc2, rayDir, true, level);
}

inline float3 GetEmissionFast(in uint instID, in uint primID)
{
    const uint lightID = LightRecordOf(instID, primID);
    if (lightID == 0xFFFFFFFF) return float3(0.0f, 0.0f, 0.0f);
    return g_EmissiveTriangles[lightID].emission * GLOBAL_EMISSION_STRENGTH;
}

inline uint GetMatIDFast(in uint instID, in uint primID){
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;
    return materialIDs[baseM + primID];
}

#include "SurfaceVertex_v8.hlsli"

