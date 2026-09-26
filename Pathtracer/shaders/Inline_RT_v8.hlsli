// Carries no data; one field keeps the struct valid.
struct [raypayload] TracePayload
{
    uint unused : read(caller) : write(caller);
};

static const uint MEDIUM_INVALID = 0xFFFFFFFFu;

// Per-raygen reorder toggles (e.g. PT_SER_REORDER=0 via RDN_SHADER_DEFINES).
#ifndef CAMERA_SER_REORDER
#define CAMERA_SER_REORDER 1
#endif
#ifndef PT_SER_REORDER
#define PT_SER_REORDER 1
#endif
#ifndef SHARC_SER_REORDER
#define SHARC_SER_REORDER 1
#endif

// Material context of a path vertex.
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
    float  uvFootprint;   // beam width at the hit, in UV units

    // Ocean state from the cascades (roughness depends on the footprint).
    bool   isOcean;
    float3 oceanKd;
    float  oceanPr;
    uint   oceanMaterialOffset;
    // For the reconstruction's albedo guide (OceanSurface).
    float  oceanFoam;
    float  oceanBubbles;
};

uint ResolveSurfaceMaterial(uint matID, HitInfo hit)
{
    return hit.isOcean ? OceanParams().materialBase +
        hit.oceanMaterialOffset : matID;
}


// Self-intersection offset (Waechter & Binder 2019, Ray Tracing Gems).
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

// Bad rays can hang the GPU. Integer tests: fast-math may drop float NaN/Inf checks.
static const float RAY_ORIGIN_LIMIT = 5.0e7f;

inline bool IsRayDescValid(RayDesc r)
{
    const uint3 origin    = asuint(r.Origin)    & 0x7FFFFFFFu;
    const uint3 direction = asuint(r.Direction) & 0x7FFFFFFFu;
    const uint  tMin      = asuint(r.TMin);
    const uint  tMax      = asuint(r.TMax);
    if (any(origin > asuint(RAY_ORIGIN_LIMIT))) return false;   // NaN, Inf, or outside the scene
    if (any(direction > asuint(2.0f))) return false;            // NaN, Inf, or far from unit length
    if (tMin > 0x7F7FFFFFu || tMax > 0x7F7FFFFFu) return false;  // negative, NaN or Inf extents
    if (tMax <= tMin) return false;                             // inverted or empty extents
    // All finite from here, so the float test is safe.
    const float d2 = dot(r.Direction, r.Direction);
    return d2 >= 0.25f && d2 <= 4.0f;
}

inline bool IsRayValid(float3 origin, float3 direction, float tMax)
{
    RayDesc r;
    r.Origin = origin; r.Direction = direction; r.TMin = 0.0f; r.TMax = tMax;
    return IsRayDescValid(r) && tMax > 1e-4f;
}

// An invalid ray becomes an empty-mask ray, i.e. a plain miss. No reorder here.
inline dx::HitObject TraceRayChecked(RaytracingAccelerationStructure bvh, uint rayFlags, uint instanceMask, RayDesc ray)
{
    if (!IsRayDescValid(ray))
    {
        ray.Origin    = float3(0.0f, 0.0f, 0.0f);
        ray.Direction = float3(0.0f, 0.0f, 1.0f);
        ray.TMin      = 0.0f;
        ray.TMax      = 1.0f;
        instanceMask  = 0u;
    }
    TracePayload payload = (TracePayload)0;
    return dx::HitObject::TraceRay(bvh, rayFlags, instanceMask, 0, 1, 0, ray, payload);
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

    RayDesc ray;
    ray.Origin    = oA;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = dist*0.998f;
    // Endpoints closer than TMin touch.
    if (ray.TMax <= ray.TMin) return true;
    if (!IsRayDescValid(ray)) return false;

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

// Previous minus current displacement at a stitched vertex.
float3 OceanVertexMotion(uint instID, float2 uv) {
    const float size = asfloat(instanceProps[instID]._pad[1]);
    const uint stitch = instanceProps[instID]._pad[0];
    const uint2 ij = (uint2)round(uv * OCEAN_TILE_GRID);
    const OceanVertexSource src = OceanStitchSource(ij.x, ij.y, stitch);
    const float step = size / OCEAN_TILE_GRID;
    const float2 origin = float2(instanceProps[instID].objectToWorld[0][3], instanceProps[instID].objectToWorld[2][3]);
    const OceanParamsGPU P = OceanParams();
    const uint current = OceanDispSlot(P), previous = OceanPrevDispSlot(P);
    // Must match the vertex build (Ocean_Tiles_v8.hlsl).
    const float2 qLo = origin + float2(src.lo) * step;
    const float wLo = OceanGeometryWidth(qLo, P);
    float3 delta = OceanDisplacementFrom(qLo, wLo, previous) - OceanDisplacementFrom(qLo, wLo, current);
    if (src.w > 0.0f) {
        const float2 qHi = origin + float2(src.hi) * step;
        const float wHi = OceanGeometryWidth(qHi, P);
        delta = lerp(delta, OceanDisplacementFrom(qHi, wHi, previous) - OceanDisplacementFrom(qHi, wHi, current), src.w);
    }
    return delta;
}
float3 OceanPreviousHit(uint instID, uint primID, float2 bary, float3 hitPos) {
    const uint base = instanceProps[instID].indexBase + 3u * primID;
    const float3 a = OceanVertexMotion(instID, (float2)BTriVertex[indices[base]].texCoord);
    const float3 b = OceanVertexMotion(instID, (float2)BTriVertex[indices[base+1]].texCoord);
    const float3 c = OceanVertexMotion(instID, (float2)BTriVertex[indices[base+2]].texCoord);
    // Current origin frame; prevView is already rebased (Camera::PollSceneOrigin).
    return hitPos + a * (1-bary.x-bary.y) + b * bary.x + c * bary.y;
}

inline float3 VisibilityTransmittance(float3 A, float3 nA, float3 B, float3 nB,
    bool explicitWater = false, bool waterAttenuation = true)
{
    const float3 link = B - A;
    const float3 oA = offset_ray(A, dot( link, nA) >= 0.0f ? nA : -nA);
    const float3 oB = offset_ray(B, dot(-link, nB) >= 0.0f ? nB : -nB);

    const float3 conn = oB - oA;
    if (dot(conn, link) <= 0.0f) return 1.0.xxx;

    const float dist = length(conn);
    if (dist <= EPSILON) return 1.0.xxx;

    const float3 direction = conn / dist;

    RayDesc ray;
    ray.Origin    = oA;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = dist * 0.998f;
    // Endpoints closer than TMin touch.
    if (ray.TMax <= ray.TMin) return 1.0.xxx;
    if (!IsRayDescValid(ray)) return 0.0.xxx;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    float3 tr = explicitWater || !waterAttenuation ? 1.0f.xxx : OceanShadowTransmittance(oA, oB);
    [loop]
    for (uint i = 0u; q.Proceed() && i < 128u; ++i)
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            const uint cInstID = q.CandidateInstanceID();
            const uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            const uint cMatID  = materialIDs[instanceProps[cInstID].materialBase + cPrimID];

            if (explicitWater && LoadIsOceanMaterial(cMatID)) continue;

            if (LoadIsThinGlass(cMatID))
            {

                tr *= ThinGlassShadowTr(cMatID, cInstID, cPrimID, direction);
            }
            else if (LoadKd_w(cMatID) < 1.0f - EPSILON)
            {
                const float3 nW = CandidateGeoNormalW(cInstID, cPrimID);
                // Crossing direction sets the IOR ratio; past the critical angle, TIR blocks it.
                const float ior = max(LoadNi(cMatID), 1.0f);
                const bool leaving = dot(direction, nW) > 0.0f;
                tr *= 1.0f - FresnelDielectricTIR(-direction, nW, leaving ? ior : 1.0f,
                                                  leaving ? 1.0f : ior).x;
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

float PixelConeAngle() { return 2.0f / max(projection._m11 * float(IMG_H), 1e-6f); }

float TexFootprintLod(Texture2D<float4> tex, float uvFootprint)
{
    uint w, h;
    tex.GetDimensions(w, h);
    return log2(max(uvFootprint * float(max(w, h)), 1e-8f)) + PT_TEXTURE_LOD_BIAS;
}

float3 EvaluateAlbedo(uint matID, float2 uv, float uvFootprint)
{
    float3 albedo = LoadKd_rgb(matID);
    const int texID = LoadAlbedoTexID(matID);
    if (texID != -1)
    {
        const float2 scale = LoadAlbedoUVScale(matID);
        Texture2D<float4> tex = ResourceDescriptorHeap[texID];
        albedo = SampleMaterialTex(tex, uv * scale, TexFootprintLod(tex, uvFootprint * max(scale.x, scale.y))).rgb;
    }
    return albedo;
}

// Returns (roughness, metalness).
float2 EvaluatePBRProperties(uint matID, float2 uv, float uvFootprint)
{

    if (FORCE_DIFFUSE)
        return float2(1.0f, 0.0f);

    const float4 pbr4 = LoadPrPmPsPc(matID);
    float2 pbrProps = pbr4.xy;

    const int rmaID = LoadRmaTexID(matID);
    if (rmaID != -1)
    {
        const float2 scale = LoadRmaUVScale(matID);
        Texture2D<float4> tex = ResourceDescriptorHeap[rmaID];
        float4 rmaSample = SampleMaterialTex(tex, uv * scale, TexFootprintLod(tex, uvFootprint * max(scale.x, scale.y)));

        pbrProps.x = rmaSample.g;
        pbrProps.y = rmaSample.b;
    }
    return pbrProps;
}

inline void RefetchMaterialUV(uint matID, float2 uv, out float3 localKd, out float localPr, out float localPm, float uvFootprint = 0.0f)
{
    localKd = EvaluateAlbedo(matID, uv, uvFootprint);
    float2 pbr = EvaluatePBRProperties(matID, uv, uvFootprint);
    localPr = pbr.x;
    localPm = pbr.y;
}

// Ocean: scene material, roughness floored by the footprint-filtered ripple spread.
inline void RefetchMaterial(uint matID, HitInfo hit, out float3 localKd, out float localPr, out float localPm)
{
    [branch]
    if (hit.isOcean)
    {
        if (OceanParams().debugMode != 0u)
        {
            localKd = hit.oceanKd;
            localPr = hit.oceanPr;
            localPm = 0.0f;
            return;
        }
        RefetchMaterialUV(matID, hit.uv, localKd, localPr, localPm, hit.uvFootprint);
        localPr = max(localPr, hit.oceanPr);
        return;
    }
    RefetchMaterialUV(matID, hit.uv, localKd, localPr, localPm, hit.uvFootprint);
}

inline dx::HitObject TraceRay_Custom(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF,
    uint lowHint = 0u,
    uint lowHintBits = 0u)
{
    dx::HitObject hitObj = TraceRayChecked(SceneBVH, rayFlags, instanceMask, ray);

#if CAMERA_SER_REORDER
    const uint hint = ((hitObj.IsHit() ? (0x40u | (hitObj.GetInstanceID() & 0x3Fu)) : 0u) << lowHintBits) | lowHint;
    dx::MaybeReorderThread(hitObj, hint, 7u + lowHintBits);
#endif
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

// footprint: beam width at the hit, world units; selects texture mips.
HitInfo EvalSurfaceStateImpl(
    uint   instID,
    uint   primID,
    float2 bc2,
    float3 originOrDir,
    bool   viewIsDir,
    float  footprint
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
    float  uvPerWorld;   // UV per world unit

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
        const float3 e1w = mul(R, e1_local);
        const float3 e2w = mul(R, e2_local);
        uvPerWorld = sqrt(abs(det) / max(length(cross(e1w, e2w)), 1e-20f));

        const float3 tanO = (e1_local * dUV2.y - e2_local * dUV1.y) * invDet;
        const float3 bitanO = (e2_local * dUV1.x - e1_local * dUV2.x) * invDet;

        tangentW_geom = mul(R, tanO);
        tangentW_geom *= rsqrt(max(dot(tangentW_geom, tangentW_geom), 1e-20f));
        bitangentW_geom = mul(R, bitanO);
    }

    HitInfo hit = (HitInfo)0.0f;
    hit.uv = uv;

    float3 viewDir = viewIsDir ? originOrDir : (posW - originOrDir);
    viewDir *= rsqrt(max(dot(viewDir, viewDir), 1e-20f));
    // Grazing angles stretch the beam.
    hit.uvFootprint = footprint * uvPerWorld / max(abs(dot(viewDir, geoNormW)), 0.05f);

    [branch]
    if (IS_OCEAN_INSTANCE(instID))
    {
        // Area-equivalent grazing footprint.
        const float cosV = max(abs(dot(viewDir, geoNormW)), 1e-3f);
        const float width = footprint * rsqrt(cosV);

        // Sample at the undisplaced UV position; posW.xz would shift off the crests.
        const float tileSize = asfloat(instanceProps[instID]._pad[1]);
        const float2 tileOrigin = float2(instanceProps[instID].objectToWorld[0][3],
                                         instanceProps[instID].objectToWorld[2][3]);
        const float2 oceanPosition = tileOrigin + uv * tileSize;
        const OceanSurface sea = OceanEvalSurface(oceanPosition, width, uv, tileSize);

        normW = sea.normal;
        // Keep it on the hit triangle's side; cascades and mesh are filtered differently.
        if (dot(normW, geoNormW) < 0.0f)
            normW = normalize(normW - 2.0f * dot(normW, geoNormW) * geoNormW);

        hit.isOcean = true;
        hit.oceanKd = sea.albedo;
        hit.oceanPr = sea.roughness;
        hit.oceanMaterialOffset = sea.materialOffset;
        hit.oceanFoam = sea.foam;
        hit.oceanBubbles = sea.bubbles;
    }

    const int normalTexID = LoadNormalTexID(materialID);
    [branch]
    if (normalTexID != -1)
    {
        const float2 normalScale = LoadNormalUVScale(materialID);
        const float2 normalUV    = uv * normalScale;

        float3 tangentW = tangentW_geom - dot(tangentW_geom, normW) * normW;
        tangentW *= rsqrt(max(dot(tangentW, tangentW), 1e-20f));

        float3 bitangentW = cross(normW, tangentW);

        if (dot(bitangentW, bitangentW_geom) < 0.0f) bitangentW = -bitangentW;

        Texture2D<float4> nTex = ResourceDescriptorHeap[normalTexID];
        const float3 n_tan = SampleMaterialTex(nTex, normalUV,
            TexFootprintLod(nTex, hit.uvFootprint * max(normalScale.x, normalScale.y))).xyz * 2.0f - 1.0f;

        normW = n_tan.x * tangentW + n_tan.y * bitangentW + n_tan.z * normW;
        normW *= rsqrt(max(dot(normW, normW), 1e-20f));
    }

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

HitInfo EvalSurfaceState(uint instID, uint primID, float2 bc2, float3 origin, float footprint = 0.0f)
{
    return EvalSurfaceStateImpl(instID, primID, bc2, origin, false, footprint);
}

HitInfo EvalSurfaceStateDir(uint instID, uint primID, float2 bc2, float3 rayDir, float footprint = 0.0f)
{
    return EvalSurfaceStateImpl(instID, primID, bc2, rayDir, true, footprint);
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
