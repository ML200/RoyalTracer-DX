// No shader stage carries data through the payload: the raygens read their hits from the hit
// object and shade them themselves. The one field keeps the struct valid.
struct [raypayload] TracePayload
{
    uint unused : read(caller) : write(caller);
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
    float  uvFootprint;   // width of the ray beam at the hit, in texture coordinates of the surface

    // The ocean resolves its own shading state from the wave cascades rather than from textures,
    // because the roughness it needs depends on the ray footprint. These carry that result past
    // the material fetch the other surfaces use.
    bool   isOcean;
    float3 oceanKd;
    float  oceanPr;
    uint   oceanMaterialOffset;
};

uint ResolveSurfaceMaterial(uint matID, HitInfo hit)
{
    return hit.isOcean ? OceanParams().materialBase +
        hit.oceanMaterialOffset : matID;
}


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

// Every ray handed to the hardware passes this check first, whether it goes through
// HitObject::TraceRay (TraceRayChecked below) or a RayQuery. NaN or Inf in any component, a
// direction far from unit length, an origin outside the representable scene, or extents that
// are inverted, negative or infinite make the traversal undefined and have hung the GPU before.
// Such a ray is never traced: the caller treats it as a miss (or as occluded) instead.
static const float RAY_ORIGIN_LIMIT = 5.0e7f;

inline bool IsRayDescValid(RayDesc r)
{
    if (any(isnan(r.Origin))    || any(isinf(r.Origin)))    return false;
    if (any(isnan(r.Direction)) || any(isinf(r.Direction))) return false;
    const float d2 = dot(r.Direction, r.Direction);
    if (!(d2 >= 0.25f && d2 <= 4.0f)) return false;
    if (!(r.TMin >= 0.0f)) return false;                   // also rejects NaN
    if (!(r.TMax > r.TMin) || isinf(r.TMax)) return false; // also rejects NaN and inverted extents
    if (any(abs(r.Origin) > RAY_ORIGIN_LIMIT)) return false;
    return true;
}

inline bool IsRayValid(float3 origin, float3 direction, float tMax)
{
    RayDesc r;
    r.Origin = origin; r.Direction = direction; r.TMin = 0.0f; r.TMax = tMax;
    return IsRayDescValid(r) && tMax > 1e-4f;
}

// The hit-object trace of every raygen. An invalid ray is swapped for a well-formed one with an
// empty instance mask, which the hardware answers with a miss, so the caller sees a plain miss
// instead of undefined traversal. No reorder happens here: each raygen keeps its single reorder
// point, and the closest-hit and miss shaders are never invoked.
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
    // Endpoints closer than the ray's own start offset touch: nothing fits between them.
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

// Previous/current displacement difference at a vertex of the current stitched topology.
float3 OceanVertexMotion(uint instID, float2 uv) {
    const float size = asfloat(instanceProps[instID]._pad[1]);
    const uint stitch = instanceProps[instID]._pad[0];
    const uint2 ij = (uint2)round(uv * OCEAN_TILE_GRID);
    const bool xEdge = (ij.x == 0 && (stitch & OCEAN_EDGE_NEG_X)) ||
                       (ij.x == OCEAN_TILE_GRID && (stitch & OCEAN_EDGE_POS_X));
    const bool zEdge = (ij.y == 0 && (stitch & OCEAN_EDGE_NEG_Z)) ||
                       (ij.y == OCEAN_TILE_GRID && (stitch & OCEAN_EDGE_POS_Z));
    const bool collapse = xEdge ? ((ij.y & 1u) != 0) : zEdge && ((ij.x & 1u) != 0);
    const float step = size / OCEAN_TILE_GRID;
    const float width = step * ((xEdge || zEdge) ? 2.0f : 1.0f);
    const float2 q = float2(instanceProps[instID].objectToWorld[0][3], instanceProps[instID].objectToWorld[2][3]) + uv * size;
    const float2 offset = collapse ? (xEdge ? float2(0,step) : float2(step,0)) : 0.0f;
    float3 delta = 0.0f;
    [unroll] for (uint i = 0; i < 2; ++i) {
        const float2 p = q + (i == 0 ? -offset : offset);
        delta += OceanDisplacementFrom(p, width, OCEAN_SRV_PREV_DISP) - OceanDisplacement(p, width);
    }
    return delta * 0.5f;
}
float3 OceanPreviousHit(uint instID, uint primID, float2 bary, float3 hitPos) {
    const uint base = instanceProps[instID].indexBase + 3u * primID;
    const float3 a = OceanVertexMotion(instID, (float2)BTriVertex[indices[base]].texCoord);
    const float3 b = OceanVertexMotion(instID, (float2)BTriVertex[indices[base+1]].texCoord);
    const float3 c = OceanVertexMotion(instID, (float2)BTriVertex[indices[base+2]].texCoord);
    // prevView is already rebased by Camera::PollSceneOrigin. Both positions must remain in
    // the CURRENT scene-origin frame; adding originDelta here would apply the rebase twice.
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
    // Endpoints closer than the ray's own start offset touch: nothing fits between them.
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
                // Which way the connection crosses the interface decides the index ratio, and with
                // it whether it can cross at all: past the critical angle a ray leaving water is
                // turned back entirely. Schlick on its own has no such angle, so it used to hand
                // the sun through a surface that should have reflected all of it - which is where
                // the isolated bright pixels below the surface came from.
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

// The angle one pixel of the render subtends; times the path length it is the beam width.
float PixelConeAngle() { return 2.0f / max(projection._m11 * float(IMG_H), 1e-6f); }

// The mip whose texels match a beam of the given width in texture coordinates.
float TexFootprintLod(Texture2D<float4> tex, float uvFootprint)
{
    uint w, h;
    tex.GetDimensions(w, h);
    return log2(max(uvFootprint * float(max(w, h)), 1e-8f)) + PT_TEXTURE_LOD_BIAS;
}

// Sample and sanitize material albedo for the beam width at the hit.
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

// Fetch roughness and metalness with material-specific texture rules.
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

// Beauty water uses the editable scene material, like other surfaces. Procedural ocean
// evaluation supplies its full-resolution wave normal. Direct lights filter their
// own highlight lobe; that filter must not enter the continuation or guide roughness.
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

// Interpolate hit attributes and derive the complete shading state. `footprint` is the width of
// the ray beam at the hit in world units; the textures are read at the mip that matches it.
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
    float  uvPerWorld;   // texture coordinates per world unit along the triangle

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
    // The beam stretches across the surface at grazing angles.
    hit.uvFootprint = footprint * uvPerWorld / max(abs(dot(viewDir, geoNormW)), 0.05f);

    [branch]
    if (IS_OCEAN_INSTANCE(instID))
    {
        // Area-equivalent grazing footprint, retained for diagnostics only.
        const float cosV = max(abs(dot(viewDir, geoNormW)), 1e-3f);
        const float width = footprint * rsqrt(cosV);

        // UVs retain the undisplaced material coordinates even on stitched tile edges.
        // Sampling at posW.xz would shift the normals/foam away from their own crests.
        const float tileSize = asfloat(instanceProps[instID]._pad[1]);
        const float2 tileOrigin = float2(instanceProps[instID].objectToWorld[0][3],
                                         instanceProps[instID].objectToWorld[2][3]);
        const float2 oceanPosition = tileOrigin + uv * tileSize;
        const OceanSurface sea = OceanEvalSurface(oceanPosition, width);

        normW = sea.normal;
        // Keep the shading normal on the visible side of the triangle the ray actually hit: the
        // cascades are filtered at a different width than the tessellation, so the two can
        // disagree by more than the interpolated normal ever would.
        if (dot(normW, geoNormW) < 0.0f)
            normW = normalize(normW - 2.0f * dot(normW, geoNormW) * geoNormW);

        hit.isOcean = true;
        hit.oceanKd = sea.albedo;
        hit.oceanPr = sea.roughness;
        hit.oceanMaterialOffset = sea.materialOffset;
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

// Evaluate a surface from a ray origin and barycentric hit.
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
