//====================================
//TRACE PAYLOAD AND HIT INFO
//====================================
//raygen uses SER HitObject path, stubs still need a nominal payload
struct [raypayload] TracePayload
{
    uint dummy : read(caller) : write(caller);
};

//no-medium sentinel for raygen absorption
static const uint MEDIUM_INVALID = 0xFFFFFFFFu;

struct HitInfo {
    float3 hitPos;
    float3 hitNormal;
    bool   backface;
    uint   lightID;
    float2 uv;
};


#ifdef ENABLE_RAY_QUERY_INLINE


//====================================
//RAY ORIGIN OFFSET, RTG CH 6
//====================================
//ULP-aware offset for self-intersection avoidance
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

//inline alpha test for shadow/visibility traversal - mirrors AnyHit.hlsl's
//AlphaTestAnyHit. Returns true when the candidate triangle is OPAQUE at the
//hit UV (so it must occlude the ray). Hardware OMMs normally resolve this in
//traversal, but they are inactive in the current test scene, so the
//visibility ray walks the regular alpha-tested geometry and runs this per
//non-opaque candidate.
inline bool AlphaCandidateOccludes(uint instID, uint primID, float2 bary)
{
#if DISABLE_ALPHA_TEST
    //TEMP: alpha testing off - every candidate occludes (geometry is opaque).
    return true;
#else
    const uint matID = materialIDs[instanceProps[instID].materialBase + primID];
    const int  texID = LoadAlbedoTexID(matID);

    //no albedo texture -> fully opaque
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

    //flip when the sampled channel is transparency (1=transparent) instead of
    //opacity - set by the loader heuristics or the editor override.
    if (LoadInvertAlpha(matID)) alpha = 1.0f - alpha;

    return alpha >= LoadAlphaThreshold(matID);
#endif
}


inline bool IsVisible(float3 A, float3 nA, float3 B, float3 nB)
{
    const float3 link = B - A;
    const float3 oA = offset_ray(A, dot( link, nA) >= 0.0f ? nA : -nA);
    const float3 oB = offset_ray(B, dot(-link, nB) >= 0.0f ? nB : -nB);

    const float3 conn = oB - oA;

    //offsets pushed inward, if they met or crossed the link sign flips
    //surfaces are within an offset epsilon, treat them as mutually visible
    if (dot(conn, link) <= 0.0f) return true;

    const float dist = length(conn);

    //sign held but the points still merged, nothing left between to occlude
    if (dist <= EPSILON) return true;

    const float3 direction = conn / dist;

    if (!IsRayValid(oA, direction, dist)) return false;

    RayDesc ray;
    ray.Origin    = oA;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = dist*0.998f;

    //FORCE_OPAQUE dropped: alpha-tested geometry now resolves per-texel below
    //so foliage/fences/etc. don't cast solid shadows. Opaque geometry still
    //auto-commits and ends the search (ACCEPT_FIRST_HIT); only non-opaque
    //triangles surface as candidates for the inline alpha test.
    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    //bounded walk guards against pathological alpha stacks tripping the TDR
    [loop]
    for (uint i = 0u; q.Proceed() && i < 128u; ++i)
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            const uint cInstID = q.CandidateInstanceID();   // == instanceProps index
            const uint cPrimID = FlatPrimID(cInstID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            if (AlphaCandidateOccludes(cInstID, cPrimID, q.CandidateTriangleBarycentrics()))
                q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

//====================================
//THIN-GLASS SHADOW TRANSMITTANCE
//====================================
//World-space geometric normal of a candidate triangle (flat face normal, no normal map —
//cheap, sufficient for the per-pane Fresnel). Mirrors EvalSurfaceState's geoNorm transform.
inline float3 CandidateGeoNormalW(uint instID, uint primID)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;
    const float3x3 R = (float3x3)instanceProps[instID].objectToWorld;
    float3 nW = mul(R, cross(p1 - p0, p2 - p0));
    return nW * rsqrt(max(dot(nW, nW), 1e-20f));
}

//Per-pane shadow-ray transmittance for a thin-glass candidate: (1-F)*Tf. F is the dielectric
//Fresnel at the pane (air->glass, real Ni) for the shadow direction; FresnelDielectric uses
//|cos| so the geo-normal sign is irrelevant.
inline float3 ThinGlassShadowTr(uint matID, uint instID, uint primID, float3 dir)
{
    const float3 nW = CandidateGeoNormalW(instID, primID);
    const float  Ni = LoadNi(matID);
    const float  F  = FresnelDielectric(-dir, nW, 1.0f, Ni).x;
    return (1.0f - F) * LoadTf(matID);
}

//====================================
//VISIBILITY TRANSMITTANCE  (RGB)
//====================================
//Generalises IsVisible: returns the product of (1-F)*Tf over every thin-glass pane crossed,
//0 if any opaque / alpha-cutout occluder blocks, 1 if fully clear. ACCEPT_FIRST_HIT is kept,
//so opaque-blocked rays end immediately (zero regression for non-thin scenes) — only thin-glass
//candidates accumulate and let traversal continue. This is the single visibility model used by
//every NEE + ReSTIR resampling target so the target functions stay consistent (unbiased).
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
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
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
                //attenuate and pass: do NOT commit, so traversal walks on through the pane.
                tr *= ThinGlassShadowTr(cMatID, cInstID, cPrimID, direction);
            }
            else if (LoadKd_w(cMatID) < 1.0f - EPSILON)
            {
                //solid (non-thin) glass is non-opaque geometry only so thin glass can be
                //intercepted here; it still FULLY occludes the shadow ray (behaviour unchanged
                //from when it was opaque). Commit regardless of any albedo cutout.
                q.CommitNonOpaqueTriangleHit();
            }
            else if (AlphaCandidateOccludes(cInstID, cPrimID, q.CandidateTriangleBarycentrics()))
            {
                q.CommitNonOpaqueTriangleHit();   // opaque at this texel -> blocks, ends search
            }
        }
    }
    return (q.CommittedStatus() == COMMITTED_NOTHING) ? tr : 0.0.xxx;
}

//====================================
//DEFERRED RECONNECTION VISIBILITY
//====================================
//One reconnection shadow ray for a resolved reservoir sample: x1 -> x2 (or, for an
//env miss, a far ray along the stored sky direction). Used by the spatial resolve
//when RS_FLAG_NO_REUSE_VIS has moved the shadow ray out of the reuse passes, so the
//stored F is unshadowed and needs its visibility applied exactly once. Mirrors the
//env-miss / surface split the temporal & spatial passes use inline.
inline float3 ReconnectVis(float3 x1, float3 n1_s, uint matID, float3 x2, float3 n2_s)
{
    if (matID == MATID_ENV_MISS)
    {
        const float3 md = normalize(x2);
        return VisibilityTransmittance(x1, n1_s, x1 + md * RAY_TMAX_PLANET, -md);
    }
    return VisibilityTransmittance(x1, n1_s, x2, n2_s);
}

//====================================
//NORMAL CLAMP TO VIEW AND REFLECTION
//====================================
inline float3 ClampNormalToViewAndReflection(float3 N, float3 V, float3 Ng, float epsView, float epsRefl)
{
    float3 Vn  = normalize(V);
    float3 Nn  = normalize(N);
    float3 NGn = normalize(Ng);

    //make sure the surface is hittable by V
    float a = dot(Nn, Vn);
    if (a < epsView)
    {
        float3 Nperp = Nn - a * Vn;
        float  len2  = dot(Nperp, Nperp);

        //degenerate, N ~ +/-V
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

    //quick reflection test
    {
        float3 R = reflect(-Vn, Nn);
        if (dot(R, NGn) >= epsRefl)
            return Nn;
    }

    //ensure reflected ray has coverage above normal
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

    //intersect with [0, thetaMax]
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

//====================================
//TEXTURE EVALUATION
//====================================
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

float2 EvaluatePBRProperties(uint matID, float2 uv, uint level)
{
    //FORCE_DIFFUSE: the RMA texture would re-introduce textured Pr/Pm over the
    //decoder-forced constants — return them directly (and skip the fetch).
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

//raygen passes depth to drop detail on deeper bounces
inline void RefetchMaterial(uint matID, float2 uv, out float3 localKd, out float localPr, out float localPm, uint level = 0)
{
    localKd = EvaluateAlbedo(matID, uv, level);
    float2 pbr = EvaluatePBRProperties(matID, uv, level);
    localPr = pbr.x;
    localPm = pbr.y;
}

//====================================
//CUSTOM TRACE RAY
//====================================
inline dx::HitObject TraceRay_Custom(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF)
{

    TracePayload payload = (TracePayload)0;
    //RayContribution=0, MultiplierForGeometry=1 (opaque vs alpha hit group per geometry), MissIndex=0
    dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, rayFlags, instanceMask, 0, 1, 0, ray, payload);

    //7-bit coherence hint (was 1 bit hit/miss). Top bit keeps the hit/miss split —
    //misses go straight to the fat sky/cloud eval — and the low 6 bits sort hits by
    //instance ID so a warp shades same-instance hits together: EvalSurfaceState's
    //index/vertex/transform gathers and RefetchMaterial's texture fetches become
    //warp-coherent instead of taking the hit population's random instance mix.
    //InstanceID comes off the hit record (no memory fetch before the reorder), and
    //reordering is execution-order only, so the output is bit-identical.
    const uint hint = hitObj.IsHit() ? (0x40u | (hitObj.GetInstanceID() & 0x3Fu)) : 0u;
    dx::MaybeReorderThread(hitObj, hint, 7);
    return hitObj;
}


//====================================
//MISS EVALUATION
//====================================
inline float3 EvalMissState(float3 rayDir, float3 sunDisk)
{
    return EvaluateSky(rayDir);
}

//====================================
//SURFACE STATE EVALUATION
//====================================
HitInfo EvalSurfaceState(
    uint   instID,
    uint   primID,
    float2 bc2,
    float3 origin,
    uint   level
)
{
    //data gather
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

    //local geometry
    const float b1 = bc2.x;
    const float b2 = bc2.y;
    const float b0 = 1.0f - b1 - b2;

    const float3 p_local = p0 * b0 + p1 * b1 + p2 * b2;
    const float2 uv      = uv0 * b0 + uv1 * b1 + uv2 * b2;

    //face normal, degeneracy test
    const float3 e1_local = p1 - p0;
    const float3 e2_local = p2 - p0;
    const float3 faceN_un = cross(e1_local, e2_local);
    const float  faceLen2 = dot(faceN_un, faceN_un);

    const float3 flatN_obj =
        (faceLen2 > 1e-20f) ? (faceN_un * rsqrt(faceLen2)) : float3(0.0f, 0.0f, 1.0f);

    //shading normal: octahedral vertex normals with degenerate/flip guards,
    //barycentric-interpolated.
    float3 n_local;
    {
        float3 n0 = UnpackNormal_INT(pn0);
        float3 n1 = UnpackNormal_INT(pn1);
        float3 n2 = UnpackNormal_INT(pn2);

        //replace degenerate/flipped normals with flat
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

        //force same hemisphere as geometric normal
        if (dot(n_local, flatN_obj) < 0.0f)
        {
            n_local = n_local - 2.0f * dot(n_local, flatN_obj) * flatN_obj;
            n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));
        }
    }

    //transform
    float3 posW;
    float3 normW;
    float3 geoNormW;
    float3 tangentW_geom;

    {
        const float3x4 M = instanceProps[instID].objectToWorld;
        const float3x3 R = (float3x3)M;

        posW     = mul(M, float4(p_local, 1.0f));
        normW    = mul(R, n_local);
        normW   *= rsqrt(max(dot(normW, normW), 1e-20f));

        geoNormW = mul(R, flatN_obj);
        geoNormW *= rsqrt(max(dot(geoNormW, geoNormW), 1e-20f));

        //tangent from edges + UV deltas
        const float2 dUV1 = uv1 - uv0;
        const float2 dUV2 = uv2 - uv0;

        const float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
        const float invDet = (abs(det) > 1e-8f) ? rcp(det) : 0.0f;

        const float3 tanO = (e1_local * dUV2.y - e2_local * dUV1.y) * invDet;

        tangentW_geom = mul(R, tanO);
        tangentW_geom *= rsqrt(max(dot(tangentW_geom, tangentW_geom), 1e-20f));
    }

    //material and texturing
    HitInfo hit = (HitInfo)0.0f;
    hit.uv = uv;

    //normal mapping
    const int normalTexID = LoadNormalTexID(materialID);
    [branch]
    if (normalTexID != -1)
    {
        const float2 normalUV = uv * LoadNormalUVScale(materialID);

        //Gram-Schmidt tangent against shading normal
        float3 tangentW = tangentW_geom - dot(tangentW_geom, normW) * normW;
        tangentW *= rsqrt(max(dot(tangentW, tangentW), 1e-20f));

        const float3 bitangentW = cross(normW, tangentW);

        Texture2D<float4> nTex = ResourceDescriptorHeap[normalTexID];
        const float3 n_tan =
            SampleMaterialTex(nTex, normalUV, level).xyz * 2.0f - 1.0f;

        //apply TBN without materializing float3x3
        normW = n_tan.x * tangentW + n_tan.y * bitangentW + n_tan.z * normW;
        normW *= rsqrt(max(dot(normW, normW), 1e-20f));
    }

    //finalize
    float3 viewDir = posW - origin;
    viewDir *= rsqrt(max(dot(viewDir, viewDir), 1e-20f));

    const bool   isBackface      = (dot(viewDir, geoNormW) > 0.0f);
    const float3 geoNormOriented = isBackface ? -geoNormW : geoNormW;

    hit.hitPos    = posW;
    hit.hitNormal = isBackface ? -normW : normW;
    hit.backface  = isBackface;

    //clamp normal so ray can proceed
    {
        const float3 Vw = -viewDir;
        hit.hitNormal = ClampNormalToViewAndReflection(hit.hitNormal, Vw, geoNormOriented, 0.005f, 0.02f);
    }

    const uint baseL        = instanceProps[instID].triToLightBase;
    const uint frontLightID = gTriToLightId[baseL + primID];
    hit.lightID = isBackface ? 0xFFFFFFFFu : frontLightID;

    return hit;
}


//====================================
//FAST EMISSION AND MATERIAL LOOKUPS
//====================================
inline float3 GetEmissionFast(in uint instID, in uint primID)
{
    uint base = instanceProps[instID].triToLightBase;
    uint lightID = gTriToLightId[base + primID];
    if (lightID == 0xFFFFFFFF)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return g_EmissiveTriangles[lightID].emission * GLOBAL_EMISSION_STRENGTH;
}

inline uint GetMatIDFast(in uint instID, in uint primID){
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;
    return materialIDs[baseM + primID];
}

#include "SurfaceVertex_v8.hlsli"

#endif
