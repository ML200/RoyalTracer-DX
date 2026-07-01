//====================================
//RANDOM-WALK SUBSURFACE SCATTERING
//====================================
//Brute-force volumetric walk inside an SSS-flagged mesh. The surface is a LAYERED
//reflect-or-enter material (see Pass_raygen_v8.hlsl): the regular BRDF reflects, and
//with probability pEnter the continuation enters the medium and runs this walk.
//
//Made reconnection-safe: the walk runs only during initial sampling. For a primary
//hit the FIRST in-medium scatter S1 becomes the reconnection vertex (a volume vertex
//reconnected like a transmission segment); everything past it is the reused suffix.
//Reconnect (Reservoir_v8.hlsli) re-evaluates the Henyey-Greenstein phase + Beer-Lambert
//transmittance on the straight x1->S1 segment and never re-runs this walk.
//
//Homogeneous medium, scalar extinction sigma_t = 1/radius; the RGB colour is carried by
//the per-scatter single-scattering albedo product. A no-scatter pass-through (thin area
//or large radius) transmits diffusely with W=1 — it must NOT be killed (that darkens
//thin regions).

#ifndef MATERIAL_SSS_V8_HLSLI
#define MATERIAL_SSS_V8_HLSLI

//Bounds: an open / non-watertight SSS mesh or a near-zero extinction must not spin the
//loop. RR kills low-throughput walks before the step cap.
static const uint  SSS_MAX_STEPS  = 64u;
static const float SSS_MIN_RADIUS = 1e-3f;   //floor on the scalar mean free path
static const float SSS_RR_FLOOR   = 0.05f;   //throughput below this enters RR
static const float SSS_INV_PI     = 0.31830988618f;  //diffuse entry/exit coupling 1/pi

struct SSSWalkResult {
    float3 entryDir;        //first step direction INTO the medium (entry-surface V2)
    float3 firstScatterPos; //S1 — the volume reconnection vertex (world)
    float3 firstScatterDir; //outgoing dir at S1 (toward S2/B); the volume V2
    float3 wRest;           //product of albedos for scatters AFTER S1 (= albedo^(n-1))
    float3 wTotal;          //product of all albedos (= albedo^n); 1 when no scatter
    float3 exitPos;         //boundary exit point B (world)
    float3 exitNormal;      //outward shading normal at B
    uint   nScatters;       //number of in-medium scatter events (0 = pass-through)
    bool   valid;           //false => escaped / absorbed / step-cap (NOT no-scatter)
};

//Self-contained orthonormal basis (Duff et al. 2017) so the walk does not depend on
//SunSampler include order.
inline void SSS_OrthoBasis(float3 n, out float3 t, out float3 b)
{
    const float s = (n.z >= 0.0f) ? 1.0f : -1.0f;
    const float a = -1.0f / (s + n.z);
    const float c = n.x * n.y * a;
    t = float3(1.0f + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

//Henyey-Greenstein importance sample around the current flight direction wo.
inline float3 SampleHenyeyGreenstein(float3 wo, float g, inout uint seed)
{
    const float u1 = RandomFloatSingle(seed);
    const float u2 = RandomFloatSingle(seed);

    float cosT;
    if (abs(g) < 1e-3f) {
        cosT = 1.0f - 2.0f * u1;                          //isotropic
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

//Deterministic HG phase value p(cosTheta). cosTheta = dot(wIn, wOut) with both
//directions pointing AWAY from the vertex (the SampleHenyeyGreenstein convention:
//forward scattering for g>0 keeps the flight direction). Normalised to integrate to 1
//over the sphere (the 1/4pi is included). Reconnect needs this.
inline float EvaluatePhaseHG(float g, float cosTheta)
{
    const float gg    = g * g;
    const float denom = 1.0f + gg - 2.0f * g * cosTheta;
    return (1.0f - gg) / (4.0f * 3.14159265359f * max(denom * sqrt(max(denom, 1e-8f)), 1e-8f));
}

//Outward shading normal at the walk's boundary exit B. The re-emergence surface is forced
//white diffuse, so the full EvalSurfaceState (UV interp, tangent frame, normal-map sample,
//light-ID lookup, view-clamp) is wasted here — only the world shading normal is needed.
//`dir` is the flight direction leaving the medium; the result is oriented along it (outward).
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

    //degenerate interpolated normal -> fall back to the geometric face normal (object space)
    if (dot(n_local, n_local) < 1e-20f)
        n_local = cross(BTriVertex[i1].vertex - BTriVertex[i0].vertex,
                        BTriVertex[i2].vertex - BTriVertex[i0].vertex);

    const float3x3 R = (float3x3)instanceProps[instID].objectToWorld;
    float3 nW = mul(R, n_local);
    nW *= rsqrt(max(dot(nW, nW), 1e-20f));

    return (dot(nW, dir) > 0.0f) ? nW : -nW;
}

//March a ray inside the SSS object. Homogeneous: scalar sigma_t = 1/radius drives the
//analog free-flight, the RGB single-scatter albedo carries the colour (throughput *=
//albedo per scatter). The first scatter S1 is recorded separately so the caller can
//anchor the reconnection vertex there.
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

    //enter the medium: diffuse refraction approximated by a cosine lobe about the
    //inward normal (keeps the boundary rough so the entry coupling is a plain Lambertian,
    //matching the 1/pi entry term Reconnect uses).
    float3 dir = -CosineUnitVectorInHemisphere(entryNormal, seed);  //into the surface
    r.entryDir = dir;

    float3 pos    = offset_ray(entryPos, -entryNormal);
    float3 wTotal = float3(1, 1, 1);
    float3 wRest  = float3(1, 1, 1);

    [loop]
    for (uint step = 0u; step < SSS_MAX_STEPS; ++step)
    {
        if (!IsRayValid(pos, dir, 10000.0f)) return r;

        //analog free-flight distance to the next scatter event. Sampled BEFORE the trace so it
        //can bound the boundary search: "did we reach the wall before dl?" == "does a ray of
        //length dl hit?". The trace consumes no RNG, so the seed stream is byte-identical.
        const float u  = RandomFloatSingle(seed);
        const float dl = -log(max(1.0f - u, 1e-6f)) / sigma_t;

        RayDesc ray;
        ray.Origin    = pos;
        ray.Direction = dir;
        ray.TMin      = 0.0001f;
        ray.TMax      = dl;        //bound to the free flight: dense media cull almost immediately

        //closest-hit traversal bounded by dl. FORCE_OPAQUE treats alpha-cutout triangles as
        //solid in fixed-function (so a cutout can't punch a false hole in the boundary) AND
        //skips any-hit, so no Proceed() loop is needed. Assumes a closed/watertight mesh: an
        //open mesh no longer reports "escaped" here, it terminates via RR / step cap instead.
        RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
        q.TraceRayInline(SceneBVH, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
        q.Proceed();

        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            //reached the boundary within dl: the nearest hit inside a closed mesh is its wall.
            //Only the outward world shading normal is needed (white-diffuse re-emergence).
            const float tHit  = q.CommittedRayT();
            const uint instID = q.CommittedInstanceID();
            const uint primID = FlatPrimID(instID, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());

            r.exitPos    = pos + dir * tHit;
            r.exitNormal = SSS_ExitNormal(instID, primID, q.CommittedTriangleBarycentrics(), dir);
            r.wRest      = wRest;
            r.wTotal     = wTotal;
            //Reaching the boundary is ALWAYS a valid exit. nScatters==0 (thin area / large
            //radius) is a diffuse pass-through (W=1) and must transmit, not die.
            r.valid      = true;
            return r;
        }

        //no boundary within dl -> scatter inside the medium at dl
        pos += dir * dl;
        const float3 newDir = SampleHenyeyGreenstein(dir, g, seed);

        if (r.nScatters == 0u) {
            r.firstScatterPos = pos;
            r.firstScatterDir = newDir;
            //report the GEOMETRIC entry direction A->S1 so it matches Reconnect's
            //reconstructed -ndirN = normalize(S1-x1) exactly (self-reconnect identity).
            r.entryDir = normalize(r.firstScatterPos - entryPos);
        } else {
            wRest *= albedo;          //albedos AFTER the first scatter -> suffix
        }
        wTotal *= albedo;             //every scatter -> full-path throughput
        r.nScatters++;
        dir = newDir;

        //Russian roulette on the brightest channel once the walk dims out. The boost
        //rides BOTH throughputs so the split estimator stays unbiased.
        const float p = max(wTotal.x, max(wTotal.y, wTotal.z));
        if (p < SSS_RR_FLOOR)
        {
            if (RandomFloatSingle(seed) > p) return r;   //absorbed
            const float inv = 1.0f / max(p, 1e-4f);
            wTotal *= inv;
            wRest  *= inv;
        }
    }

    return r;   //step cap hit -> treat as absorbed (valid stays false)
}

#endif
