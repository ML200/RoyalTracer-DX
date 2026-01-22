using namespace dx;

#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 3
#endif

#ifndef MEDIUM_INVALID_15
#define MEDIUM_INVALID_15 0x7FFFu
#endif

[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pix = DispatchRaysIndex().xy;

    // Reset output texture
    gScratchPing[uint3(pix, 1)] = float4(0, 0, 0, 0);
    // TODO: reservoir init
    uint idx1 = MapPixelID(float2(DispatchRaysDimensions().xy), pix);
    storeReservoirDI(g_Reservoirs_current_di, idx1, (Reservoir_DI)0);
    store_wsum_di(g_Reservoirs_current_di, idx1, 0.0f);
    store_W_di(g_Reservoirs_current_di, idx1, 0.0f);
    store_phat_di(g_Reservoirs_current_di, idx1, 0.0f);

    storeReservoirGI(g_Reservoirs_current_gi, idx1, (Reservoir_GI)0);
    store_wsum_gi(g_Reservoirs_current_gi, idx1, 0.0f);
    store_W_gi   (g_Reservoirs_current_gi, idx1, 0.0f);
    store_F_gi   (g_Reservoirs_current_gi, idx1, 0.0f);
    store_M_gi   (g_Reservoirs_current_gi, idx1, 0u);

    gScratchPing[uint3(DispatchRaysIndex().xy, 3)] = 0.0f;

    store_Tpost_gi(g_Reservoirs_current_gi, idx1, 1.0f);

    uint seed = initRandomData(pix, uint2(8, 4), time, 1u);

    float3 rayOrigin = InitOrigin();

    // Keep direction compressed across bounces: oct float2
    float3 dir0    = InitDirection(pix, float2(DispatchRaysDimensions().xy), seed);
    float2 rayDir2 = OctEncodeFloat2_payload(dir0);

    // Keep throughput compressed across bounces: RGB9E5
    uint packedThroughput = PackRGB9E5(float3(1.0f, 1.0f, 1.0f));

    // Keep previous normal compressed across bounces: octsnorm16x2
    uint prevNPacked = PackOctSnorm16_payload(float3(0.0f, 1.0f, 0.0f));

    // compressed
    VolumeIOR_Packed viorP;
    VolumeAux_Packed aiorP;

    {
        VolumeIOR v0 = InitVolumeIOR();
        VolumeAux a0 = InitVolumeAux();

        viorP.raw  = PackIORStackAndPtr(v0.ior_stack, v0.pointer);
        aiorP.mat16 = PackMatStack16(a0.matID_stack);
        aiorP.obj8  = PackPrioStack8(a0.objID_stack);
    }

    float prev_pdf = 1.0f;

    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        // Decode direction only for traversal work that needs float3
        float3 rayDir = OctDecodeFloat2_payload(rayDir2);

        // Safety validation (keep; if you want release perf, compile-guard this)
        float d2 = dot(rayDir, rayDir);
        bool badDir = any(isnan(rayDir)) || any(isinf(rayDir)) || (d2 < 1e-12f);
        bool badOrg = any(isnan(rayOrigin)) || any(isinf(rayOrigin));
        if (badDir || badOrg) break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;

        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (!hitObj.IsHit())
        {
            // Store sample data miss case
            if(depth == 0){
                SampleData sdata = (SampleData)0;
                sdata.L1 = EvalMissState();
                uint idx2 = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
                gScratchPing[uint3(pix, 1)] += float4(sdata.L1, 0);
                storeSampleData(g_sample_current, idx2, sdata);
                break;
            }
            // Unpack throughput only when needed
            float3 T = UnpackRGB9E5(packedThroughput);
            //gScratchPing[uint3(pix, 1)] += float4(T * EvalMissState(), 0);
            // TODO: UPDATE RESERVOIR HERE
            if(depth == 1){
                uint idx4 = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
                float p_hat = GetPHat(T * EvalMissState() * prev_pdf); // We need to remove the previous pdf; Cancel it my multiplying with it
                float wi = p_hat / prev_pdf;
                float3 dir = rayDir;
                bool update = UpdateReservoirDI_Infinite(g_Reservoirs_current_di, idx4, wi, dir, EvalMissState(), 0xFFFFFFFFu, seed);
                if(update)store_phat_di(g_Reservoirs_current_di, idx4, p_hat);
            }

            if (depth >= 2)
            {
                uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);

                float3 envL = EvalMissState();

                // Cancel only the immediately previous BSDF pdf (at x2), consistent with your existing DI pattern
                //gScratchPing[uint3(DispatchRaysIndex().xy, 3)] += float4(T * envL,0);
                float p_hat = GetPHat(T * envL);
                float wi    = p_hat;

                float3 V2_new = -rayDir; // incoming at x2 from x3->x2
                if(depth > 2) V2_new = load_Vpost_gi(g_Reservoirs_current_gi, idx_gi);

                float4 J_new = float4(0.0f, 1.0f, 0.0f, 0.0f);
                if(depth > 2) J_new = float4(0.0f, 0.0f, 0.0f, 0.0f);


                float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, idx_gi);
                bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, idx_gi,
                                                    wi,
                                                    envL * tpostgi, J_new, V2_new,
                                                    seed);

                if (update) store_F_gi(g_Reservoirs_current_gi, idx_gi, p_hat);
            }
            break;
        }

        // Pre-invoke setup
        const uint instID = hitObj.GetInstanceIndex();
        const uint primID = hitObj.GetPrimitiveIndex();

        uint matID = GetMatIDFast(instID, primID);

        // Packed volume IOR query (returns float2 in your packed implementation)
        float2 iorsF = GetIORs_packed(viorP, aiorP, matID, instID);
        uint   iorsPacked = PackHalf2_payload(iorsF);

        // Phantom surface: if ior.y == 0, advance and continue
        if ((iorsPacked >> 16) == 0u)
        {
            rayOrigin += hitObj.GetRayTCurrent() * rayDir;
            UpdateIORStack_packed(viorP, aiorP, matID, instID);
            continue;
        }

        uint currentMediumMatID = GetCurrentMediumMaterialID_packed(viorP, aiorP);
        if (currentMediumMatID == 0x0000FFFFu) currentMediumMatID = MEDIUM_INVALID_15;

        // Pointer from packed stack
        int viorPtr = GetVolumePtrFast_packed(viorP);

        // Metadata packed once
        uint meta0 = PackMeta0_payload((uint)depth, viorPtr, matID, currentMediumMatID);
        uint meta1 = PackMeta1_payload(currentMediumMatID, (depth == 0) ? PF_FIRST_BOUNCE : 0);

        // Invoke ClosestHit
        float2 nextDir2;
        uint   nextNPacked;
        uint   nextPackedThroughput; // CHS returns updateWeight in this slot
        float  nextPdf;
        uint   nextSeed;
        uint   outMeta1;

        {
            PathRayPayload payload = InitPayload_Raygen_packed_payload(
                rayDir2,
                prevNPacked,
                meta0,
                meta1,
                seed,
                iorsPacked,
                packedThroughput,
                prev_pdf
            );

            dx::HitObject::Invoke(hitObj, payload);

            nextDir2             = payload.dir2;
            nextNPacked          = payload.packedNs;
            nextPackedThroughput = payload.packedThroughput;
            nextPdf              = payload.bsdfPdf;
            nextSeed             = payload.seed;
            outMeta1             = payload.meta1;

            // Transmission flag update uses meta1 from CHS
            if ((GetFlags_payload(outMeta1) & PF_TRANSMIT_BACK) != 0)
            {
                UpdateIORStack_packed(viorP, aiorP, matID, instID);
            }
        }

        // Validate and update state
        float3 s = OctDecodeFloat2_payload(nextDir2);
        float3 weight = UnpackRGB9E5(nextPackedThroughput);

        if (dot(s, s) < 1e-12f || nextPdf <= 1e-6f || any(isnan(weight)) || any(isinf(weight)))
            break;

        // Apply throughput update
        {
            float3 T = UnpackRGB9E5(packedThroughput);
            T *= weight;
            packedThroughput = PackRGB9E5(T);
        }

        // Update RNG
        seed = nextSeed;

        // Russian Roulette
        if (depth > 0)
        {
            float3 T = UnpackRGB9E5(packedThroughput);
            float survivalProb = min(1.0f, Luma(T));
            if (RandomFloatSingle(seed) >= survivalProb) break;
            T /= max(survivalProb, 0.1f);
            packedThroughput = PackRGB9E5(T);
        }

        // Advance ray
        rayOrigin += hitObj.GetRayTCurrent() * rayDir;

        // Carry compressed state to next bounce
        rayDir2     = nextDir2;
        prevNPacked = nextNPacked;
        prev_pdf    = nextPdf;
    }

    // Finally set W of the reservoirs (TODO: GI)
    uint idx3 = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
    float p_hat = load_phat_di(g_Reservoirs_current_di, idx3);
    float w_sum = load_wsum_di(g_Reservoirs_current_di, idx3);
    float W = 0.0f;
    if (p_hat > 1e-6f && w_sum > 0.0f)
    {
        W = w_sum / p_hat;
    }

    store_W_di(g_Reservoirs_current_di, idx3, W);
    store_M_di(g_Reservoirs_current_di, idx3, 1);

    uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);

    float F    = load_F_gi   (g_Reservoirs_current_gi, idx_gi);
    float wsum = load_wsum_gi(g_Reservoirs_current_gi, idx_gi);

    float Wgi = 0.0f;
    if (F > 1e-6f && wsum > 0.0f)
    {
        Wgi = wsum / F;
        if (isnan(Wgi) || isinf(Wgi)) Wgi = 0.0f;
    }
    if(Wgi == 0)
        InvalidateReservoirGI_ShadingNormal(g_Reservoirs_current_gi, idx_gi);

    store_W_gi(g_Reservoirs_current_gi, idx_gi, Wgi);
    store_M_gi(g_Reservoirs_current_gi, idx_gi, 1u);

    // Update canonical Jacobian cache for the stored GI sample
    if (F > 1e-6f)
    {
        SampleData  sd  = loadSampleData(g_sample_current, idx_gi);
        Reservoir_GI r  = loadReservoirGI(g_Reservoirs_current_gi, idx_gi);

        // Use cached pdfx2 in r.J_gi.x (0 for non-NEE, nonzero for NEE)
        float Jc = PSSJacobian(sd.x1, sd.n1_s, sd.n1_g, sd.o, sd.matID,
                               sd.localKd, sd.localPr, sd.localPm, sd.etai, sd.etat,
                               r.x2_gi, r.n2_s_gi, r.n2_g_gi, r.V2_gi, r.matID_gi,
                               r.localKd_gi, r.localPr_gi, r.localPm_gi, r.etai_gi, r.etat_gi,
                               r.J_gi.x);

        // Write back only J.y (keep J.x = pdfx2 as-is)
        r.J_gi.y = Jc;
        storeReservoirGI(g_Reservoirs_current_gi, idx_gi, r);
    }
}
