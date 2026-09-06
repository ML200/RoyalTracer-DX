#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//AUTO EXPOSURE REDUCTION
//====================================
//8x8 thread groups, each averages its tile's log2-luminance of the real HDR
//scene (inverse Reinhard of the DLSS-RR output) and atomically adds the
//(fixed-point) tile mean to gAutoExpose. Finalize pass converts the
//accumulator back to a float and smooths it.
//
//The shading pass writes per-channel Reinhard'd color to DLSS input; DLSS
//returns Reinhard'd output. Auto-exposure must measure real luminance, not
//the Reinhard-compressed image (everything maps into [0,1], so the histogram
//collapses against the upper rail and AE underestimates bright scenes). We
//invert Reinhard before taking log2.
//
//We sum per-group means (not per-pixel sums) so the accumulator stays inside
//uint32 even at 4K: tilesAt4K * AE_LOG_SCALE * (2 * AE_LOG_OFFSET)
//= 32400 * 8 * 28 = 7.26M, well below 4.3e9.
//Finalize divides by tileCount instead of pixelCount to undo this.

#define AE_GROUP_SIZE 64u  // 8x8

groupshared float g_tileLL[AE_GROUP_SIZE];
groupshared uint  g_tileMask[AE_GROUP_SIZE];

float Luminance(float3 c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

//matches InverseDlssReinhard in postprocess: shading writes c/(1+L), this
//recovers c = r/(1-L_r). Saturation guards against tiny negative DLSS residuals.
inline float3 InverseReinhardForAE(float3 r) {
    r = saturate(r);
    const float lumR = 0.2126f * r.x + 0.7152f * r.y + 0.0722f * r.z;
    return r / max(1.0f - lumR, 1e-4f);
}

[numthreads(8, 8, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint GIdx : SV_GroupIndex)
{
    //per-thread log2-luminance + valid-pixel mask written to groupshared
    if (DTid.x < gImageWidth && DTid.y < gImageHeight) {
        //PT-mode bypass (RS_FLAG_PT_ONLY): shading fed DLSS PRE-EXPOSED
        //linear radiance (radiance * AE exposure — this state's own output),
        //so divide the exposure back out to keep measuring TRUE scene
        //luminance; the closed loop then has the same fixed point as the
        //Reinhard path. (Inverting the Reinhard instead would saturate any
        //value >= 1 and blow the measurement up 10000x.) Must stay in
        //lockstep with DlssEncode / DlssDecode in Pass_shading /
        //Pass_postprocess — same AE state, same 0.18 key.
        const float3 dlssOut = g_dlssOutput[DTid.xy].xyz;
        float3 hdr;
        if (PT_ONLY_MODE) {
            const float expNow =
                0.18f / max(exp2(asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED))), 1e-6f);
            hdr = max(dlssOut, 0.0f) / max(expNow, 1e-8f);
        } else {
            hdr = InverseReinhardForAE(dlssOut);
        }
        const float  lum     = max(Luminance(hdr), 1e-6f);
        g_tileLL[GIdx]   = clamp(log2(lum), -AE_LOG_OFFSET, AE_LOG_OFFSET);
        g_tileMask[GIdx] = 1u;
    } else {
        g_tileLL[GIdx]   = 0.0f;
        g_tileMask[GIdx] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    //thread 0 sums + atomically adds the tile mean. With 8x8 a serial sum is
    //fine (64 adds), groupshared reduction would only save ~5 cycles per group
    if (GIdx == 0u) {
        float sumLL = 0.0f;
        uint  cnt   = 0u;
        [unroll] for (uint i = 0u; i < AE_GROUP_SIZE; ++i) {
            sumLL += g_tileLL[i];
            cnt   += g_tileMask[i];
        }
        if (cnt > 0u) {
            const float tileMean = sumLL / float(cnt);
            const uint  packed   = (uint)((tileMean + AE_LOG_OFFSET) * AE_LOG_SCALE);
            gAutoExpose.InterlockedAdd(AE_OFFS_SUM, packed);
            gAutoExpose.InterlockedAdd(AE_OFFS_TILE_COUNT, 1u);
        }
    }
}
