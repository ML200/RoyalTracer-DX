//====================================================================
//MATERIAL DECODER
//====================================================================
//Accessors over the compressed AoS g_mat buffer.
//Each Load* function fetches exactly the field it needs from g_mat[matID].
//The HLSL compiler CSEs same-material accesses across multiple accessors
//in the same basic block, so call them freely, only the fields touched
//survive after optimization.
//
//Buffer declaration and struct layout live in Includes_v8.hlsli and
//Data_v8.hlsli. CPU-side packing is in src/Components/Vertex.h.

#ifndef MATERIAL_DECODER_V8_HLSLI
#define MATERIAL_DECODER_V8_HLSLI

//====================================================================
//CORE, KD / NI / PBR
//====================================================================
inline float3 LoadKd_rgb(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].Kd_rgb);
}

inline float LoadKd_w(uint matID)
{
    return f16tof32(g_mat[matID].w_Ni & 0xFFFFu);
}

inline float4 LoadKd(uint matID)
{
    const MatPacked m = g_mat[matID];
    return float4(UnpackRGB9E5(m.Kd_rgb), f16tof32(m.w_Ni & 0xFFFFu));
}

inline float LoadNi(uint matID)
{
    return f16tof32(g_mat[matID].w_Ni >> 16);
}

inline float4 LoadPrPmPsPc(uint matID)
{
    const uint p = g_mat[matID].PrPmPsPc;
    return float4(
        float((p >>  0) & 0xFFu) * (1.0f / 255.0f),
        float((p >>  8) & 0xFFu) * (1.0f / 255.0f),
        float((p >> 16) & 0xFFu) * (1.0f / 255.0f),
        float((p >> 24) & 0xFFu) * (1.0f / 255.0f));
}

inline float LoadPr(uint matID)
{
    return float(g_mat[matID].PrPmPsPc & 0xFFu) * (1.0f / 255.0f);
}

inline float LoadPm(uint matID)
{
    return float((g_mat[matID].PrPmPsPc >> 8) & 0xFFu) * (1.0f / 255.0f);
}

inline float LoadPs(uint matID)
{
    return float((g_mat[matID].PrPmPsPc >> 16) & 0xFFu) * (1.0f / 255.0f);
}

inline float LoadPc(uint matID)
{
    return float((g_mat[matID].PrPmPsPc >> 24) & 0xFFu) * (1.0f / 255.0f);
}

//====================================================================
//TRANSMISSION, COAT, ANISO, ALPHA
//====================================================================
inline float3 LoadTf(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].Tf_rgb);
}

inline float LoadPcr(uint matID)
{
    return float(g_mat[matID].Pcr_Aniso_Rot_AlphaTh & 0xFFu) * (1.0f / 255.0f);
}

//Anisotropy stored as int8 in [-127, 127], mapped to [-1, 1].
inline float LoadAniso(uint matID)
{
    const uint raw = (g_mat[matID].Pcr_Aniso_Rot_AlphaTh >> 8) & 0xFFu;
    const int  s   = (int)(raw << 24) >> 24;   //sign-extend
    return float(s) * (1.0f / 127.0f);
}

inline float LoadAnisoRot(uint matID)
{
    return float((g_mat[matID].Pcr_Aniso_Rot_AlphaTh >> 16) & 0xFFu) * (1.0f / 255.0f);
}

inline float3 LoadPcrAnisoAnisor(uint matID)
{
    const uint p = g_mat[matID].Pcr_Aniso_Rot_AlphaTh;
    const uint rawA = (p >> 8) & 0xFFu;
    const int  aS   = (int)(rawA << 24) >> 24;
    return float3(
        float( p        & 0xFFu) * (1.0f / 255.0f),
        float(aS)                * (1.0f / 127.0f),
        float((p >> 16) & 0xFFu) * (1.0f / 255.0f));
}

inline float LoadAlphaThreshold(uint matID)
{
    return float((g_mat[matID].Pcr_Aniso_Rot_AlphaTh >> 24) & 0xFFu) * (1.0f / 255.0f);
}

//====================================================================
//TEXTURE IDS AND UV SCALES
//====================================================================
inline int LoadAlbedoTexID(uint matID)
{
    const uint lo = g_mat[matID].texIDs_01 & 0xFFFFu;
    return (int)(lo << 16) >> 16;   //sign-extend 16 to 32
}

inline int LoadNormalTexID(uint matID)
{
    const uint hi = g_mat[matID].texIDs_01 >> 16;
    return (int)(hi << 16) >> 16;
}

inline int LoadRmaTexID(uint matID)
{
    const uint lo = g_mat[matID].texIDs_2 & 0xFFFFu;
    return (int)(lo << 16) >> 16;
}

inline float2 LoadAlbedoUVScale(uint matID)
{
    const uint p = g_mat[matID].uv_albedo;
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}

inline float2 LoadNormalUVScale(uint matID)
{
    const uint p = g_mat[matID].uv_normal;
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}

inline float2 LoadRmaUVScale(uint matID)
{
    const uint p = g_mat[matID].uv_rma;
    return float2(f16tof32(p & 0xFFFFu), f16tof32(p >> 16));
}

#endif //MATERIAL_DECODER_V8_HLSLI
