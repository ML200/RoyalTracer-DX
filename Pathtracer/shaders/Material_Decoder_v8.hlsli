//====================================
//MATERIAL DECODER
//====================================
//accessors over compressed AoS g_mat, HLSL CSEs same material fetches

#ifndef MATERIAL_DECODER_V8_HLSLI
#define MATERIAL_DECODER_V8_HLSLI

//====================================
//CORE KD NI PBR
//====================================
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

//====================================
//TRANSMISSION COAT ANISO ALPHA
//====================================
inline float3 LoadTf(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].Tf_rgb);
}

inline float LoadPcr(uint matID)
{
    return float(g_mat[matID].Pcr_Aniso_Rot_AlphaTh & 0xFFu) * (1.0f / 255.0f);
}

//int8 in [-127,127] mapped to [-1,1]
inline float LoadAniso(uint matID)
{
    const uint raw = (g_mat[matID].Pcr_Aniso_Rot_AlphaTh >> 8) & 0xFFu;
    const int  s   = (int)(raw << 24) >> 24;
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

//====================================
//TEXTURE IDS AND UV SCALES
//====================================
inline int LoadAlbedoTexID(uint matID)
{
    const uint lo = g_mat[matID].texIDs_01 & 0xFFFFu;
    return (int)(lo << 16) >> 16;
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

//bit 16 of texIDs_2: 1 = opacity-channel sample is transparency (1=transparent, 0=opaque),
//AnyHit flips the cutout test so map_Tr-style and inverted-PNG opacity textures render correctly.
//(bit 17 = SSS-enable, bits 24..31 = SSS entry weight — see the SSS decoders below.)
inline bool LoadInvertAlpha(uint matID)
{
    return (g_mat[matID].texIDs_2 & (1u << 16)) != 0u;
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

//====================================
//SUBSURFACE SCATTERING
//====================================
//bit 17 of texIDs_2: 1 = random-walk SSS enabled.
inline bool LoadIsSSS(uint matID)
{
    return (g_mat[matID].texIDs_2 & (1u << 17)) != 0u;
}

//RGB single-scattering albedo (the inside colour); per-scatter throughput.
inline float3 LoadSSSAlbedo(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].sss_albedo);
}

//Scalar scattering distance / mean free path. sigma_t = 1/radius.
inline float LoadSSSRadius(uint matID)
{
    return f16tof32(g_mat[matID].sss_radius_g & 0xFFFFu);
}

//Henyey-Greenstein anisotropy g in [-1,1].
inline float LoadPhaseG(uint matID)
{
    return f16tof32(g_mat[matID].sss_radius_g >> 16);
}

//Entry-probability scale (texIDs_2 bits 24..31, unorm8). p_enter = weight * Fresnel-T.
inline float LoadSSSWeight(uint matID)
{
    return float((g_mat[matID].texIDs_2 >> 24) & 0xFFu) * (1.0f / 255.0f);
}

#endif
