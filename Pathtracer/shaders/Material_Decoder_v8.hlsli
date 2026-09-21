#pragma once
#include "OceanLayout.h"
#include "OceanOptics.hlsli"

// Only the editable water material uses this adapter; diagnostic palette entries and
// unrelated glass keep their original closures and proposal probabilities.
inline bool LoadIsOceanMaterial(uint matID)
{
    [branch] if (!OCEAN_ENABLED) return false;
    StructuredBuffer<OceanParamsGPU> ocean = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    return matID == ocean[0].materialBase;
}

// Lobe width the sun sampler and NEE widen the water surface to; see OceanHighlightRoughness.
inline float LoadOceanSunLobeRoughness()
{
    [branch] if (!OCEAN_ENABLED) return 0.0f;
    StructuredBuffer<OceanParamsGPU> ocean = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    return ocean[0].sunLobeRoughness;
}

// Volume controls never change the surface opacity or its Fresnel energy split.
inline float3 LoadKd_rgb(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].Kd_rgb);
}

inline float LoadKd_w(uint matID)
{
    if (FORCE_DIFFUSE) return 1.0f;
    return f16tof32(g_mat[matID].w_Ni & 0xFFFFu);
}

// Decode base color and material flags from packed storage.
inline float4 LoadKd(uint matID)
{
    return float4(LoadKd_rgb(matID), LoadKd_w(matID));
}

inline float LoadNi(uint matID)
{
    return FORCE_DIFFUSE ? 1.0f : f16tof32(g_mat[matID].w_Ni >> 16);
}

// Decode roughness, metalness, sheen, and coat parameters.
inline float4 LoadPrPmPsPc(uint matID)
{
    if (FORCE_DIFFUSE) return float4(1.0f, 0.0f, 0.0f, 0.0f);
    const uint p = g_mat[matID].PrPmPsPc;
    return float4(
        float((p >>  0) & 0xFFu) * (1.0f / 255.0f),
        float((p >>  8) & 0xFFu) * (1.0f / 255.0f),
        float((p >> 16) & 0xFFu) * (1.0f / 255.0f),
        float((p >> 24) & 0xFFu) * (1.0f / 255.0f));
}

inline float LoadPr(uint matID)
{
    return FORCE_DIFFUSE ? 1.0f : float(g_mat[matID].PrPmPsPc & 0xFFu) * (1.0f / 255.0f);
}

inline float LoadDiffuseRoughness(uint matID)
{
    return float((g_mat[matID].texIDs_2 >> 19) & 31u) * (1.0f / 31.0f);
}

inline float LoadPm(uint matID)
{
    return FORCE_DIFFUSE ? 0.0f : float((g_mat[matID].PrPmPsPc >> 8) & 0xFFu) * (1.0f / 255.0f);
}

inline float LoadPs(uint matID)
{
    return FORCE_DIFFUSE ? 0.0f : float((g_mat[matID].PrPmPsPc >> 16) & 0xFFu) * (1.0f / 255.0f);
}

inline float LoadPc(uint matID)
{
    return FORCE_DIFFUSE ? 0.0f : float((g_mat[matID].PrPmPsPc >> 24) & 0xFFu) * (1.0f / 255.0f);
}

inline float3 LoadTf(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].Tf_rgb);
}

static float g_regularizeRoughness = 0.0f;
inline float LoadPcr(uint matID)
{
    return max(float(g_mat[matID].Pcr_Aniso_Rot_AlphaTh & 0xFFu) * (1.0f / 255.0f), g_regularizeRoughness);
}

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

inline bool LoadInvertAlpha(uint matID)
{
    return (g_mat[matID].texIDs_2 & (1u << 16)) != 0u;
}

inline bool LoadIsThinGlass(uint matID)
{
    return !FORCE_DIFFUSE && (g_mat[matID].texIDs_2 & (1u << 18)) != 0u;
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

inline bool LoadIsSSS(uint matID)
{
    if (FORCE_DIFFUSE || (g_mat[matID].texIDs_2 & (1u << 17)) == 0u) return false;
    // Water's controls drive segment volume transport, never the solid-object walk.
    return !LoadIsOceanMaterial(matID);
}

inline float3 LoadSSSAlbedo(uint matID)
{
    return UnpackRGB9E5(g_mat[matID].sss_albedo);
}

inline float LoadSSSRadius(uint matID)
{
    return f16tof32(g_mat[matID].sss_radius_g & 0xFFFFu);
}

inline float LoadPhaseG(uint matID)
{
    return f16tof32(g_mat[matID].sss_radius_g >> 16);
}

inline float LoadSSSWeight(uint matID)
{
    return float((g_mat[matID].texIDs_2 >> 24) & 0xFFu) * (1.0f / 255.0f);
}

inline bool MaterialIsFreeBounce(uint matID)
{

    if (LoadKd_w(matID) >= 1e-3f) return false;

    const bool isGlass      = (LoadPm(matID) < 0.5f) &&
                              (any(LoadTf(matID) > 0.0f) || LoadIsThinGlass(matID));
    const bool isTranslucent = LoadIsSSS(matID);
    return isGlass || isTranslucent;
}
