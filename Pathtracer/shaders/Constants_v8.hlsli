#pragma once
#define EPSILON 0.000003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f
#define PI 3.1415926535
#define INV_PI 0.3183098861

#define SMOOTH_SPECULAR_THRESHOLD 0.06f
#define BROAD_GGX_ROUGHNESS 0.8f

// Texture level of detail: the width of the ray beam at a hit (the pixel cone over the path
// length, plus the spread of rough scatters) selects the mip. The bias shifts every level;
// negative is sharper.
#define PT_TEXTURE_LOD_BIAS 0.0f

#include "DlssGuideLayout.h"

#define RAY_TMAX_PLANET 1e9f

#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

#define MATID_ENV_MISS 0xFFFFFFFFu

// Debug switches: extra output slices of the shading pass, and the atmosphere inspection views.
static const bool SHADING_DEBUG_SLICES = true;
static const uint ATM_DEBUG_RING = 0u;
