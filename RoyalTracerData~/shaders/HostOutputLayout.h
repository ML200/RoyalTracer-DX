#pragma once

// Results for a host process (the Unity plugin), written by Pass_host_output_v8.hlsl into textures the host shares.
// Descriptor heap slots (ResourceDescriptorHeap), between the ocean range and the bindless textures.
#define HOST_COLOR_UAV_SLOT 240u // RWTexture2D<float4>: DLSS-RR result, pre-exposed; alpha = log2 of that exposure
#define HOST_DEPTH_UAV_SLOT 241u // RWTexture2D<float>: view depth (m) of the unjittered primary hit, 0 = miss
// The host's sky (hostSkyScale > 0), Texture2D<float4>: radiance over a latitude-longitude map of the renderer's
// directions, u = atan2(z, x) / 2 pi + 1/2, v = acos(y) / pi.
#define HOST_SKY_SRV_SLOT 242u
