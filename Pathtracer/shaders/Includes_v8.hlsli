#pragma once
// Entry include of every render pass. The training pass uses IncludesTraining_v8.hlsli instead,
// which binds the radiance cache coherently for its lock protocol.
RWByteAddressBuffer g_sharc : register(u27);
#include "Globals_v8.hlsli"
