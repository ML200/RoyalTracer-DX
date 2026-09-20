#pragma once
// Entry include of the cache training pass: its lock protocol needs coherent cache accesses.
globallycoherent RWByteAddressBuffer g_sharc : register(u27);
#include "Globals_v8.hlsli"
