#pragma once
// Training pass: its lock protocol needs coherent cache access.
globallycoherent RWByteAddressBuffer g_sharc : register(u27);
#include "Globals_v8.hlsli"
