#pragma once
//====================================
//REUSE TEXTURE GENERATOR
//====================================
//self-inverting textures for paired spatial reuse (Lin, Kettunen, Wyman 2026)
//each texel stores 2D int16 delta (dx,dy), partner's delta lands back on source
//texture is tileable under per-axis wrap at size
//
//Restored from the deprecated ReSTIR pipeline for ReSTIR lite
//(shaders/RestirLite_v8.hlsli): the three tables live in the SHaRC buffer
//(SharcLayout.h LITE_REUSE_*) instead of SRVs.

#include <cstdint>
#include <vector>

//size must be even (pairing needs even pixel count)
//sigma >= 0.8 approximates stddev in pixels, paper uses 254/230/210 for 3 slots
//outRG is 2*size*size int16 interleaved [dx0,dy0,dx1,dy1,...], row-major
void GenerateReuseTexture(int size,
                          float sigma,
                          uint32_t seed,
                          std::vector<int16_t>& outRG);

//verifies self-inverting property
//outFirstBadTexel gets linear index of first offending texel on failure
bool ValidateReuseTexture(int size,
                          const std::vector<int16_t>& rg,
                          int* outFirstBadTexel = nullptr);
