#pragma once
//====================================
//REUSE TEXTURE GENERATOR
//====================================
//self-inverting textures for paired spatial reuse (Lin, Kettunen, Wyman 2026)
//each texel stores 2D int16 delta (dx,dy), partner's delta lands back on source
//texture is tileable under per-axis wrap at size

#include <cstdint>
#include <vector>

//size must be even (pairing needs even pixel count)
//sigma >= 0.8 approximates stddev in pixels, paper uses 254/230/210 for 3 slots
//outRG is 2*size*size int16 interleaved [dx0,dy0,dx1,dy1,...], row-major, DXGI_FORMAT_R16G16_SINT
void GenerateReuseTexture(int size,
                          float sigma,
                          uint32_t seed,
                          std::vector<int16_t>& outRG);

//verifies self-inverting property
//outFirstBadTexel gets linear index of first offending texel on failure
bool ValidateReuseTexture(int size,
                          const std::vector<int16_t>& rg,
                          int* outFirstBadTexel = nullptr);
