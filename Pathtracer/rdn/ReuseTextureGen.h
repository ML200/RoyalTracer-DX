#pragma once

#include <cstdint>
#include <vector>

void GenerateReuseTexture(int size, float sigma, uint32_t seed, std::vector<int16_t>& outRG);

bool ValidateReuseTexture(int size, const std::vector<int16_t>& rg, int* outFirstBadTexel = nullptr);
