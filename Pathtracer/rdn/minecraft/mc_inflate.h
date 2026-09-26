#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace mc {

enum class Compression : uint8_t { Zlib, Gzip, RawDeflate, None };

bool inflate_buffer(const uint8_t* data, size_t size, Compression kind,
                    std::vector<uint8_t>& out, size_t expected = 0);

}
