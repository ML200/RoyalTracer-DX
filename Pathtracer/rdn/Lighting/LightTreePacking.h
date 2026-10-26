#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace lt {
// The shader reads the 64-byte node records as uint4s.
template <class Node> inline std::vector<uint32_t> EncodeLightBLAS(const std::vector<Node>& nodes, uint32_t leafBase = 0u) {
    static_assert(sizeof(Node) == 64);
    auto based = nodes;
    for (auto& node : based)
        if (!node.childCount) node.triFirst += leafBase;
    std::vector<uint32_t> words(based.size() * sizeof(Node) / sizeof(uint32_t));
    if (!words.empty()) std::memcpy(words.data(), based.data(), words.size() * sizeof(uint32_t));
    return words;
}
} // namespace lt
