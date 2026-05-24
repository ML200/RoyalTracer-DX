#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace pb {

//====================================
//FNV-1a 64-bit. Sufficient for M3 cache keys (small inputs, low collision
//risk at our scale). Will be swapped for xxh3 in M5+ when per-field-content
//hashing kicks in and speed matters. The Hasher accumulator is for building
//composite keys from heterogeneous pieces.
//====================================

constexpr std::uint64_t kFnv1a64Seed   = 0xcbf29ce484222325ull;
constexpr std::uint64_t kFnv1a64Prime  = 0x00000100000001B3ull;

inline std::uint64_t fnv1a64(const void* data, std::size_t n, std::uint64_t seed = kFnv1a64Seed) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    std::uint64_t h = seed;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= kFnv1a64Prime;
    }
    return h;
}

class Hasher {
public:
    Hasher() = default;

    void feed_bytes(const void* data, std::size_t n) {
        state_ = fnv1a64(data, n, state_);
    }

    template <typename T>
    void feed_pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>, "feed_pod needs trivially-copyable T");
        feed_bytes(&v, sizeof(T));
    }

    void feed_string(std::string_view s) {
        std::uint64_t len = s.size();
        feed_pod(len);
        feed_bytes(s.data(), s.size());
    }

    std::uint64_t finish() const { return state_; }

private:
    std::uint64_t state_ = kFnv1a64Seed;
};

}
