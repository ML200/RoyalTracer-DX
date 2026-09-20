#pragma once

#include "../../shaders/LightTreePacked.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lt {
inline DirectX::XMFLOAT3 UnpackLightAxis(uint32_t packed) {
    float x = float(packed & 65535u) * (2.f / 65535.f) - 1.f;
    float y = float(packed >> 16u) * (2.f / 65535.f) - 1.f;
    float z = 1.f - std::fabs(x) - std::fabs(y);
    const float t = (std::max)(-z, 0.f);
    x += x >= 0.f ? -t : t;
    y += y >= 0.f ? -t : t;
    const float inv = 1.f / std::sqrt(x * x + y * y + z * z);
    return {x * inv, y * inv, z * inv};
}

inline uint32_t PackLightAxis(DirectX::XMFLOAT3 axis) {
    const float len = std::fabs(axis.x) + std::fabs(axis.y) + std::fabs(axis.z);
    if (!(len > 0.f))
        axis = {0, 0, 1};
    else {
        axis.x /= len;
        axis.y /= len;
        axis.z /= len;
    }
    if (axis.z < 0.f) {
        const float x = (1.f - std::fabs(axis.y)) * (axis.x >= 0.f ? 1.f : -1.f);
        axis.y = (1.f - std::fabs(axis.x)) * (axis.y >= 0.f ? 1.f : -1.f);
        axis.x = x;
    }
    auto q = [](float v) { return uint32_t(std::floor(std::clamp(v * .5f + .5f, 0.f, 1.f) * 65535.f + .5f)); };
    return q(axis.x) | (q(axis.y) << 16u);
}

inline float PackedLightCosine(float cosine) {
    // Oct16 has < 0.00013 rad direction error. Include FP32 decode/normalize
    // error as well. Rotate the cone outward without acos in the encoder.
    constexpr double margin = 0.001;
    const double c = std::clamp(double(cosine), -1.0, 1.0);
    if (c <= -std::cos(margin))
        return -1.f;
    const double expanded = c * std::cos(margin) - std::sqrt((std::max)(0.0, 1.0 - c * c)) * std::sin(margin);
    return (std::max)(-1.f, std::nextafter(float(expanded), -std::numeric_limits<float>::infinity()));
}

// IEEE half conversions, with a directed variant for the conservatively rounded fields.
inline uint16_t FloatToHalfBits(float value) {
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16u) & 0x8000u;
    const uint32_t exponent = (f >> 23u) & 0xFFu;
    uint32_t mantissa = f & 0x7FFFFFu;
    if (exponent == 0xFFu)
        return uint16_t(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
    const int32_t e = int32_t(exponent) - 127 + 15;
    if (e >= 31)
        return uint16_t(sign | 0x7C00u);
    if (e <= 0) {
        if (e < -10)
            return uint16_t(sign);
        mantissa |= 0x800000u;
        const uint32_t shift = uint32_t(14 - e);
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half & 1u)))
            ++half;
        return uint16_t(sign | half);
    }
    uint32_t half = (uint32_t(e) << 10u) | (mantissa >> 13u);
    const uint32_t remainder = mantissa & 0x1FFFu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u)))
        ++half;
    return uint16_t(sign | half);
}
inline float HalfBitsToFloat(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16u;
    uint32_t exponent = (h >> 10u) & 0x1Fu, mantissa = h & 0x3FFu, f;
    if (exponent == 0u) {
        if (mantissa == 0u)
            f = sign;
        else {
            exponent = 1u;
            while ((mantissa & 0x400u) == 0u) {
                mantissa <<= 1u;
                --exponent;
            }
            f = sign | ((exponent + 127u - 15u) << 23u) | ((mantissa & 0x3FFu) << 13u);
        }
    } else if (exponent == 31u)
        f = sign | 0x7F800000u | (mantissa << 13u);
    else
        f = sign | ((exponent + 127u - 15u) << 23u) | (mantissa << 13u);
    float value;
    std::memcpy(&value, &f, sizeof(value));
    return value;
}
// The half nearest to value, moved one ulp towards +inf (up) or -inf (down) when it fell on the
// other side, and kept finite.
inline uint16_t HalfDirected(float value, bool up) {
    value = std::clamp(value, -65504.f, 65504.f);
    uint16_t h = FloatToHalfBits(value);
    const float back = HalfBitsToFloat(h);
    if ((up && back < value) || (!up && back > value)) {
        const bool negative = (h & 0x8000u) != 0u;
        if ((h & 0x7FFFu) == 0u)
            h = up ? 0x0001u : 0x8001u;
        else if (negative == up)
            --h;
        else
            ++h;
        if ((h & 0x7FFFu) >= 0x7C00u)
            h = uint16_t((h & 0x8000u) | 0x7BFFu);
    }
    return h;
}
inline uint32_t PackHalf2(float low, bool lowUp, float high, bool highUp) {
    return uint32_t(HalfDirected(low, lowUp)) | (uint32_t(HalfDirected(high, highUp)) << 16u);
}
inline float UnpackHalfLow(uint32_t v) { return HalfBitsToFloat(uint16_t(v & 0xFFFFu)); }
inline float UnpackHalfHigh(uint32_t v) { return HalfBitsToFloat(uint16_t(v >> 16u)); }

inline float MeanResultantLength(const DirectX::XMFLOAT3& r) { return std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z); }
inline DirectX::XMFLOAT3 MeanResultantAxis(const DirectX::XMFLOAT3& r) {
    const float len = MeanResultantLength(r);
    return len > 1e-9f ? DirectX::XMFLOAT3{r.x / len, r.y / len, r.z / len} : DirectX::XMFLOAT3{0, 0, 1};
}
inline DirectX::XMFLOAT3 MeanResultantOf(uint32_t axis, float length) {
    const DirectX::XMFLOAT3 a = UnpackLightAxis(axis);
    return {a.x * length, a.y * length, a.z * length};
}
// The mean resultant length rounds down (a blurrier lobe) and the cone cosine rounds down (a
// wider cone): both towards the conservative side.
inline uint32_t PackLengthCosine(const DirectX::XMFLOAT3& rbar, float cosTheta_o) {
    return PackHalf2((std::min)(MeanResultantLength(rbar), 1.f), false, PackedLightCosine(cosTheta_o), false);
}

template <class Node> inline LightTLASNodePacked PackTLASNode(const Node& n) {
    LightTLASNodePacked p{};
    p.mean = n.mean;
    p.power = n.power;
    p.variance = n.variance;
    p.radius = n.radius;
    p.axis = PackLightAxis(MeanResultantAxis(n.rbar));
    p.lengthCos = PackLengthCosine(n.rbar, n.cosTheta_o);
    p.index = n.childCount ? n.firstChild : n.slot;
    p.childCount = n.childCount;
    return p;
}
template <class Node> inline std::vector<LightTLASNodePacked> PackTLAS(const std::vector<Node>& nodes) {
    std::vector<LightTLASNodePacked> result;
    result.reserve(nodes.size());
    for (const auto& node : nodes)
        result.push_back(PackTLASNode(node));
    return result;
}
template <class Node> inline Node UnpackTLASNode(const LightTLASNodePacked& p) {
    Node n{};
    n.mean = p.mean;
    n.power = p.power;
    n.variance = p.variance;
    n.radius = p.radius;
    n.rbar = MeanResultantOf(p.axis, UnpackHalfLow(p.lengthCos));
    n.cosTheta_o = UnpackHalfHigh(p.lengthCos);
    n.firstChild = p.childCount ? p.index : UINT32_MAX;
    n.childCount = p.childCount;
    n.slot = p.childCount ? UINT32_MAX : p.index;
    return n;
}

// The unit of a packed mesh: twice its root radius, which every member radius and standard
// deviation stays below.
inline float LightBLASUnit(float rootRadius) { return (std::max)(2.f * rootRadius, 1e-30f); }

template <class Node>
inline std::vector<LightBLASNodePacked> PackBLAS(const std::vector<Node>& nodes, uint32_t leafBase = 0u) {
    if (nodes.empty())
        return {};
    std::vector<LightBLASNodePacked> result(nodes.size() + 1u);
    const float unit = LightBLASUnit(nodes[0].radius);
    const float header[8] = {unit, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(&result[0], header, sizeof(header));
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        auto& p = result[i + 1u];
        if (!n.childCount && n.triCount != 1u)
            throw std::logic_error("Packed light BLAS requires one triangle per leaf");
        const uint32_t index = n.childCount ? n.firstChild : n.triFirst + leafBase;
        if (index > LT_PACKED_INDEX_MASK || n.childCount > 4u)
            throw std::logic_error("Packed light BLAS index does not fit its field");
        p.mean = n.mean;
        p.power = n.power;
        p.axis = PackLightAxis(MeanResultantAxis(n.rbar));
        // A wider cluster is the conservative one: both round up.
        p.sigmaRadius = PackHalf2(std::sqrt((std::max)(n.variance, 0.f)) / unit, true, n.radius / unit, true);
        p.lengthCos = PackLengthCosine(n.rbar, n.cosTheta_o);
        p.indexCount = index | (n.childCount << LT_PACKED_INDEX_BITS);
    }
    return result;
}
template <class Node> inline Node UnpackBLASNode(const LightBLASNodePacked* mesh, uint32_t index) {
    float h[8];
    std::memcpy(h, mesh, sizeof(h));
    const auto& p = mesh[index + 1u];
    Node n{};
    n.mean = p.mean;
    n.power = p.power;
    const float sigma = UnpackHalfLow(p.sigmaRadius) * h[0];
    n.variance = sigma * sigma;
    n.radius = UnpackHalfHigh(p.sigmaRadius) * h[0];
    n.rbar = MeanResultantOf(p.axis, UnpackHalfLow(p.lengthCos));
    n.cosTheta_o = UnpackHalfHigh(p.lengthCos);
    const uint32_t count = p.indexCount >> LT_PACKED_INDEX_BITS, idx = p.indexCount & LT_PACKED_INDEX_MASK;
    n.firstChild = count ? idx : UINT32_MAX;
    n.childCount = count;
    n.triFirst = count ? 0u : idx;
    n.triCount = count ? 0u : 1u;
    return n;
}
} // namespace lt

namespace lt {
// A single representation is allocated/uploaded. The shader views either layout
// as uint4 records, so changing layout does not require a second resource table.
template <class Node>
inline std::vector<uint32_t> EncodeLightBLAS(const std::vector<Node>& nodes, bool compact, uint32_t leafBase = 0u) {
    std::vector<uint32_t> words;
    if (compact) {
        const auto packed = PackBLAS(nodes, leafBase);
        words.resize(packed.size() * sizeof(LightBLASNodePacked) / sizeof(uint32_t));
        if (!words.empty()) std::memcpy(words.data(), packed.data(), words.size() * sizeof(uint32_t));
    } else {
        static_assert(sizeof(Node) == 64);
        auto uncompressed = nodes;
        for (auto& node : uncompressed)
            if (!node.childCount) node.triFirst += leafBase;
        words.resize(uncompressed.size() * sizeof(Node) / sizeof(uint32_t));
        if (!words.empty()) std::memcpy(words.data(), uncompressed.data(), words.size() * sizeof(uint32_t));
    }
    return words;
}
} // namespace lt
