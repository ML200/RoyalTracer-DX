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
    // Margin covers oct16 (< 0.13 mrad) and FP32 decode error.
    constexpr double margin = 0.001;
    const double c = std::clamp(double(cosine), -1.0, 1.0);
    if (c <= -std::cos(margin))
        return -1.f;
    const double expanded = c * std::cos(margin) - std::sqrt((std::max)(0.0, 1.0 - c * c)) * std::sin(margin);
    return (std::max)(-1.f, std::nextafter(float(expanded), -std::numeric_limits<float>::infinity()));
}

template <class Node> inline LightTLASNodePacked PackTLASNode(const Node& n) {
    LightTLASNodePacked p{};
    p.bmin = n.bmin;
    p.bmax = n.bmax;
    p.power = n.power;
    p.axis = PackLightAxis(n.axis);
    p.cosTheta_o = PackedLightCosine(n.cosTheta_o);
    p.sinTheta_o = std::sqrt((std::max)(0.f, 1.f - p.cosTheta_o * p.cosTheta_o));
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
    n.bmin = p.bmin;
    n.bmax = p.bmax;
    n.power = p.power;
    n.axis = UnpackLightAxis(p.axis);
    n.cosTheta_o = p.cosTheta_o;
    n.sinTheta_o = p.sinTheta_o;
    n.firstChild = p.childCount ? p.index : UINT32_MAX;
    n.childCount = p.childCount;
    n.slot = p.childCount ? UINT32_MAX : p.index;
    return n;
}

inline uint32_t QuantizeLightBounds(float low, float high, float origin, float end) {
    if (origin == end)
        return 0u;
    const double scale = 65535.0 / (double(end) - double(origin));
    // One extra unit covers FP32 lerp error, fused or not.
    const auto lo = uint32_t(std::clamp(std::floor((double(low) - origin) * scale) - 1.0, 0.0, 65535.0));
    const auto hi = uint32_t(std::clamp(std::ceil((double(high) - origin) * scale) + 1.0, 0.0, 65535.0));
    return lo | (hi << 16u);
}
inline float UnpackLightBound(uint32_t q, float origin, float end, bool upper) {
    if (q == 0u)
        return origin;
    if (q == 65535u)
        return end;
    const float t = float(q) * (1.f / 65535.f);
    const float v = (1.f - t) * origin + t * end;
    const float direction = upper ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    return std::clamp(std::nextafter(std::nextafter(v, direction), direction), origin, end);
}
template <class Node>
inline std::vector<LightBLASNodePacked> PackBLAS(const std::vector<Node>& nodes, uint32_t leafBase = 0u) {
    if (nodes.empty())
        return {};
    std::vector<LightBLASNodePacked> result(nodes.size() + 1u);
    const auto lo = nodes[0].bmin, hi = nodes[0].bmax;
    const float header[8] = {lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, 0, 0};
    std::memcpy(&result[0], header, sizeof(header));
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& n = nodes[i];
        auto& p = result[i + 1u];
        if (!n.childCount && n.triCount != 1u)
            throw std::logic_error("Packed light BLAS requires one triangle per leaf");
        p.boundsX = QuantizeLightBounds(n.bmin.x, n.bmax.x, lo.x, hi.x);
        p.boundsY = QuantizeLightBounds(n.bmin.y, n.bmax.y, lo.y, hi.y);
        p.boundsZ = QuantizeLightBounds(n.bmin.z, n.bmax.z, lo.z, hi.z);
        p.power = n.power;
        p.axis = PackLightAxis(n.axis);
        p.cosTheta_o = PackedLightCosine(n.cosTheta_o);
        p.index = n.childCount ? n.firstChild : n.triFirst + leafBase;
        p.childCount = n.childCount;
    }
    return result;
}
template <class Node> inline Node UnpackBLASNode(const LightBLASNodePacked* mesh, uint32_t index) {
    float h[8];
    std::memcpy(h, mesh, sizeof(h));
    const auto& p = mesh[index + 1u];
    Node n{};
    n.bmin = {UnpackLightBound(p.boundsX & 65535u, h[0], h[3], false),
              UnpackLightBound(p.boundsY & 65535u, h[1], h[4], false),
              UnpackLightBound(p.boundsZ & 65535u, h[2], h[5], false)};
    n.bmax = {UnpackLightBound(p.boundsX >> 16u, h[0], h[3], true),
              UnpackLightBound(p.boundsY >> 16u, h[1], h[4], true),
              UnpackLightBound(p.boundsZ >> 16u, h[2], h[5], true)};
    n.power = p.power;
    n.axis = UnpackLightAxis(p.axis);
    n.cosTheta_o = p.cosTheta_o;
    n.sinTheta_o = std::sqrt((std::max)(0.f, 1.f - p.cosTheta_o * p.cosTheta_o));
    n.firstChild = p.childCount ? p.index : UINT32_MAX;
    n.childCount = p.childCount;
    n.triFirst = p.childCount ? 0u : p.index;
    n.triCount = p.childCount ? 0u : 1u;
    return n;
}
} // namespace lt

namespace lt {
// Shader reads either layout as uint4 records.
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
