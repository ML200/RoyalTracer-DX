//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VERTEX_H
#define PATHTRACER_VERTEX_H


#include <functional>
#include "../../rdn/Renderer.h"

using namespace DirectX;

struct Vertex {
    XMFLOAT3 position;
    XMFLOAT4 color;
    // #DXR Extra: Indexed Geometry
    Vertex(XMFLOAT4 pos, XMFLOAT4 /*n*/, XMFLOAT4 col)
            : position(pos.x, pos.y, pos.z), color(col) {}
    Vertex(XMFLOAT3 pos, XMFLOAT4 col) : position(pos), color(col) {}

    // Equality operator
    bool operator==(const Vertex& other) const {
        return XMVector3Equal(XMLoadFloat3(&position), XMLoadFloat3(&other.position)) &&
               XMVector4Equal(XMLoadFloat4(&color), XMLoadFloat4(&other.color));
    }

};

namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(const Vertex& vertex) const {
            auto hashFloat3 = [](const XMFLOAT3& v) {
                return std::hash<float>()(v.x) ^ std::hash<float>()(v.y) << 1 ^ std::hash<float>()(v.z) << 2;
            };
            auto hashFloat4 = [](const XMFLOAT4& v) {
                return std::hash<float>()(v.x) ^ std::hash<float>()(v.y) << 1 ^ std::hash<float>()(v.z) << 2 ^ std::hash<float>()(v.w) << 3;
            };

            return hashFloat3(vertex.position) ^ hashFloat4(vertex.color) << 1;
        }
    };
}


#endif //PATHTRACER_VERTEX_H
