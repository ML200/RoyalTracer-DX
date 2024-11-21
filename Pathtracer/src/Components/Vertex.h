//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VERTEX_H
#define PATHTRACER_VERTEX_H


#include <functional>
#include "../../rdn/Renderer.h"

using namespace DirectX;

struct Material{
 XMFLOAT4 Kd = {1,1,1,1};
 XMFLOAT3 Ks = {1,1,1};
 XMFLOAT3 Ke = {0,0,0};
 XMFLOAT4 Pr_Pm_Ps_Pc = {0,0,0,0};
 XMFLOAT2 aniso_anisor = {0,0};
 float Ni = 1;
 float LUT[16][16] = {0};

 //ADD MAP IDs LATER
 Material(XMFLOAT4 kd, XMFLOAT4 pr_pm_ps_pc):Kd(kd), Pr_Pm_Ps_Pc(pr_pm_ps_pc){}
};

struct Vertex {
    XMFLOAT3 position;
    XMFLOAT4 normal_material = {1,1,1, 0};
    // #DXR Extra: Indexed Geometry
    Vertex(XMFLOAT3 pos, XMFLOAT4 norm) : position(pos), normal_material(norm.x,norm.y,norm.z,norm.w) {}

    // Equality operator
    bool operator==(const Vertex& other) const {
        return XMVector3Equal(XMLoadFloat3(&position), XMLoadFloat3(&other.position));
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

            return hashFloat3(vertex.position) << 1;
        }
    };
}


#endif //PATHTRACER_VERTEX_H
