//
// Created by m on 31.01.2024.
//

#ifndef PATHTRACER_D3DVERTEX_H
#define PATHTRACER_D3DVERTEX_H

#include <DirectXMath.h>
#include "../Components/Vertex.h"

class D3DVertex {
public:
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT3 normal;
    DirectX::XMFLOAT2 uv;

    D3DVertex(const Vertex& vertex)
            : position(vertex.position.x, vertex.position.y, vertex.position.z),
              normal(vertex.normal.x, vertex.normal.y, vertex.normal.z),
              uv(vertex.uv.x, vertex.uv.y) {}
};
#endif //PATHTRACER_D3DVERTEX_H
