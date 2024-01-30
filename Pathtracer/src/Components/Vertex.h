//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VERTEX_H
#define PATHTRACER_VERTEX_H


#include "../Math/Vector3.h"

class Vertex {
    Vector3 position;
    Vector3 normal;
    Vector3 uv;

    Vertex(const Vector3& pos, const Vector3& norm, const Vector3& uv)
            : position(pos), normal(norm), uv(uv) {}
};


#endif //PATHTRACER_VERTEX_H
