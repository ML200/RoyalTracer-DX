//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VERTEX_H
#define PATHTRACER_VERTEX_H


#include "../Math/Vector3.h"
#include "../Math/Vector2.h"
#include <functional>

class Vertex {
public:
    Vector3 position;
    Vector3 normal;
    Vector2 uv; // UV should be a 2D vector

    Vertex(const Vector3& pos, const Vector3& norm, const Vector2& uv)
            : position(pos), normal(norm), uv(uv) {}

    bool operator==(const Vertex& other) const {
        return position == other.position;
    }
};

namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(const Vertex& vertex) const {
            // Implement your hashing logic here
            // A simple example (you might need a better hash combination method):
            return hash<Vector3>()(vertex.position);
        }
    };
}


#endif //PATHTRACER_VERTEX_H
