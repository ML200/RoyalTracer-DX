//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_MESH_H
#define PATHTRACER_MESH_H


#include <vector>
#include "Material.h"
#include "Vertex.h"

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices; // For indexed drawing
    Material material; // Material properties for the mesh

    Mesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, const Material& mat)
            : vertices(vertices), indices(indices), material(mat) {}
};


#endif //PATHTRACER_MESH_H
