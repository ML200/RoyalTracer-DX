//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_OBJLOADER_H
#define PATHTRACER_OBJLOADER_H
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "../../lib/tiny_obj_loader.h"
#include "../Objects/SceneObject.h"
#include "../Math/Vector2.h"
#include <iostream>
#include <unordered_map>

class ObjLoader{
public:
    static Mesh loadObjFile(const std::string& inputfile, const std::string& material_search_path = "./") {
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = material_search_path; // Path to material files

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(inputfile, reader_config)) {
            if (!reader.Error().empty()) {
                std::cerr << "TinyObjReader: " << reader.Error();
                exit(1);
            }
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader: " << reader.Warning();
        }

        auto& attrib = reader.GetAttrib();
        auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        std::vector<Vertex> vertices;
        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        std::vector<uint32_t> indices;

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                tinyobj::real_t vx = attrib.vertices[3 * index.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * index.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * index.vertex_index + 2];
                Vector3 pos(vx, vy, vz);

                tinyobj::real_t nx = attrib.normals[3 * index.normal_index + 0];
                tinyobj::real_t ny = attrib.normals[3 * index.normal_index + 1];
                tinyobj::real_t nz = attrib.normals[3 * index.normal_index + 2];
                Vector3 norm(nx, ny, nz);

                tinyobj::real_t tx = attrib.texcoords[2 * index.texcoord_index + 0];
                tinyobj::real_t ty = attrib.texcoords[2 * index.texcoord_index + 1];
                Vector2 uv(tx, ty);

                Vertex vertex(pos, norm, uv);

                // Check if this vertex is unique
                if (uniqueVertices.count(vertex) == 0) {
                    // New unique vertex, add it to the vector
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                // Add index for this vertex
                indices.push_back(uniqueVertices[vertex]);
            }
        }

        // Create a default material
        Material defaultMaterial(Vector4(1.0f, 1.0f, 1.0f, 1.0f), Vector4(1.0f, 1.0f, 1.0f, 1.0f), 32.0f);

        return Mesh(vertices, indices, defaultMaterial);
    }
};

#endif //PATHTRACER_OBJLOADER_H
