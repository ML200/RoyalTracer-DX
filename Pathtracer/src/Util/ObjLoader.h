//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_OBJLOADER_H
#define PATHTRACER_OBJLOADER_H
#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation. Requires C++11
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "../../lib/tiny_obj_loader.h"
#include <iostream>
#include <unordered_map>

class ObjLoader {
public:
    static void loadObjFile(const std::string& inputfile, std::vector<Vertex> *vertices, std::vector<UINT> *indices, std::vector<Material> *mats, std::vector<UINT> *materialIDs, const std::string& material_search_path = "./") {
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

        const auto& attrib = reader.GetAttrib();
        const auto& shapes = reader.GetShapes();
        const auto& materials = reader.GetMaterials();

        // Process materials
        for (const auto& mat : materials) {
            XMFLOAT4 diffuse(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2], 1.0f);
            mats->push_back(Material(diffuse));
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices;

        for (const auto& shape : shapes) {
            size_t index_offset = 0;
            // For each face
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                int fv = shape.mesh.num_face_vertices[f];

                // For each vertex in the face
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                    XMFLOAT3 pos(vx, vy, vz);

                    Vertex vertex(pos, XMFLOAT4 {1,0,1,1});

                    // Check if this vertex is unique
                    if (uniqueVertices.count(vertex) == 0) {
                        uniqueVertices[vertex] = static_cast<uint32_t>(vertices->size());
                        vertices->push_back(vertex);
                    }

                    // Add index for this vertex
                    indices->push_back(uniqueVertices[vertex]);
                }
                // Assign material ID for this face
                if (shape.mesh.material_ids[f] >= 0) {
                    for (size_t v = 0; v < fv; v++) {
                        materialIDs->push_back(shape.mesh.material_ids[f]);
                    }
                } else {
                    // Optional: Handle faces without a material
                }
                index_offset += fv;
            }
        }
    }
};


#endif //PATHTRACER_OBJLOADER_H
