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
    static void loadObjFile(const std::string& inputfile, std::vector<Vertex> *vertices, std::vector<UINT> *indices, std::vector<Material> *mats, std::vector<UINT> *materialIDs, UINT *materialOffset, UINT *materialVertexOffset, const std::string& material_search_path = "./") {
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
        auto& materials = reader.GetMaterials();

        // Create a default material if a face has no material assigned
        Material defaultMaterial(XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f), XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f)); // Example default material
        mats->push_back(defaultMaterial);
        (*materialOffset)++;

        // Process materials
        for (const auto& mat : materials) {
            // Convert the material name to a wide string (assuming mat.name is a std::string)
            std::wstring wideName(mat.name.begin(), mat.name.end());

            // Print the material name and dissolve value (alpha)
            std::wcout << L"Loading Material: " << wideName << L", Dissolve: " << mat.dissolve << std::endl;

            // Set up material properties
            XMFLOAT4 diffuse(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2], mat.dissolve);
            XMFLOAT4 Pr_Pm_Ps_Pc(mat.roughness, mat.metallic, 0, 0);
            Material t_mat(diffuse, Pr_Pm_Ps_Pc);

            // Set emission
            t_mat.Ke = XMFLOAT3(mat.emission);

            // Add the material to the list
            mats->push_back(t_mat);
        }


        std::unordered_map<Vertex, uint32_t> uniqueVertices;

        for (const auto& shape : shapes) {
            size_t index_offset = 0;
            // For each face
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                int fv = shape.mesh.num_face_vertices[f];

                // Assign material ID for this face, use default if none assigned
                int materialID = shape.mesh.material_ids.size() >= 0 ? shape.mesh.material_ids[f] : -1;
                for (size_t v = 0; v < fv; v++) {
                    int id = materialID + *materialOffset;
                    materialIDs->push_back(id);
                }

                // For each vertex in the face
                for (size_t v = 0; v < fv; v++) {
                    tinyobj::index_t idx = shape.mesh.indices[index_offset + v];
                    tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                    tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                    tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                    XMFLOAT3 pos(vx, vy, vz);

                    // Extract normal, default to (0, 0, 0) if not present
                    XMFLOAT4 normal(0.0f, 0.0f, 0.0f, (float)*materialOffset); // Default normal
                    if (idx.normal_index >= 0) {
                        tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
                        tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
                        tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
                        normal = XMFLOAT4(nx, ny, nz, *materialVertexOffset);
                    }

                    Vertex vertex(pos, normal);

                    // Check if this vertex is unique
                    if (uniqueVertices.count(vertex) == 0) {
                        uniqueVertices[vertex] = static_cast<uint32_t>(vertices->size());
                        vertices->push_back(vertex);
                    }

                    // Add index for this vertex
                    indices->push_back(uniqueVertices[vertex]);
                }
                index_offset += fv;
            }
        }

        *materialOffset+=materials.size();
    }
};


#endif //PATHTRACER_OBJLOADER_H
