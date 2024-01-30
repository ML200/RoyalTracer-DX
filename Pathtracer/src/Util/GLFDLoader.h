//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_GLFDLOADER_H
#define PATHTRACER_GLFDLOADER_H


#include <iostream>
#include <string>
#include <vector>
#include "libs/tiny_gltf.h"

class GLDFLoader {
public:
    GLDFLoader() {}

    // Load a 3D model and associated materials from a glTF file
    bool LoadModel(const std::string& filePath) {
        tinygltf::Model model;
        tinygltf::TinyGLTF loader;
        std::string err;
        std::string warn;

        // Load the glTF model
        bool result = loader.LoadASCIIFromFile(&model, &err, &warn, filePath);

        if (!warn.empty()) {
            std::cerr << "Warning: " << warn << std::endl;
        }

        if (!err.empty()) {
            std::cerr << "Error: " << err << std::endl;
            return false;
        }

        if (!result) {
            std::cerr << "Failed to load glTF model." << std::endl;
            return false;
        }

        // Extract model data
        if (!ExtractModelData(model)) {
            std::cerr << "Failed to extract model data." << std::endl;
            return false;
        }

        return true;
    }

    // Print information about the loaded model
    void PrintModelInfo() {
        std::cout << "Model Name: " << model_.name << std::endl;

        // Print information about each mesh
        for (const auto& mesh : model_.meshes) {
            std::cout << "Mesh Name: " << mesh.name << std::endl;

            // Print information about each primitive
            for (const auto& primitive : mesh.primitives) {
                std::cout << "Primitive:" << std::endl;
                std::cout << "  - Material Index: " << primitive.material << std::endl;
                std::cout << "  - Vertex Count: " << primitive.attributes.size() << std::endl;
            }
        }

        // Print information about each material
        for (const auto& material : model_.materials) {
            std::cout << "Material Name: " << material.name << std::endl;
            // Print other material properties as needed
        }
    }

private:
    struct MeshInfo {
        std::string name;
        std::vector<tinygltf::Primitive> primitives;
    };

    struct ModelInfo {
        std::string name;
        std::vector<MeshInfo> meshes;
        std::vector<tinygltf::Material> materials;
    };

    ModelInfo model_;

    // Extract model data from the glTF model
    bool ExtractModelData(const tinygltf::Model& gltfModel) {
        model_.name = gltfModel.asset.version;

        // Extract meshes
        for (const auto& gltfMesh : gltfModel.meshes) {
            MeshInfo meshInfo;
            meshInfo.name = gltfMesh.name;
            meshInfo.primitives = gltfMesh.primitives;
            model_.meshes.push_back(meshInfo);
        }

        // Extract materials
        model_.materials = gltfModel.materials;

        return true;
    }
};



#endif //PATHTRACER_GLFDLOADER_H
