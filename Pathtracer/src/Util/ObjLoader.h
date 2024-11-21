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

#include <cmath>
#include <random>
#include <iostream>
#include <vector>

#include <chrono>
#include <iomanip>

constexpr float PI = 3.14159265359f;
constexpr int LUT_SIZE_THETA = 16; // Number of samples for cos(theta)
constexpr int LUT_SIZE_ROUGHNESS = 16; // Number of samples for roughness
constexpr int NUM_SAMPLES_MC = 10000; // Monte Carlo samples per integral

#include <DirectXMath.h>
using namespace DirectX;

// Add two XMFLOAT3
inline XMFLOAT3 operator+(const XMFLOAT3& a, const XMFLOAT3& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMLoadFloat3(&b);
    XMVECTOR result = XMVectorAdd(va, vb);
    XMFLOAT3 sum;
    XMStoreFloat3(&sum, result);
    return sum;
}

// Subtract two XMFLOAT3
inline XMFLOAT3 operator-(const XMFLOAT3& a, const XMFLOAT3& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMLoadFloat3(&b);
    XMVECTOR result = XMVectorSubtract(va, vb);
    XMFLOAT3 diff;
    XMStoreFloat3(&diff, result);
    return diff;
}

// Add a scalar to an XMFLOAT3
inline XMFLOAT3 operator+(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b); // Replicate scalar to all components
    XMVECTOR result = XMVectorAdd(va, vb);
    XMFLOAT3 sum;
    XMStoreFloat3(&sum, result);
    return sum;
}

// Subtract a scalar from an XMFLOAT3
inline XMFLOAT3 operator-(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b); // Replicate scalar to all components
    XMVECTOR result = XMVectorSubtract(va, vb);
    XMFLOAT3 diff;
    XMStoreFloat3(&diff, result);
    return diff;
}

// Multiply an XMFLOAT3 by a scalar
inline XMFLOAT3 operator*(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b); // Replicate scalar to all components
    XMVECTOR result = XMVectorMultiply(va, vb);
    XMFLOAT3 product;
    XMStoreFloat3(&product, result);
    return product;
}

// Divide an XMFLOAT3 by a scalar
inline XMFLOAT3 operator/(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b); // Replicate scalar to all components
    XMVECTOR result = XMVectorDivide(va, vb);
    XMFLOAT3 quotient;
    XMStoreFloat3(&quotient, result);
    return quotient;
}




// Cross product
inline XMFLOAT3 cross(const XMFLOAT3& a, const XMFLOAT3& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMLoadFloat3(&b);
    XMVECTOR result = XMVector3Cross(va, vb);
    XMFLOAT3 crossProduct;
    XMStoreFloat3(&crossProduct, result);
    return crossProduct;
}

// Dot product
inline float dot(const XMFLOAT3& a, const XMFLOAT3& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMLoadFloat3(&b);
    return XMVectorGetX(XMVector3Dot(va, vb));
}

// Normalize a vector
inline XMFLOAT3 normalize(const XMFLOAT3& v) {
    XMVECTOR vec = XMLoadFloat3(&v);
    XMVECTOR norm = XMVector3Normalize(vec);
    XMFLOAT3 normalizedVec;
    XMStoreFloat3(&normalizedVec, norm);
    return normalizedVec;
}

// Reflect a vector
inline XMFLOAT3 reflect(const XMFLOAT3& I, const XMFLOAT3& N) {
    XMVECTOR vi = XMLoadFloat3(&I);
    XMVECTOR vn = XMLoadFloat3(&N);
    XMVECTOR reflected = XMVector3Reflect(vi, vn);
    XMFLOAT3 reflectedVec;
    XMStoreFloat3(&reflectedVec, reflected);
    return reflectedVec;
}




// Schlick Fresnel approximation
inline XMFLOAT3 SchlickFresnel(const XMFLOAT3& F0, float cosTheta) {
    return F0 + (F0 + (-1.0f)) * pow(std::abs(1.0f - cosTheta), 5.0f);
}

// GGX Distribution Function (D)
inline float D_GGX(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    denom = std::max(denom, 1e-7f);
    return alpha2 / (PI * denom * denom);
}

// Smith's Geometry function 1 for GGX
inline float G1_SmithGGX(float NdotV, float alpha) {
    float alpha2 = alpha * alpha;
    float denomC = sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV;

    return 2.0f * NdotV / std::max(denomC, 1e-7f); // Avoid division by zero
}

// Smith Geometry Function G2
inline float G2_SmithGGX(float NdotV, float NdotL, float alpha) {
    float alpha2 = alpha * alpha;
    float denomA = NdotV * sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    float denomB = NdotL * sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
    return 2.0f * NdotL * NdotV / (denomA + denomB);
}

// GGX Importance Sampling
void SampleGGX(const XMFLOAT3& V, const XMFLOAT3& N, float roughness, float u1, float u2, XMFLOAT3& H, XMFLOAT3& L) {
    float alpha = roughness * roughness;

    // Build coordinate system
    XMFLOAT3 T, B;
    if (std::abs(N.z) < 0.999f) {
        T = normalize(cross({0.0f, 0.0f, 1.0f}, N));
    } else {
        T = normalize(cross({1.0f, 0.0f, 0.0f}, N));
    }
    B = cross(N, T);

    // Sample GGX distribution
    float phi = 2.0f * PI * u1;
    float cosTheta = sqrt((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    // Spherical to Cartesian
    XMFLOAT3 H_local = {sinTheta * cos(phi), sinTheta * sin(phi), cosTheta};
    H = normalize(T * H_local.x + B*H_local.y + N*H_local.z);

    // Reflect the view vector
    L = reflect(V * (-1.0f), H);
}

// Evaluate GGX BRDF
XMFLOAT3 EvaluateBRDF_GGX(const XMFLOAT3& V, const XMFLOAT3& L, const XMFLOAT3& N, const XMFLOAT3& F0, float roughness) {
    XMFLOAT3 H = normalize(V + L);
    float NdotV = std::max(dot(N, V), 0.0f);
    float NdotL = std::max(dot(N, L), 0.0f);
    float NdotH = std::max(dot(N, H), 0.0f);
    float VdotH = std::max(dot(V, H), 0.0f);

    XMFLOAT3 F = SchlickFresnel(F0, VdotH);
    float D = D_GGX(NdotH, roughness);
    float G = G2_SmithGGX(NdotV, NdotL, roughness * roughness);

    return F * D * G / std::max(4.0f * NdotV * NdotL, 1e-7f);
}

// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(const float roughness, const XMFLOAT3& normal, const XMFLOAT3& incoming, const XMFLOAT3& outgoing) {
    XMFLOAT3 N = normalize(normal);
    XMFLOAT3 V = normalize(outgoing);  // View direction
    XMFLOAT3 L = normalize(incoming * -1.0f); // Light direction
    XMFLOAT3 H = normalize(V + L);

    float NdotH = std::max(dot(N, H), 0.0f);
    float VdotH = std::max(dot(V, H), 0.0f);
    float NdotV = std::max(dot(N, V), 0.0f);

    float alpha = roughness * roughness; // Roughness squared
    float G1 = G1_SmithGGX(NdotV, alpha);
    float D = D_GGX(NdotH, roughness);

    float denom = std::max(NdotV * 4.0f, 1e-7f); // Avoid division by zero
    return G1 * D / denom;
}

// Compute E_ss with Monte Carlo Integration
float ComputeEss(const XMFLOAT3& N, const XMFLOAT3& V, float roughness, int numSamples) {
    float Ess = 0.0f;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);


    for (int i = 0; i < numSamples; ++i) {
        float u1 = dist(gen);
        float u2 = dist(gen);

        // Sample GGX
        XMFLOAT3 H, L;
        SampleGGX(V, N, roughness, u1, u2, H, L);

        // Ensure L is valid
        if (dot(N, L) <= 0.0f) continue;

        float NdotL = std::max(dot(N, L), 0.0f);
        XMFLOAT3 brdf = EvaluateBRDF_GGX(V, L, N, {1.0f, 1.0f, 1.0f}, roughness);

        // Calculate the PDF
        float pdf = BRDF_PDF_GGX(roughness, N, L * -1.0f, V);
        pdf = std::max(pdf, 1e-7f); // Avoid division by zero

        // Safeguard against zero or invalid BRDF values
        float luminance = (brdf.x + brdf.y + brdf.z) / 3.0f;
        if (luminance > 0.0f) {
            Ess += (NdotL * luminance) / pdf;
        }
    }

    // Avoid division by zero
    return numSamples > 0 ? Ess / numSamples : 0.0f;
}




void PrintLUTAsMatrix16x16(const Material& mat) {
    std::wcout << L"16x16 LUT (Rows: cosTheta, Columns: Roughness):\n\n";

    // Print column headers (Roughness values)
    std::wcout << L"      "; // Padding for the row labels
    for (int col = 0; col < 16; ++col) {
        float roughness = static_cast<float>(col) / 15.0f; // Normalize column index to [0, 1]
        std::wcout << std::fixed << std::setprecision(2) << roughness << L"  ";
    }
    std::wcout << L"\n";

    // Print rows with cosTheta values as labels
    for (int row = 0; row < 16; ++row) {
        float cosTheta = static_cast<float>(row) / 15.0f; // Normalize row index to [0, 1]

        // Print row label (cosTheta value)
        std::wcout << std::fixed << std::setprecision(2) << cosTheta << L" | ";

        // Print LUT values for this row
        for (int col = 0; col < 16; ++col) {
            std::wcout << std::fixed << std::setprecision(3) << mat.LUT[row][col] << L" ";
        }
        std::wcout << L"\n";
    }
}



void GenerateEssLUT(Material& mat) {
    constexpr float EPSILON = 0.0001f; // Small value to replace 0

    // Start measuring time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Loop over theta (view angle cosine)
    for (int thetaIdx = 0; thetaIdx < LUT_SIZE_THETA; ++thetaIdx) {
        // Replace 0 with EPSILON for cosTheta
        float cosTheta = EPSILON + static_cast<float>(thetaIdx) / (LUT_SIZE_THETA - 1) * (1.0f - EPSILON);

        // Ensure sinTheta is calculated safely
        float sinTheta = sqrt(std::max(EPSILON, 1.0f - cosTheta * cosTheta));

        // Loop over roughness values
        for (int roughIdx = 0; roughIdx < LUT_SIZE_ROUGHNESS; ++roughIdx) {
            // Replace 0 with EPSILON for roughness
            float roughness = EPSILON + static_cast<float>(roughIdx) / (LUT_SIZE_ROUGHNESS - 1) * (1.0f - EPSILON);

            // Compute normal and view direction
            XMFLOAT3 N = {0.0f, 0.0f, 1.0f}; // Fixed normal
            XMFLOAT3 V = {sinTheta, 0.0f, cosTheta}; // View vector aligned with cosTheta

            // Compute E_ss using Monte Carlo integration
            mat.LUT[thetaIdx][roughIdx] = ComputeEss(N, V, roughness, NUM_SAMPLES_MC);
        }

        // Log progress to the console every 10% completed
        if (thetaIdx % (LUT_SIZE_THETA / 10) == 0) {
            std::wcout << L"Progress: " << (thetaIdx * 100 / LUT_SIZE_THETA) << L"% completed\n";
        }
    }

    // Stop measuring time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Log processing time
    std::wcout << L"GenerateEssLUT completed in "
               << duration.count() << L" ms ("
               << std::fixed << std::setprecision(2)
               << duration.count() / 1000.0 << L" seconds)\n";
}







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

            XMFLOAT4 Pr_Pm_Ps_Pc(mat.roughness, mat.metallic, mat.sheen, mat.clearcoat_thickness);
            XMFLOAT2 aniso_anisor(mat.anisotropy, mat.anisotropy_rotation);
            Material t_mat(diffuse, Pr_Pm_Ps_Pc);

            // Set emission
            t_mat.Ke = XMFLOAT3(mat.emission);
            t_mat.Ks = XMFLOAT3(mat.specular);
            t_mat.aniso_anisor = aniso_anisor;

            //Calculate LUT
            GenerateEssLUT(t_mat);
            PrintLUTAsMatrix16x16(t_mat);

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
                    uint32_t id = materialID + *materialOffset;
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
                    XMFLOAT4 normal(0.0f, 0.0f, 0.0f, *materialVertexOffset); // Default normal
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
