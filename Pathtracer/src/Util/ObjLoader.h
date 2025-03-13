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
constexpr int NUM_SAMPLES_MC = 16000; // Monte Carlo samples per integral

#include <DirectXMath.h>
#include <DirectXPackedVector.h>
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

// Subtract a scalar from an XMFLOAT3
inline XMFLOAT3 operator-(const float& b, const XMFLOAT3& a) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b); // Replicate scalar to all components
    XMVECTOR result = XMVectorSubtract(vb, va);
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
    float alpha2 = alpha*alpha;
    float denomC = sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV;

    return 2.0f * NdotV / std::max(denomC, 1e-7f); // Avoid division by zero
}

// Smith Geometry Function G2
inline float G2_SmithGGX(float NdotV, float NdotL, float alpha) {
    float alpha2 = alpha*alpha;
    float denomA = NdotV * sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    float denomB = NdotL * sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
    return 2.0f * NdotL * NdotV / (denomA + denomB);
}

// Constructs an orthonormal basis (T1, T2) given a normal vector N
void CoordinateSystem(const XMFLOAT3& N, XMFLOAT3& T1, XMFLOAT3& T2) {
    if (fabs(N.z) < 0.999f) {
        T1 = normalize(cross(XMFLOAT3(0.0f, 0.0f, 1.0f), N));
    } else {
        T1 = normalize(cross(XMFLOAT3(1.0f, 0.0f, 0.0f), N));
    }
    T2 = cross(N, T1);
}

// SampleGGX Function
void SampleGGX(
        const Material& mat,
        const XMFLOAT3& outgoing,      // View direction (V)
        const XMFLOAT3& normal,        // Surface normal (N)
        XMFLOAT3& sample,
        float e0,
        float e1// Output sample direction (L)
)
{
    // Extract and compute alpha (roughness squared)
    float alpha = mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x;

    // Normalize input vectors
    XMFLOAT3 N = normalize(normal);
    XMFLOAT3 V = normalize(outgoing);


    // Construct orthonormal basis (T1, T2, N)
    XMFLOAT3 T1, T2;
    CoordinateSystem(N, T1, T2);

    // Transform view vector V into the tangent space
    float VdotT1 = dot(T1, V);
    float VdotT2 = dot(T2, V);
    float VdotN = dot(N, V);
    XMFLOAT3 Vh = normalize(XMFLOAT3(VdotT1, VdotT2, VdotN));

    // Stretch the view vector by alpha
    float alpha_x = alpha;
    float alpha_y = alpha;
    XMFLOAT3 Vh_stretched = normalize(XMFLOAT3(alpha_x * Vh.x, alpha_y * Vh.y, Vh.z));

    // Build an orthonormal basis for the stretched space
    float lensq = Vh_stretched.x * Vh_stretched.x + Vh_stretched.y * Vh_stretched.y;
    XMFLOAT3 T1h, T2h;
    if (lensq > 0.0f)
    {
        float invSqrtLensq = 1.0f / sqrtf(lensq);
        T1h = normalize(XMFLOAT3(-Vh_stretched.y * invSqrtLensq, Vh_stretched.x * invSqrtLensq, 0.0f));
        T2h = cross(Vh_stretched, T1h);
    }
    else
    {
        T1h = normalize(XMFLOAT3(1.0f, 0.0f, 0.0f));
        T2h = normalize(XMFLOAT3(0.0f, 1.0f, 0.0f));
    }

    // Sample point on unit disk using polar coordinates
    float r = sqrtf(e0);
    float phi = 2.0f * static_cast<float>(XM_PI) * e1;
    float x = r * cosf(phi);
    float y = r * sinf(phi);

    // Compute normal in stretched hemisphere
    float z = sqrtf(std::max(0.0f, 1.0f - x * x - y * y));
    XMFLOAT3 Nh_stretched = normalize(XMFLOAT3(
            x * T1h.x + y * T2h.x + z * Vh_stretched.x,
            x * T1h.y + y * T2h.y + z * Vh_stretched.y,
            x * T1h.z + y * T2h.z + z * Vh_stretched.z
    ));

    // Unstretch the normal
    XMFLOAT3 Nh = normalize(XMFLOAT3(alpha_x * Nh_stretched.x, alpha_y * Nh_stretched.y, std::max(0.0f, Nh_stretched.z)));

    // Transform back to world space
    XMFLOAT3 H = normalize(XMFLOAT3(
            Nh.x * T1.x + Nh.y * T2.x + Nh.z * N.x,
            Nh.x * T1.y + Nh.y * T2.y + Nh.z * N.y,
            Nh.x * T1.z + Nh.y * T2.z + Nh.z * N.z
    ));

    // Reflect view vector V about H to get sample direction L
    XMFLOAT3 negV = V * (-1.0f);
    XMFLOAT3 L = reflect(negV, H);
    sample = normalize(L);

}


// Sample GGX VNDF for sampling H
/*XMFLOAT3 SampleGGX_H(const XMFLOAT3& V, float roughness, float u1, float u2) {
    float alpha = roughness * roughness;

    float phi = 2.0f * PI * u1;
    float cosTheta = sqrt((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    float x = sinTheta * cosf(phi);
    float y = sinTheta * sinf(phi);
    float z = cosTheta;

    // Create orthonormal basis (T1, T2, N)
    XMFLOAT3 N = {0.0f, 0.0f, 1.0f}; // Assuming N is along the z-axis
    XMFLOAT3 T1, T2;
    CoordinateSystem(N, T1, T2);

    // Transform H from tangent space to world space
    XMFLOAT3 H_world = normalize(XMFLOAT3(
            x * T1.x + y * T2.x + z * N.x,
            x * T1.y + y * T2.y + z * N.y,
            x * T1.z + y * T2.z + z * N.z
    ));

    return H_world;
}

// Updated SampleGGX Function using standard GGX VNDF sampling
void SampleGGX(
        const Material& mat,
        const XMFLOAT3& outgoing,      // View direction (V)
        const XMFLOAT3& normal,        // Surface normal (N)
        XMFLOAT3& sample,              // Output L
        float u1,
        float u2
) {
    // Normalize vectors
    XMFLOAT3 N = normalize(normal);
    XMFLOAT3 V = normalize(outgoing);

    // Sample H from GGX distribution
    XMFLOAT3 H = SampleGGX_H(V, mat.Pr_Pm_Ps_Pc.x, u1, u2);

    // Compute L as reflection of V about H
    XMFLOAT3 negV = V * (-1.0f);
    XMFLOAT3 L = reflect(negV, H);
    sample = normalize(L);
}*/




// Evaluate GGX BRDF
XMFLOAT3 EvaluateBRDF_GGX(const XMFLOAT3& V, const XMFLOAT3& L, const XMFLOAT3& N, const XMFLOAT3& F0, float roughness) {
    XMFLOAT3 H = normalize(V + L);
    float NdotV = std::max(dot(N, V), 0.0f);
    float NdotL = std::max(dot(N, L), 0.0f);
    float NdotH = std::max(dot(N, H), 0.0f);
    float VdotH = std::max(dot(V, H), 0.0f);

    XMFLOAT3 F = XMFLOAT3(1.0f,1.0f,1.0f);
    float D = D_GGX(NdotH, roughness);
    float G = G2_SmithGGX(NdotV, NdotL, roughness*roughness);

    return F * G / std::max(4.0f * NdotV * NdotL, 1e-7f);
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

    float denom = std::max(NdotV * 4.0f, 1e-7f); // Avoid division by zero
    return G1 / denom;

    /*float denom = 4.0f * VdotH;
    return NdotH  / denom;*/
}



// Compute E_ss with Monte Carlo Integration
float ComputeEss(const XMFLOAT3& N, const XMFLOAT3& V, float roughness, XMFLOAT3 Ks, int numSamples, Material& mat) {
    float Ess = 0.0f;

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);


    for (int i = 0; i < numSamples; ++i) {
        float u1 = dist(gen);
        float u2 = dist(gen);

        // Sample GGX
        XMFLOAT3 L;
        SampleGGX(mat,V,N,L,u1,u2);

        // Ensure L is valid
        if (dot(N, L) <= 0.0f) continue;

        float NdotL = std::abs(dot(normalize(N), normalize(L)));
        XMFLOAT3 brdf = EvaluateBRDF_GGX(normalize(V), normalize(L), normalize(N), Ks, roughness);

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




void PrintLUTAsVector(const Material& mat) {
    std::wcout << L"1D LUT (Indexed by cosTheta):\n\n";

    // Print column headers for cosTheta values
    for (int idx = 0; idx < LUT_SIZE_THETA; ++idx) {
        float cosTheta = static_cast<float>(idx) / (LUT_SIZE_THETA - 1); // Normalize index to [0, 1]

        // Print cosTheta value as label
        std::wcout << L"cosTheta = " << std::fixed << std::setprecision(2) << cosTheta << L": ";

        // Print LUT value at this index
        std::wcout << std::fixed << std::setprecision(3) << 1.0f + (1.0f -mat.LUT[idx]) / mat.LUT[idx] << L"\n";
    }
}


void GenerateEssLUT(Material& mat) {
    constexpr float EPSILON = 0.04f; // Small value to replace 0

    // Start measuring time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Loop over theta (view angle cosine)
    for (int thetaIdx = 0; thetaIdx < LUT_SIZE_THETA; ++thetaIdx) {
        // Replace 0 with EPSILON for cosTheta
        float cosTheta = EPSILON + static_cast<float>(thetaIdx) / (LUT_SIZE_THETA - 1) * (1.0f - EPSILON);

        // Ensure sinTheta is calculated safely
        float sinTheta = sqrt(std::max(EPSILON, 1.0f - cosTheta * cosTheta));

        // Compute normal and view direction
        XMFLOAT3 N = {0.0f, 0.0f, 1.0f}; // Fixed normal
        XMFLOAT3 V = {sinTheta, 0.0f, cosTheta}; // View vector aligned with cosTheta

        // Compute E_ss using Monte Carlo integration
        mat.LUT[thetaIdx] = ComputeEss(N, V, mat.Pr_Pm_Ps_Pc.x, XMFLOAT3(1.0f, 1.0f, 1.0f), NUM_SAMPLES_MC, mat);

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
            Material t_mat(diffuse, Pr_Pm_Ps_Pc);

            // Set emission
            t_mat.Ke = XMFLOAT3(mat.emission);
            t_mat.Ks = XMFLOAT3(mat.specular);

            //Calculate LUT
            GenerateEssLUT(t_mat);
            PrintLUTAsVector(t_mat);
            //TestKMOffsetGGX(200,t_mat);

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
