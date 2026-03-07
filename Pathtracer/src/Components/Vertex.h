#pragma once
#include <DirectXMath.h>
#include <DirectXTex.h>

struct TextureData {
    DirectX::ScratchImage image;
    int width;
    int height;
    int channels;
    int original_width = 0;
    int original_height = 0;
};

struct alignas(16) Material {
    DirectX::XMFLOAT4 Kd;
    DirectX::XMFLOAT3 Ke;
    float Ni;
    DirectX::XMFLOAT4 Pr_Pm_Ps_Pc;
    DirectX::XMFLOAT3 Pcr_aniso_anisor;
    DirectX::XMFLOAT3 Tf;
    DirectX::XMFLOAT2 albedoUVScale = { 1.0f, 1.0f };
    DirectX::XMFLOAT2 normalUVScale = { 1.0f, 1.0f };
    DirectX::XMFLOAT2 rmaUVScale    = { 1.0f, 1.0f };
    int albedoTexID;
    int normalTexID;
    int rmaTexID;
    float alphaThreshold;

    Material() :
        Kd(1.0f, 1.0f, 1.0f, 1.0f), Ke(0.0f, 0.0f, 0.0f), Ni(1.0f),
        Pr_Pm_Ps_Pc(0.5f, 0.0f, 0.0f, 0.0f), Pcr_aniso_anisor(0.0f, 0.0f, 0.0f), Tf(1.0f,1.0f,1.0f),
        albedoTexID(-1), normalTexID(-1), rmaTexID(-1), alphaThreshold(1.0f)
    {}
};


struct Vertex {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT4 normal_material; // w component holds material id
    DirectX::XMFLOAT2 texCoord;

    // Default constructor
    Vertex() : position{}, normal_material{}, texCoord{} {}

    // Parameterized constructor
    Vertex(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT4& norm_mat, const DirectX::XMFLOAT2& uv)
        : position(pos), normal_material(norm_mat), texCoord(uv) {}

    // MODIFIED: Now correctly compares all components, including the material ID in .w
    bool operator==(const Vertex& other) const {
        return position.x == other.position.x && position.y == other.position.y && position.z == other.position.z &&
               normal_material.x == other.normal_material.x && normal_material.y == other.normal_material.y && normal_material.z == other.normal_material.z &&
               normal_material.w == other.normal_material.w && // <-- CRITICAL FIX
               texCoord.x == other.texCoord.x && texCoord.y == other.texCoord.y;
    }
};

// Hash function for Vertex
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            // A simple hash combining position, normal, texcoord, and material ID
            size_t h1 = hash<float>()(vertex.position.x) ^ hash<float>()(vertex.position.y) ^ hash<float>()(vertex.position.z);
            // MODIFIED: Now correctly hashes all components of normal_material
            size_t h2 = hash<float>()(vertex.normal_material.x) ^ hash<float>()(vertex.normal_material.y) ^ hash<float>()(vertex.normal_material.z) ^ hash<float>()(vertex.normal_material.w); // <-- CRITICAL FIX
            size_t h3 = hash<float>()(vertex.texCoord.x) ^ hash<float>()(vertex.texCoord.y);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}