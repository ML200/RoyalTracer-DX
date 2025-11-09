#pragma once
#include <DirectXMath.h>


struct TextureData
{
    std::vector<unsigned char> pixels;
    int width;
    int height;
    int channels;
};

struct alignas(16) Material {
    DirectX::XMFLOAT4 Kd;
    DirectX::XMFLOAT3 Ke;
    float Ni;
    DirectX::XMFLOAT4 Pr_Pm_Ps_Pc;
    DirectX::XMFLOAT3 Pcr_aniso_anisor;
    float pcr_pad; // ADDED: Pad the float3 to align to 16 bytes
    float LUT[16];
    float SheenLUT[16];
    int albedoTexID;
    int normalTexID;
    int rmaTexID;
    int pad;       // ADDED: Pad the three ints to align to 16 bytes

    Material() :
        Kd(1.0f, 1.0f, 1.0f, 1.0f), Ke(0.0f, 0.0f, 0.0f), Ni(1.0f),
        Pr_Pm_Ps_Pc(0.5f, 0.0f, 0.0f, 0.0f), Pcr_aniso_anisor(0.0f, 0.0f, 0.0f),
        pcr_pad(0.0f), albedoTexID(-1), normalTexID(-1), rmaTexID(-1), pad(0)
    {
        for (int i = 0; i < 16; ++i) {
            LUT[i] = 0.0f;
            SheenLUT[i] = 0.0f;
        }
    }
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

    bool operator==(const Vertex& other) const {
        return position.x == other.position.x && position.y == other.position.y && position.z == other.position.z &&
               normal_material.x == other.normal_material.x && normal_material.y == other.normal_material.y && normal_material.z == other.normal_material.z &&
               texCoord.x == other.texCoord.x && texCoord.y == other.texCoord.y;
    }
};

// Hash function for Vertex
namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            size_t h1 = hash<float>()(vertex.position.x) ^ hash<float>()(vertex.position.y) ^ hash<float>()(vertex.position.z);
            size_t h2 = hash<float>()(vertex.normal_material.x) ^ hash<float>()(vertex.normal_material.y) ^ hash<float>()(vertex.normal_material.z);
            size_t h3 = hash<float>()(vertex.texCoord.x) ^ hash<float>()(vertex.texCoord.y);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}