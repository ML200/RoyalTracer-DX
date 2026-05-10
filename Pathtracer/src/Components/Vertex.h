#pragma once
#include <DirectXMath.h>
#include <DirectXTex.h>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

struct TextureData {
    DirectX::ScratchImage image;
    int width;
    int height;
    int channels;
    int original_width = 0;
    int original_height = 0;
};

// Authoring / import-time material. Populated by OBJ/GLTF loaders and the
// editor, then absorbed field-by-field into MaterialSoA (below). Ke stays
// CPU-only — emission travels to the GPU via the LightTriangle buffer.
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
    bool invertAlpha = false;  // true = sample is transparency (1=transparent), AnyHit will flip the test

    Material() :
        Kd(1.0f, 1.0f, 1.0f, 1.0f), Ke(0.0f, 0.0f, 0.0f), Ni(1.0f),
        Pr_Pm_Ps_Pc(0.5f, 0.0f, 0.0f, 0.0f), Pcr_aniso_anisor(0.0f, 0.0f, 0.0f), Tf(1.0f,1.0f,1.0f),
        albedoTexID(-1), normalTexID(-1), rmaTexID(-1), alphaThreshold(1.0f)
    {}
};

// ─────────────────────────────────────────────────────────────────────────
// Material packing helpers. Mirror the HLSL-side Unpack/Load routines in
// Compression_v8.hlsli and Material_Decoder_v8.hlsli so CPU packing and
// GPU unpacking stay bit-exact.
// ─────────────────────────────────────────────────────────────────────────
namespace MaterialPack {

// IEEE half (binary16) encoder — matches f32tof16_custom in HLSL.
// Subnormal input flushes to zero; Inf / NaN preserved.
inline uint16_t PackHalf(float v)
{
    uint32_t f = 0;
    std::memcpy(&f, &v, 4);
    const uint32_t sign = (f >> 31) & 0x1u;
    const uint32_t exp  = (f >> 23) & 0xFFu;
    const uint32_t mant =  f        & 0x7FFFFFu;
    if (exp == 0xFFu) return uint16_t((sign << 15) | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp == 0u)    return uint16_t(sign << 15);
    int ne = int(exp) - 127;
    if (ne < -14) return uint16_t(sign << 15);
    if (ne >  15) return uint16_t((sign << 15) | 0x7C00u);
    return uint16_t((sign << 15) | (uint32_t(ne + 15) << 10) | (mant >> 13));
}

// RGB9E5 encoder — bit-for-bit matches PackRGB9E5 in Compression_v8.hlsli,
// including the post-quantization overflow fix (mantissa == 512 → shift +1
// and bump shared exponent).
inline uint32_t PackRGB9E5(float r, float g, float b)
{
    static constexpr float kMax = 65408.0f;   // (511/512) * 2^16
    const float rC = std::max(0.0f, std::min(kMax, r));
    const float gC = std::max(0.0f, std::min(kMax, g));
    const float bC = std::max(0.0f, std::min(kMax, b));
    const float m  = std::max(rC, std::max(gC, bC));

    uint32_t bits = 0;
    std::memcpy(&bits, &m, 4);
    const int      expUnb = int((bits >> 23) & 0xFFu) - 127;
    const uint32_t frac   = bits & 0x7FFFFFu;

    int sharedExp = expUnb + (frac != 0u ? 1 : 0) + 15;
    sharedExp = std::max(0, std::min(31, sharedExp));

    const float denom = std::ldexp(1.0f, sharedExp - 15 - 9);

    uint32_t rm = (m > 0.0f) ? uint32_t(std::floor(rC / denom + 0.5f)) : 0u;
    uint32_t gm = (m > 0.0f) ? uint32_t(std::floor(gC / denom + 0.5f)) : 0u;
    uint32_t bm = (m > 0.0f) ? uint32_t(std::floor(bC / denom + 0.5f)) : 0u;

    const uint32_t maxMant = std::max(rm, std::max(gm, bm));
    if (maxMant > 511u) {
        rm >>= 1; gm >>= 1; bm >>= 1;
        sharedExp = std::min(sharedExp + 1, 31);
    }
    rm &= 511u; gm &= 511u; bm &= 511u;

    return rm | (gm << 9) | (bm << 18) | (uint32_t(sharedExp) << 27);
}

inline uint8_t PackUnorm8(float v)
{
    const float c = std::max(0.0f, std::min(1.0f, v));
    return uint8_t(std::floor(c * 255.0f + 0.5f));
}
// Signed int8 in [-1, 1] → [-127, 127].
inline uint8_t PackSnorm8(float v)
{
    const float c = std::max(-1.0f, std::min(1.0f, v));
    const int   q = int(std::floor(c * 127.0f + (c >= 0.0f ? 0.5f : -0.5f)));
    return uint8_t(q & 0xFF);
}

// Pack one material into 10 consecutive uint32s (40 B, matches
// MatPacked in Data_v8.hlsli exactly).
inline void PackOne(const Material& m, uint32_t dst[10])
{
    dst[0] = PackRGB9E5(m.Kd.x, m.Kd.y, m.Kd.z);
    dst[1] = uint32_t(PackHalf(m.Kd.w))
           | (uint32_t(PackHalf(m.Ni)) << 16);
    dst[2] = uint32_t(PackUnorm8(m.Pr_Pm_Ps_Pc.x))
           | (uint32_t(PackUnorm8(m.Pr_Pm_Ps_Pc.y)) <<  8)
           | (uint32_t(PackUnorm8(m.Pr_Pm_Ps_Pc.z)) << 16)
           | (uint32_t(PackUnorm8(m.Pr_Pm_Ps_Pc.w)) << 24);

    dst[3] = PackRGB9E5(m.Tf.x, m.Tf.y, m.Tf.z);
    dst[4] = uint32_t(PackUnorm8(m.Pcr_aniso_anisor.x))
           | (uint32_t(PackSnorm8(m.Pcr_aniso_anisor.y)) <<  8)
           | (uint32_t(PackUnorm8(m.Pcr_aniso_anisor.z)) << 16)
           | (uint32_t(PackUnorm8(m.alphaThreshold))     << 24);

    // Tex IDs: clamp to int16, store -1 as 0xFFFF (sign-extended on GPU).
    const auto clip16 = [](int v) -> uint16_t {
        v = std::max(-32768, std::min(32767, v));
        return uint16_t(v & 0xFFFF);
    };
    dst[5] = uint32_t(clip16(m.albedoTexID))
           | (uint32_t(clip16(m.normalTexID)) << 16);
    // texIDs_2: bits 0..15 = rmaTexID, bit 16 = invertAlpha flag (high 15 still spare)
    dst[6] = uint32_t(clip16(m.rmaTexID))
           | (m.invertAlpha ? (1u << 16) : 0u);
    dst[7] = uint32_t(PackHalf(m.albedoUVScale.x))
           | (uint32_t(PackHalf(m.albedoUVScale.y)) << 16);
    dst[8] = uint32_t(PackHalf(m.normalUVScale.x))
           | (uint32_t(PackHalf(m.normalUVScale.y)) << 16);
    dst[9] = uint32_t(PackHalf(m.rmaUVScale.x))
           | (uint32_t(PackHalf(m.rmaUVScale.y)) << 16);
}

} // namespace MaterialPack


// SoA material storage. Full-precision authoring arrays + bindless tex IDs.
// Editor and loaders mutate the per-field vectors in place, GPU upload
// packs them into the compressed AoS upload buffer via BuildGpuPacked().
//
// push_back(Material) and append(std::vector<Material>) provide a
// drop-in-replacement API for the old std::vector<Material>.
struct MaterialSoA {
    std::vector<DirectX::XMFLOAT4> Kd;
    std::vector<DirectX::XMFLOAT3> Ke;              // CPU-only
    std::vector<float>             Ni;
    std::vector<DirectX::XMFLOAT4> Pr_Pm_Ps_Pc;
    std::vector<DirectX::XMFLOAT3> Pcr_aniso_anisor;
    std::vector<DirectX::XMFLOAT3> Tf;
    std::vector<DirectX::XMFLOAT2> albedoUVScale;
    std::vector<DirectX::XMFLOAT2> normalUVScale;
    std::vector<DirectX::XMFLOAT2> rmaUVScale;
    std::vector<int32_t>           albedoTexID;
    std::vector<int32_t>           normalTexID;
    std::vector<int32_t>           rmaTexID;
    std::vector<float>             alphaThreshold;
    std::vector<uint8_t>           invertAlpha;     // 0/1, AnyHit flips the cutout test when 1

    size_t size()  const { return Ni.size(); }
    bool   empty() const { return Ni.empty(); }

    void reserve(size_t n)
    {
        Kd.reserve(n); Ke.reserve(n); Ni.reserve(n);
        Pr_Pm_Ps_Pc.reserve(n); Pcr_aniso_anisor.reserve(n); Tf.reserve(n);
        albedoUVScale.reserve(n); normalUVScale.reserve(n); rmaUVScale.reserve(n);
        albedoTexID.reserve(n); normalTexID.reserve(n); rmaTexID.reserve(n);
        alphaThreshold.reserve(n); invertAlpha.reserve(n);
    }

    void push_back(const Material& m)
    {
        Kd.push_back(m.Kd);
        Ke.push_back(m.Ke);
        Ni.push_back(m.Ni);
        Pr_Pm_Ps_Pc.push_back(m.Pr_Pm_Ps_Pc);
        Pcr_aniso_anisor.push_back(m.Pcr_aniso_anisor);
        Tf.push_back(m.Tf);
        albedoUVScale.push_back(m.albedoUVScale);
        normalUVScale.push_back(m.normalUVScale);
        rmaUVScale.push_back(m.rmaUVScale);
        albedoTexID.push_back(m.albedoTexID);
        normalTexID.push_back(m.normalTexID);
        rmaTexID.push_back(m.rmaTexID);
        alphaThreshold.push_back(m.alphaThreshold);
        invertAlpha.push_back(m.invertAlpha ? 1u : 0u);
    }

    void append(const std::vector<Material>& src)
    {
        reserve(size() + src.size());
        for (const auto& m : src) push_back(m);
    }

    // Round-trip helper: produce a Material view for a given index.
    Material Get(size_t i) const
    {
        Material m;
        m.Kd = Kd[i]; m.Ke = Ke[i]; m.Ni = Ni[i];
        m.Pr_Pm_Ps_Pc = Pr_Pm_Ps_Pc[i];
        m.Pcr_aniso_anisor = Pcr_aniso_anisor[i];
        m.Tf = Tf[i];
        m.albedoUVScale = albedoUVScale[i];
        m.normalUVScale = normalUVScale[i];
        m.rmaUVScale = rmaUVScale[i];
        m.albedoTexID = albedoTexID[i];
        m.normalTexID = normalTexID[i];
        m.rmaTexID = rmaTexID[i];
        m.alphaThreshold = alphaThreshold[i];
        m.invertAlpha = (i < invertAlpha.size()) ? (invertAlpha[i] != 0u) : false;
        return m;
    }

    // Pack into one flat uint32 array (10 uint32 per material = 40 B)
    // for direct GPU upload. Matches the HLSL `MatPacked` struct.
    void BuildGpuPacked(std::vector<uint32_t>& out) const
    {
        const size_t n = size();
        out.resize(n * 10);
        for (size_t i = 0; i < n; ++i) {
            Material m = Get(i);
            MaterialPack::PackOne(m, &out[i * 10]);
        }
    }

    // Repack just material `i` into the flat buffer (caller provides the
    // whole contiguous array). Used when the editor modifies a single
    // material and we want to avoid a full rebuild.
    void PackInto(size_t i, uint32_t* buf) const
    {
        Material m = Get(i);
        MaterialPack::PackOne(m, buf + i * 10);
    }
};


struct Vertex {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT4 normal_material; // w component holds material id
    DirectX::XMFLOAT2 texCoord;

    Vertex() : position{}, normal_material{}, texCoord{} {}

    Vertex(const DirectX::XMFLOAT3& pos, const DirectX::XMFLOAT4& norm_mat, const DirectX::XMFLOAT2& uv)
        : position(pos), normal_material(norm_mat), texCoord(uv) {}

    bool operator==(const Vertex& other) const {
        return position.x == other.position.x && position.y == other.position.y && position.z == other.position.z &&
               normal_material.x == other.normal_material.x && normal_material.y == other.normal_material.y && normal_material.z == other.normal_material.z &&
               normal_material.w == other.normal_material.w &&
               texCoord.x == other.texCoord.x && texCoord.y == other.texCoord.y;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            size_t h1 = hash<float>()(vertex.position.x) ^ hash<float>()(vertex.position.y) ^ hash<float>()(vertex.position.z);
            size_t h2 = hash<float>()(vertex.normal_material.x) ^ hash<float>()(vertex.normal_material.y) ^ hash<float>()(vertex.normal_material.z) ^ hash<float>()(vertex.normal_material.w);
            size_t h3 = hash<float>()(vertex.texCoord.x) ^ hash<float>()(vertex.texCoord.y);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}