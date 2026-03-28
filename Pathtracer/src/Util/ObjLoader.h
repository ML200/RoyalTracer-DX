//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_OBJLOADER_H
#define PATHTRACER_OBJLOADER_H
#define TINYOBJLOADER_IMPLEMENTATION
#define TINYGLTF3_IMPLEMENTATION
#define TINYGLTF3_ENABLE_STB_IMAGE
//#define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "../../lib/tiny_obj_loader.h"
#include "../../lib/tiny_gltf_v3.h"
#include <iostream>
#include <unordered_map>

#include <cmath>
#include <random>
#include <iostream>
#include <vector>

#include <chrono>
#include <iomanip>

#include <string>
#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

struct TextureData;
constexpr float PI = 3.14159265359f;
// Texture packing
constexpr int TARGET_TEXTURE_DIM = 2048;

#include <DirectXMath.h>
#include <DirectXPackedVector.h>
using namespace DirectX;


// HELPERS
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
    XMVECTOR vb = XMVectorReplicate(b);
    XMVECTOR result = XMVectorAdd(va, vb);
    XMFLOAT3 sum;
    XMStoreFloat3(&sum, result);
    return sum;
}

// Subtract a scalar from an XMFLOAT3
inline XMFLOAT3 operator-(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b);
    XMVECTOR result = XMVectorSubtract(va, vb);
    XMFLOAT3 diff;
    XMStoreFloat3(&diff, result);
    return diff;
}

// Subtract a scalar from an XMFLOAT3
inline XMFLOAT3 operator-(const float& b, const XMFLOAT3& a) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b);
    XMVECTOR result = XMVectorSubtract(vb, va);
    XMFLOAT3 diff;
    XMStoreFloat3(&diff, result);
    return diff;
}

// Multiply an XMFLOAT3 by a scalar
inline XMFLOAT3 operator*(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b);
    XMVECTOR result = XMVectorMultiply(va, vb);
    XMFLOAT3 product;
    XMStoreFloat3(&product, result);
    return product;
}

// Divide an XMFLOAT3 by a scalar
inline XMFLOAT3 operator/(const XMFLOAT3& a, const float& b) {
    XMVECTOR va = XMLoadFloat3(&a);
    XMVECTOR vb = XMVectorReplicate(b);
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

// GGX Distribution Function
inline float D_GGX(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    denom = (std::fmax)(denom, 1e-7f);
    return alpha2 / (PI * denom * denom);
}

// Smith Geometry function 1 for GGX
inline float G1_SmithGGX(float NdotV, float alpha) {
    float alpha2 = alpha*alpha;
    float denomC = sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV;

    return 2.0f * NdotV / (std::fmax)(denomC, 1e-7f); // Avoid division by zero
}

// Smith Geometry Function G2
inline float G2_SmithGGX(float NdotV, float NdotL, float alpha) {
    return G1_SmithGGX(NdotV, alpha) * G1_SmithGGX(NdotL, alpha);
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
    const XMFLOAT3& outgoing,  // V
    const XMFLOAT3& normal,    // N
    XMFLOAT3& sample,          // L (output)
    float e0, float e1)
{
    float alpha = mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x;

    XMFLOAT3 N = normalize(normal);
    XMFLOAT3 V = normalize(outgoing);

    // Build world-to-local basis
    XMFLOAT3 T1, T2;
    CoordinateSystem(N, T1, T2);

    // Local view
    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = dot(N,  V);

    // Stretch V
    XMFLOAT3 Ve = normalize(XMFLOAT3(alpha * vx, alpha * vy, vz));

    // Orthonormal basis around Ve
    float lensq = Ve.x*Ve.x + Ve.y*Ve.y;
    XMFLOAT3 T1h = (lensq > 0.0f) ? normalize(XMFLOAT3(-Ve.y, Ve.x, 0.0f))
                                  : XMFLOAT3(1.0f, 0.0f, 0.0f);
    XMFLOAT3 T2h = cross(Ve, T1h);

    // Disk sample
    float r   = sqrtf(e0);
    float phi = 2.0f * PI * e1;
    float t1  = r * cosf(phi);
    float t2  = r * sinf(phi);

    float s = 0.5f * (1.0f + Ve.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;

    // Reprojection
    float t3 = sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    XMFLOAT3 Nh = XMFLOAT3(
        t1*T1h.x + t2*T2h.x + t3*Ve.x,
        t1*T1h.y + t2*T2h.y + t3*Ve.y,
        t1*T1h.z + t2*T2h.z + t3*Ve.z
    );

    // Un-stretch and clamp to upper hemisphere
    XMFLOAT3 Ne = normalize(XMFLOAT3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));

    // Back to world half-vector
    XMFLOAT3 H = normalize(XMFLOAT3(
        Ne.x * T1.x + Ne.y * T2.x + Ne.z * N.x,
        Ne.x * T1.y + Ne.y * T2.y + Ne.z * N.y,
        Ne.x * T1.z + Ne.y * T2.z + Ne.z * N.z
    ));

    // Reflect
    sample = normalize(reflect(V * -1.0f, H));
    if (dot(N, sample) <= 0.0f) sample = XMFLOAT3(0,0,0);
}


// Evaluate GGX BRDF
inline XMFLOAT3 EvaluateBRDF_GGX(
    const XMFLOAT3& V,
    const XMFLOAT3& L,
    const XMFLOAT3& N,
    const XMFLOAT3& /*F0_unused*/,
    float roughness)
{
    XMFLOAT3 H  = normalize(V + L);
    float NdotV = (std::fmax)(dot(N, V), 0.0f);
    float NdotL = (std::fmax)(dot(N, L), 0.0f);
    if (NdotV <= 0.0f || NdotL <= 0.0f) return XMFLOAT3(0,0,0);

    float NdotH = (std::fmax)(dot(N, H), 0.0f);

    float D     = D_GGX(NdotH, roughness);
    float alpha = (std::fmax)(1e-4f, roughness * roughness);
    float G2    = G2_SmithGGX(NdotV, NdotL, alpha);

    float denom = (std::fmax)(4.0f * NdotV * NdotL, 1e-7f);
    float brdf  = (D * G2) / denom;

    return XMFLOAT3(brdf, brdf, brdf);
}

// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(const float roughness,
                          const XMFLOAT3& normal,
                          const XMFLOAT3& incoming,
                          const XMFLOAT3& outgoing)
{
    XMFLOAT3 N = normalize(normal);
    XMFLOAT3 V = normalize(outgoing);
    XMFLOAT3 L = normalize(incoming * -1.0f);

    float NdotV = (std::fmax)(dot(N, V), 0.0f);
    float NdotL = (std::fmax)(dot(N, L), 0.0f);
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    XMFLOAT3 H = normalize(V + L);
    float NdotH = (std::fmax)(dot(N, H), 0.0f);

    float D     = D_GGX(NdotH, roughness);
    float alpha = (std::fmax)(1e-4f, roughness * roughness);
    float G1    = G1_SmithGGX(NdotV, alpha);

    return (D * G1) / (4.0f * (std::fmax)(NdotV, 1e-7f));
}


// Compute E_ss with Monte Carlo Integration
float ComputeEss(const XMFLOAT3& N,
                 const XMFLOAT3& V,
                 float roughness,
                 XMFLOAT3 /*Ks*/,
                 int numSamples,
                 Material& mat)
{
    float Ess = 0.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < numSamples; ++i) {
        float u1 = dist(gen);
        float u2 = dist(gen);

        // VNDF sample
        XMFLOAT3 L;
        SampleGGX(mat, V, N, L, u1, u2);

        // Keep to the shading hemisphere
        float NdotL = dot(N, L);
        if (NdotL <= 0.0f) continue;

        // BRDF with F=1 (scalar replicated across RGB using Schlick)
        XMFLOAT3 brdf3 = EvaluateBRDF_GGX(normalize(V), normalize(L), normalize(N), XMFLOAT3(1,1,1), roughness);
        float brdf = brdf3.x;

        // VNDF direction pdf
        float pdf = BRDF_PDF_GGX(roughness, N, L * -1.0f, V);
        pdf = (std::fmax)(pdf, 1e-7f);

        Ess += (NdotL * brdf) / pdf;
    }

    return (numSamples > 0) ? (Ess / numSamples) : 0.0f;
}


// Sheen
inline float D_Charlie(float NdotH, float r)
{
    r = (std::fmax)(1e-4f, r);
    float invr   = 1.0f / r;
    float sin2Th = (std::fmax)(0.0f, 1.0f - NdotH * NdotH);
    float sinTh  = sqrtf(sin2Th);
    return (2.0f + invr) * powf((std::fmax)(1e-8f, sinTh), invr) * (0.5f / PI);
}

// A fit
inline void Sheen_LambdaFitParams(float r, float& a, float& b, float& c, float& d, float& e)
{
    r = (std::fmin)(1.0f, (std::fmax)(0.0f, r));
    float w0 = (1.0f - r); w0 *= w0;
    float w1 = 1.0f - w0;

    const float a0=25.3245f, b0=3.32435f, c0=0.16801f, d0=-1.27393f, e0=-4.85967f;
    const float a1=21.5473f, b1=3.82987f, c1=0.19823f, d1=-1.97760f, e1=-4.32054f;

    a = w0*a0 + w1*a1;
    b = w0*b0 + w1*b1;
    c = w0*c0 + w1*c1;
    d = w0*d0 + w1*d1;
    e = w0*e0 + w1*e1;
}

inline float Sheen_L_eval(float x, float a, float b, float c, float d, float e)
{
    return a / (1.0f + b * powf((std::fmax)(1e-4f, x), c)) + d * x + e;
}

inline float Lambda_Charlie(float cosTheta, float r)
{
    float a,b,c,d,e; Sheen_LambdaFitParams(r, a,b,c,d,e);
    float x = (std::fmin)(1.0f, (std::fmax)(0.0f, cosTheta));
    float Lx    = Sheen_L_eval(x,        a,b,c,d,e);
    float Lhalf = Sheen_L_eval(0.5f,     a,b,c,d,e);
    float L1mx  = Sheen_L_eval(1.0f - x, a,b,c,d,e);
    return (x < 0.5f) ? expf(Lx) : expf(2.0f * Lhalf - L1mx);
}

inline float G_Charlie(float NdotV, float NdotL, float r)
{
    float lambdaV = Lambda_Charlie(NdotV, r);
    float lambdaL = Lambda_Charlie(NdotL, r);
    return 1.0f / (1.0f + lambdaV + lambdaL);
}

// Sheen BRDF with F=1
inline float EvaluateBRDF_SHEEN_scalar(const XMFLOAT3& V,
                                       const XMFLOAT3& L,
                                       const XMFLOAT3& N,
                                       float r)
{
    float NdotV = (std::fmax)(dot(N, V), 0.0f);
    float NdotL = (std::fmax)(dot(N, L), 0.0f);
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    XMFLOAT3 H = normalize(XMFLOAT3(V.x + L.x, V.y + L.y, V.z + L.z));
    float NdotH = (std::fmax)(dot(N, H), 0.0f);

    float D = D_Charlie(NdotH, r);
    float G = G_Charlie(NdotV, NdotL, r);
    float denom = (std::fmax)(4.0f * NdotV * NdotL, 1e-7f);
    return (D * G) / denom; // F=1 here
}

// Cosine hemisphere sample
inline XMFLOAT3 CosineHemisphereSample(float u1, float u2)
{
    float r = sqrtf(u1);
    float phi = 2.0f * PI * u2;
    float x = r * cosf(phi);
    float y = r * sinf(phi);
    float z = sqrtf((std::fmax)(0.0f, 1.0f - x*x - y*y));
    return XMFLOAT3(x,y,z);
}

inline float ComputeSheenDirectionalAlbedo(const XMFLOAT3& N,
                                           const XMFLOAT3& V,
                                           float sheenR,
                                           int numSamples)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    XMFLOAT3 T1, T2; CoordinateSystem(N, T1, T2);

    float sum_f = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        float u1 = dist(gen), u2 = dist(gen);
        XMFLOAT3 Llocal = CosineHemisphereSample(u1, u2);
        //local -> world
        XMFLOAT3 L = normalize(XMFLOAT3(
            Llocal.x * T1.x + Llocal.y * T2.x + Llocal.z * N.x,
            Llocal.x * T1.y + Llocal.y * T2.y + Llocal.z * N.y,
            Llocal.x * T1.z + Llocal.y * T2.z + Llocal.z * N.z
        ));
        sum_f += EvaluateBRDF_SHEEN_scalar(V, L, N, sheenR);
    }
    float mean_f = (numSamples > 0) ? (sum_f / numSamples) : 0.0f;
    return mean_f * PI;
}

void PrintFullLutMatrix(const std::vector<float>& lutSliceData, const std::wstring& title)
{
    std::wcout << L"\n======================================================================================\n";
    std::wcout << L" Full 32x32 Matrix for: " << title << L"\n";
    std::wcout << L"======================================================================================\n";
    std::wcout << L"(Rows are cos(theta) from 0.0 to 1.0, Columns are Roughness from 0.0 to 1.0)\n\n";

    // Print header row for Roughness values (printing every 4th value for brevity)
    std::wcout << L"cos\\r |";
    for (int x = 0; x < LUT_RESOLUTION; ++x) {
        float roughness = static_cast<float>(x) / (LUT_RESOLUTION - 1);
        std::wcout << L" " << std::fixed << std::setprecision(2) << roughness << L" ";
    }
    std::wcout << L"\n" << std::wstring(120, L'-') << L"\n";


    // Print each row of the matrix
    for (int y = 0; y < LUT_RESOLUTION; ++y) {
        float cosTheta = static_cast<float>(y) / (LUT_RESOLUTION - 1);

        // Print row header (cosTheta value)
        std::wcout << std::fixed << std::setprecision(3) << cosTheta << L" | ";

        // Print all values in the row
        for (int x = 0; x < LUT_RESOLUTION; ++x) {
            float value = lutSliceData[y * LUT_RESOLUTION + x];
            // Use scientific notation for small values, fixed for others
            if (value < 0.001f && value != 0.0f) {
                std::wcout << std::scientific << std::setprecision(1) << value << " ";
            } else {
                std::wcout << std::fixed << std::setprecision(3) << value << " ";
            }
        }
        std::wcout << L"\n";
        // Add a separator every 8 rows to make it easier to read
        if ((y + 1) % 8 == 0) {
            std::wcout << std::wstring(120, L'-') << L"\n";
        }
    }
    std::wcout << L"\n";
}

class ObjLoader {
private:
    // Compare a tg3_str to a C string literal.
    static bool tg3_str_eq(const tg3_str& s, const char* lit) {
        size_t n = strlen(lit);
        return s.len == (uint32_t)n && memcmp(s.data, lit, n) == 0;
    }

    // Convert a tg3_str to std::string.
    static std::string tg3_to_string(const tg3_str& s) {
        return (s.data && s.len > 0) ? std::string(s.data, s.len) : std::string();
    }

    // Look up a named extension in a tg3_extras_ext block.
    // Returns nullptr if not found.
    static const tg3_value* tg3_find_extension(const tg3_extras_ext& ext, const char* name) {
        for (uint32_t i = 0; i < ext.extensions_count; ++i) {
            if (tg3_str_eq(ext.extensions[i].name, name))
                return &ext.extensions[i].value;
        }
        return nullptr;
    }

    // Get a named double field from a tg3_value of type OBJECT.
    // Returns fallback if not found.
    static double tg3_obj_get_double(const tg3_value* obj, const char* key, double fallback) {
        if (!obj || obj->type != TG3_VALUE_OBJECT) return fallback;
        for (uint32_t i = 0; i < obj->object_count; ++i) {
            if (tg3_str_eq(obj->object_data[i].key, key)) {
                const tg3_value& v = obj->object_data[i].value;
                if (v.type == TG3_VALUE_REAL) return v.real_val;
                if (v.type == TG3_VALUE_INT)  return (double)v.int_val;
                return fallback;
            }
        }
        return fallback;
    }

    // Read a typed accessor from the v3 model into a flat vector.
    template <typename T>
    static std::vector<T> ReadGltfAccessorV3(const tg3_model& model, int accessorIdx)
    {
        if (accessorIdx < 0 || accessorIdx >= (int)model.accessors_count) return {};
        const tg3_accessor& acc = model.accessors[accessorIdx];
        if (acc.buffer_view < 0 || acc.buffer_view >= (int)model.buffer_views_count) return {};
        const tg3_buffer_view& bv = model.buffer_views[acc.buffer_view];
        if (bv.buffer < 0 || bv.buffer >= (int)model.buffers_count) return {};
        const tg3_buffer& buf = model.buffers[bv.buffer];

        uint32_t componentCount = 1;
        switch (acc.type) {
            case TG3_TYPE_SCALAR: componentCount = 1; break;
            case TG3_TYPE_VEC2:   componentCount = 2; break;
            case TG3_TYPE_VEC3:   componentCount = 3; break;
            case TG3_TYPE_VEC4:   componentCount = 4; break;
            default: break;
        }

        size_t compSize = 0;
        switch (acc.component_type) {
            case TG3_COMPONENT_TYPE_BYTE:
            case TG3_COMPONENT_TYPE_UNSIGNED_BYTE:  compSize = 1; break;
            case TG3_COMPONENT_TYPE_SHORT:
            case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: compSize = 2; break;
            case TG3_COMPONENT_TYPE_INT:
            case TG3_COMPONENT_TYPE_UNSIGNED_INT:
            case TG3_COMPONENT_TYPE_FLOAT:          compSize = 4; break;
            default: break;
        }

        size_t stride = bv.byte_stride;
        if (stride == 0) stride = compSize * componentCount;

        std::vector<T> result;
        result.reserve((size_t)acc.count * componentCount);
        const uint8_t* base = buf.data.data + bv.byte_offset + acc.byte_offset;

        for (uint64_t i = 0; i < acc.count; ++i) {
            const uint8_t* elem = base + i * stride;
            for (uint32_t c = 0; c < componentCount; ++c) {
                T value{};
                switch (acc.component_type) {
                    case TG3_COMPONENT_TYPE_FLOAT: {
                        float f; memcpy(&f, elem + c * sizeof(float), sizeof(float));
                        value = static_cast<T>(f); break;
                    }
                    case TG3_COMPONENT_TYPE_UNSIGNED_SHORT: {
                        uint16_t u; memcpy(&u, elem + c * sizeof(uint16_t), sizeof(uint16_t));
                        value = static_cast<T>(u); break;
                    }
                    case TG3_COMPONENT_TYPE_UNSIGNED_INT: {
                        uint32_t u; memcpy(&u, elem + c * sizeof(uint32_t), sizeof(uint32_t));
                        value = static_cast<T>(u); break;
                    }
                    case TG3_COMPONENT_TYPE_UNSIGNED_BYTE: {
                        value = static_cast<T>(elem[c]); break;
                    }
                    case TG3_COMPONENT_TYPE_SHORT: {
                        int16_t s; memcpy(&s, elem + c * sizeof(int16_t), sizeof(int16_t));
                        value = static_cast<T>(s); break;
                    }
                    case TG3_COMPONENT_TYPE_BYTE: {
                        value = static_cast<T>(reinterpret_cast<const int8_t*>(elem)[c]); break;
                    }
                    default: break;
                }
                result.push_back(value);
            }
        }
        return result;
    }

    // Recursively collect mesh instances with accumulated transforms (v3 structs).
    static void CollectGltfNodesV3(
        const tg3_model& model, int nodeIdx,
        const XMMATRIX& parentTransform,
        std::vector<std::pair<int, XMMATRIX>>& outMeshes)
    {
        if (nodeIdx < 0 || nodeIdx >= (int)model.nodes_count) return;
        const tg3_node& node = model.nodes[nodeIdx];
        XMMATRIX local = XMMatrixIdentity();

        if (node.has_matrix) {
            // tg3_node::matrix is double[16], column-major per glTF spec
            XMFLOAT4X4 m;
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    m.m[r][c] = (float)node.matrix[c * 4 + r];
            local = XMLoadFloat4x4(&m);
        } else {
            XMMATRIX T = XMMatrixTranslation((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]);
            XMMATRIX R = XMMatrixRotationQuaternion(XMVectorSet((float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2], (float)node.rotation[3]));
            XMMATRIX S = XMMatrixScaling((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]);
            local = S * R * T;
        }

        XMMATRIX world = local * parentTransform;
        if (node.mesh >= 0) outMeshes.push_back({ node.mesh, world });
        for (uint32_t i = 0; i < node.children_count; ++i)
            CollectGltfNodesV3(model, node.children[i], world, outMeshes);
    }

    // Find a named attribute accessor index in a tg3_primitive.
    static int tg3_find_attribute(const tg3_primitive& prim, const char* name) {
        for (uint32_t i = 0; i < prim.attributes_count; ++i) {
            if (tg3_str_eq(prim.attributes[i].key, name))
                return prim.attributes[i].value;
        }
        return -1;
    }

public:
    static void loadObjFile(
        const std::string& inputfile,
        std::vector<Vertex>* vertices,
        std::vector<UINT>* indices,
        std::vector<Material>* mats,
        std::vector<UINT>* materialIDs,
        UINT* materialOffset,
        UINT* materialVertexOffset,
        std::map<std::string, uint32_t>& textureMap,
        std::vector<TextureData>& albedoTextures,
        std::vector<TextureData>& normalTextures,
        std::vector<TextureData>& rmaTextures,
        const std::string& material_search_path = "./")
    {
        using namespace DirectX;

        std::cout << "[ObjLoader] Starting to load OBJ file: " << inputfile << std::endl;

        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = material_search_path;
        reader_config.triangulate = true;

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(inputfile, reader_config)) {
            if (!reader.Error().empty()) { std::cerr << "TinyObjReader: " << reader.Error(); exit(1); }
        }
        if (!reader.Warning().empty()) { std::cout << "TinyObjReader: " << reader.Warning(); }

        const auto& attrib = reader.GetAttrib();
        const auto& shapes = reader.GetShapes();
        auto& materials = reader.GetMaterials();

        std::cout << "[ObjLoader] Parsed '" << inputfile << "':" << std::endl;
        std::cout << "  - Shapes: " << shapes.size() << std::endl;
        std::cout << "  - Materials: " << materials.size() << std::endl;
        std::cout << "  - Vertices: " << (attrib.vertices.size() / 3) << std::endl;

        auto processTexture = [&](
            const std::string& filename, const std::string& materialPath,
            std::vector<TextureData>& textureList, bool isBumpMap,
            bool isSrgb // ADDED: Flag to determine the color space
            ) -> int
        {
            if (filename.empty()) return -1;

            // CHANGED: Include sRGB status in the cache key to prevent conflicts
            std::string srgb_suffix = isSrgb ? "_srgb" : "_linear";
            std::string fullPath = materialPath + filename;
            std::string cacheKey = fullPath + (isBumpMap ? "_bump_v2_uncompressed" : "_uncompressed") + srgb_suffix;

            if (textureMap.count(cacheKey)) {
                std::cout << "[ObjLoader] Texture '" << filename << "' found in cache. Reusing ID " << textureMap[cacheKey] << std::endl;
                return static_cast<int>(textureMap[cacheKey]);
            }

            std::cout << "[ObjLoader] Loading and processing new texture: " << fullPath << std::endl;

            int width, height, channels;
            unsigned char* data = stbi_load(fullPath.c_str(), &width, &height, &channels, 4);
            if (!data) {
                std::cerr << "  ERROR: Failed to load texture with stb_image: " << fullPath << std::endl;
                return -1;
            }

            // ADDED: Dynamically select the format based on the isSrgb flag
            DXGI_FORMAT format = isSrgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
            std::cout << "      - Using format: " << (isSrgb ? "DXGI_FORMAT_R8G8B8A8_UNORM_SRGB" : "DXGI_FORMAT_R8G8B8A8_UNORM") << std::endl;

            ScratchImage scratchImage;
            // CHANGED: Use the dynamically selected format
            HRESULT hr = scratchImage.Initialize2D(format, width, height, 1, 1);
            if (FAILED(hr)) { stbi_image_free(data); std::cerr << "  ERROR: Failed to initialize ScratchImage." << std::endl; return -1; }
            memcpy(scratchImage.GetPixels(), data, scratchImage.GetPixelsSize());
            stbi_image_free(data);

            TextureData texData;
            texData.original_width = width;
            texData.original_height = height;

            if (isBumpMap) {
                if (channels < 3) {
                    std::cout << "  - Detected grayscale bump map. Converting to normal map." << std::endl;
                    ScratchImage normalMapImage;
                    // CHANGED: Ensure the converted normal map uses a LINEAR format.
                    hr = ComputeNormalMap(*scratchImage.GetImage(0, 0, 0), CNMAP_DEFAULT, 1.0f, DXGI_FORMAT_R8G8B8A8_UNORM, normalMapImage);
                    if (FAILED(hr)) { std::cerr << "  ERROR: Failed to convert bump map." << std::endl; return -1; }
                    scratchImage = std::move(normalMapImage);
                } else {
                    std::cout << "  - WARNING: Texture was specified as a bump map but has " << channels << " channels. Treating it as a pre-existing normal map." << std::endl;
                }
            }

            // --- RESIZING LOGIC TO ENSURE ARRAY CONSISTENCY ---
            const TexMetadata& metadata = scratchImage.GetMetadata();
            if (metadata.width != TARGET_TEXTURE_DIM || metadata.height != TARGET_TEXTURE_DIM) {
                std::cout << "      - Note: Resizing texture from " << metadata.width << "x" << metadata.height
                          << " to " << TARGET_TEXTURE_DIM << "x" << TARGET_TEXTURE_DIM << " for array consistency." << std::endl;
                ScratchImage resizedImage;
                hr = Resize(*scratchImage.GetImage(0, 0, 0), TARGET_TEXTURE_DIM, TARGET_TEXTURE_DIM, TEX_FILTER_DEFAULT, resizedImage);
                if (FAILED(hr)) {
                    std::cerr << "      - ERROR: Failed to resize texture." << std::endl;
                    return -1;
                }
                scratchImage = std::move(resizedImage);
            }
            // --- END RESIZING LOGIC ---

            ScratchImage mipChain;
            std::cout << "  - Generating mipmaps..." << std::endl;
            hr = GenerateMipMaps(*scratchImage.GetImage(0, 0, 0), TEX_FILTER_DEFAULT, 0, mipChain);
            if (FAILED(hr)) { std::cerr << "  ERROR: Failed to generate mipmaps." << std::endl; return -1; }

            // Storing the final uncompressed, mipmapped image
            const TexMetadata& finalMetadata = mipChain.GetMetadata();
            texData.width = static_cast<int>(finalMetadata.width);
            texData.height = static_cast<int>(finalMetadata.height);
            texData.channels = 4;
            texData.image = std::move(mipChain);

            uint32_t textureID = static_cast<uint32_t>(textureList.size());
            textureList.push_back(std::move(texData));
            textureMap[cacheKey] = textureID;
            std::cout << "[ObjLoader] Finished processing '" << filename << "'. New ID: " << textureID << std::endl;
            return static_cast<int>(textureID);
        };

        auto processAndCombineRMA = [&](
            const std::string& r_fname, const std::string& m_fname,
            float constant_roughness, float constant_metallic,
            const std::string& materialPath, std::vector<TextureData>& rmaTextureList) -> int
        {
            if (r_fname.empty() && m_fname.empty()) return -1;
            std::string combinedKey = materialPath + r_fname + "+" + m_fname + "_uncompressed";

            if (textureMap.count(combinedKey)) {
                std::cout << "[ObjLoader] Combined RMA texture found in cache. Reusing ID " << textureMap[combinedKey] << std::endl;
                return static_cast<int>(textureMap[combinedKey]);
            }

            std::cout << "      - Combining Roughness ('" << (r_fname.empty() ? "None" : r_fname) << "') and Metallic ('" << (m_fname.empty() ? "None" : m_fname) << "')." << std::endl;

            auto load_single_channel = [&](const std::string& fname, ScratchImage& out_image) -> bool {
                if (fname.empty()) return false;
                int w, h, c;
                unsigned char* img_data = stbi_load((materialPath + fname).c_str(), &w, &h, &c, 1);
                if (!img_data) return false;
                out_image.Initialize2D(DXGI_FORMAT_R8_UNORM, w, h, 1, 1);
                memcpy(out_image.GetPixels(), img_data, out_image.GetPixelsSize());
                stbi_image_free(img_data);
                return true;
            };

            ScratchImage r_img, m_img;
            bool has_r = load_single_channel(r_fname, r_img);
            bool has_m = load_single_channel(m_fname, m_img);

            size_t width = has_r ? r_img.GetMetadata().width : (has_m ? m_img.GetMetadata().width : 0);
            size_t height = has_r ? r_img.GetMetadata().height : (has_m ? m_img.GetMetadata().height : 0);
            if (width == 0) return -1;

            if (has_r && has_m && (r_img.GetMetadata().width != m_img.GetMetadata().width || r_img.GetMetadata().height != m_img.GetMetadata().height)) {
                std::cout << "      - Note: Roughness and Metallic maps have different dimensions. Resizing to match." << std::endl;
                ScratchImage& smaller = (r_img.GetPixelsSize() < m_img.GetPixelsSize()) ? r_img : m_img;
                ScratchImage& larger = (r_img.GetPixelsSize() < m_img.GetPixelsSize()) ? m_img : r_img;
                ScratchImage resized;
                Resize(*smaller.GetImage(0, 0, 0), larger.GetMetadata().width, larger.GetMetadata().height, TEX_FILTER_DEFAULT, resized);
                smaller = std::move(resized);
                width = larger.GetMetadata().width;
                height = larger.GetMetadata().height;
            }

            ScratchImage combinedImage;
            // NOTE: Combined data textures like RMA are always linear.
            combinedImage.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, 1);
            uint8_t* dest_pixels = combinedImage.GetPixels();
            const uint8_t* r_pixels = has_r ? r_img.GetPixels() : nullptr;
            const uint8_t* m_pixels = has_m ? m_img.GetPixels() : nullptr;
            uint8_t r_const = static_cast<uint8_t>(constant_roughness * 255.0f);
            uint8_t m_const = static_cast<uint8_t>(constant_metallic * 255.0f);

            for (size_t i = 0; i < width * height; ++i) {
                dest_pixels[i * 4 + 0] = 255; // R: Ambient Occlusion (default 1.0)
                dest_pixels[i * 4 + 1] = r_pixels ? r_pixels[i] : r_const; // G: Roughness
                dest_pixels[i * 4 + 2] = m_pixels ? m_pixels[i] : m_const; // B: Metallic
                dest_pixels[i * 4 + 3] = 255; // A: Unused
            }

            TextureData texData;
            texData.original_width = width;
            texData.original_height = height;

            const TexMetadata& metadata = combinedImage.GetMetadata();
            if (metadata.width != TARGET_TEXTURE_DIM || metadata.height != TARGET_TEXTURE_DIM) {
                std::cout << "      - Note: Resizing combined RMA texture from " << metadata.width << "x" << metadata.height
                          << " to " << TARGET_TEXTURE_DIM << "x" << TARGET_TEXTURE_DIM << " for array consistency." << std::endl;
                ScratchImage resizedImage;
                HRESULT hr = Resize(*combinedImage.GetImage(0, 0, 0), TARGET_TEXTURE_DIM, TARGET_TEXTURE_DIM, TEX_FILTER_DEFAULT, resizedImage);
                if (FAILED(hr)) {
                    std::cerr << "      - ERROR: Failed to resize combined RMA texture." << std::endl;
                    return -1;
                }
                combinedImage = std::move(resizedImage);
            }

            ScratchImage mipChain;
            GenerateMipMaps(*combinedImage.GetImage(0, 0, 0), TEX_FILTER_DEFAULT, 0, mipChain);

            const TexMetadata& finalMetadata = mipChain.GetMetadata();
            texData.width = static_cast<int>(finalMetadata.width);
            texData.height = static_cast<int>(finalMetadata.height);
            texData.channels = 4;
            texData.image = std::move(mipChain);

            uint32_t textureID = static_cast<uint32_t>(rmaTextureList.size());
            rmaTextureList.push_back(std::move(texData));
            textureMap[combinedKey] = textureID;
            std::cout << "[ObjLoader] Finished combining RMA. New ID: " << textureID << std::endl;
            return static_cast<int>(textureID);
        };

        auto processAlbedoWithOpacity = [&](
            const std::string& diffuse_fname, const std::string& opacity_fname,
            float constant_dissolve, const std::string& materialPath,
            std::vector<TextureData>& albedoList) -> int
        {
            if (diffuse_fname.empty() && opacity_fname.empty()) return -1;

            std::string cacheKey = materialPath + diffuse_fname + "+opacity_" + opacity_fname
                                 + "_d" + std::to_string(constant_dissolve) + "_srgb";

            if (textureMap.count(cacheKey)) {
                std::cout << "[ObjLoader] Albedo+Opacity found in cache. Reusing ID " << textureMap[cacheKey] << std::endl;
                return static_cast<int>(textureMap[cacheKey]);
            }

            // Load diffuse as RGBA
            int dw = 1, dh = 1;
            ScratchImage diffuseImage;
            if (!diffuse_fname.empty()) {
                int dc;
                unsigned char* ddata = stbi_load((materialPath + diffuse_fname).c_str(), &dw, &dh, &dc, 4);
                if (!ddata) {
                    std::cerr << "  ERROR: Failed to load diffuse texture: " << diffuse_fname << std::endl;
                    return -1;
                }
                diffuseImage.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, dw, dh, 1, 1);
                memcpy(diffuseImage.GetPixels(), ddata, diffuseImage.GetPixelsSize());
                stbi_image_free(ddata);
            } else {
                // No diffuse texture — create a white 1x1 placeholder
                diffuseImage.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 1, 1, 1, 1);
                uint8_t* px = diffuseImage.GetPixels();
                px[0] = px[1] = px[2] = px[3] = 255;
            }

            // Load opacity map (single channel) and write into alpha
            if (!opacity_fname.empty()) {
                int ow, oh, oc;
                unsigned char* odata = stbi_load((materialPath + opacity_fname).c_str(), &ow, &oh, &oc, 1);
                if (!odata) {
                    std::cerr << "  ERROR: Failed to load opacity texture: " << opacity_fname << std::endl;
                    return -1;
                }

                // If dimensions mismatch, resize opacity to match diffuse
                ScratchImage opacityImage;
                opacityImage.Initialize2D(DXGI_FORMAT_R8_UNORM, ow, oh, 1, 1);
                memcpy(opacityImage.GetPixels(), odata, opacityImage.GetPixelsSize());
                stbi_image_free(odata);

                if (ow != dw || oh != dh) {
                    std::cout << "      - Resizing opacity map from " << ow << "x" << oh
                              << " to " << dw << "x" << dh << " to match diffuse." << std::endl;
                    ScratchImage resized;
                    Resize(*opacityImage.GetImage(0, 0, 0), dw, dh, TEX_FILTER_DEFAULT, resized);
                    opacityImage = std::move(resized);
                }

                uint8_t* diffuse_px = diffuseImage.GetPixels();
                const uint8_t* opacity_px = opacityImage.GetPixels();
                for (size_t i = 0; i < (size_t)dw * dh; ++i) {
                    diffuse_px[i * 4 + 3] = opacity_px[i]; // Bake opacity into alpha
                }
                std::cout << "      - Baked opacity map into albedo alpha channel." << std::endl;
            } else {
                // No opacity texture but dissolve < 1.0 — apply constant alpha
                uint8_t alpha_const = static_cast<uint8_t>(constant_dissolve * 255.0f);
                uint8_t* diffuse_px = diffuseImage.GetPixels();
                for (size_t i = 0; i < (size_t)dw * dh; ++i) {
                    diffuse_px[i * 4 + 3] = alpha_const;
                }
                std::cout << "      - Applied constant dissolve (" << constant_dissolve << ") to albedo alpha." << std::endl;
            }

            // From here, same pipeline as processTexture: resize → mip → store
            TextureData texData;
            texData.original_width = dw;
            texData.original_height = dh;

            const TexMetadata& metadata = diffuseImage.GetMetadata();
            if (metadata.width != TARGET_TEXTURE_DIM || metadata.height != TARGET_TEXTURE_DIM) {
                ScratchImage resizedImage;
                HRESULT hr = Resize(*diffuseImage.GetImage(0, 0, 0), TARGET_TEXTURE_DIM, TARGET_TEXTURE_DIM, TEX_FILTER_DEFAULT, resizedImage);
                if (FAILED(hr)) { std::cerr << "  ERROR: Failed to resize albedo+opacity." << std::endl; return -1; }
                diffuseImage = std::move(resizedImage);
            }

            ScratchImage mipChain;
            HRESULT hr = GenerateMipMaps(*diffuseImage.GetImage(0, 0, 0), TEX_FILTER_DEFAULT, 0, mipChain);
            if (FAILED(hr)) { std::cerr << "  ERROR: Failed to generate mipmaps for albedo+opacity." << std::endl; return -1; }

            const TexMetadata& finalMetadata = mipChain.GetMetadata();
            texData.width = static_cast<int>(finalMetadata.width);
            texData.height = static_cast<int>(finalMetadata.height);
            texData.channels = 4;
            texData.image = std::move(mipChain);

            uint32_t textureID = static_cast<uint32_t>(albedoList.size());
            albedoList.push_back(std::move(texData));
            textureMap[cacheKey] = textureID;
            std::cout << "[ObjLoader] Finished albedo+opacity. New ID: " << textureID << std::endl;
            return static_cast<int>(textureID);
        };

        Material defaultMaterial;
        mats->push_back(defaultMaterial);
        (*materialOffset)++;

        std::cout << "[ObjLoader] Processing materials..." << std::endl;
        for (const auto& mat : materials) {
            std::cout << "  [Material] Processing '" << mat.name << "'" << std::endl;
            Material t_mat;
            t_mat.Kd = { mat.diffuse[0], mat.diffuse[1], mat.diffuse[2], mat.dissolve };
            t_mat.Ke = { mat.emission[0], mat.emission[1], mat.emission[2] };
            t_mat.Ni = mat.ior;
            t_mat.Pr_Pm_Ps_Pc = { mat.roughness, mat.metallic, mat.sheen, mat.clearcoat_thickness };
            t_mat.Pcr_aniso_anisor = { mat.clearcoat_roughness, mat.anisotropy, mat.anisotropy_rotation };
            t_mat.Tf = {mat.transmittance[0],mat.transmittance[1], mat.transmittance[2]};

            std::cout << "    - Albedo map: " << (mat.diffuse_texname.empty() ? "None" : mat.diffuse_texname) << std::endl;
            // CHANGED: Pass `true` for isSrgb for albedo textures
            bool has_opacity_tex = !mat.alpha_texname.empty();
            bool has_partial_dissolve = mat.dissolve < 1.0f;

            if (has_opacity_tex || has_partial_dissolve) {
                std::cout << "    - Opacity map: " << (mat.alpha_texname.empty() ? "None (constant d=" + std::to_string(mat.dissolve) + ")" : mat.alpha_texname) << std::endl;
                t_mat.albedoTexID = processAlbedoWithOpacity(
                    mat.diffuse_texname, mat.alpha_texname, mat.dissolve,
                    material_search_path, albedoTextures);

                // Custom threshold from .mtl, otherwise default 0.5
                if (mat.unknown_parameter.count("alpha_cutoff")) {
                    t_mat.alphaThreshold = std::stof(mat.unknown_parameter.at("alpha_cutoff"));
                } else {
                    t_mat.alphaThreshold = 0.5f;
                }
            } else {
                t_mat.albedoTexID = processTexture(mat.diffuse_texname, material_search_path, albedoTextures, false, true);
                t_mat.alphaThreshold = 1.0f; // opaque — no alpha test
            }

            bool isBump = !mat.bump_texname.empty() && mat.normal_texname.empty();
            std::string normalTexName = !mat.normal_texname.empty() ? mat.normal_texname : mat.bump_texname;
            std::cout << "    - Normal/Bump map: " << (normalTexName.empty() ? "None" : normalTexName) << std::endl;
            // CHANGED: Pass `false` for isSrgb for normal maps (they are linear data)
            t_mat.normalTexID = processTexture(normalTexName, material_search_path, normalTextures, isBump, false);

            if (mat.unknown_parameter.count("map_rma")) {
                std::cout << "    - Found custom RMA map: " << mat.unknown_parameter.at("map_rma") << std::endl;
                // CHANGED: Pass `false` for isSrgb for RMA maps (they are linear data)
                t_mat.rmaTexID = processTexture(mat.unknown_parameter.at("map_rma"), material_search_path, rmaTextures, false, false);
            } else if (!mat.roughness_texname.empty() || !mat.metallic_texname.empty()) {
                t_mat.rmaTexID = processAndCombineRMA(mat.roughness_texname, mat.metallic_texname, mat.roughness, mat.metallic, material_search_path, rmaTextures);
            } else {
                std::cout << "    - No RMA or PBR maps found for this material." << std::endl;
                t_mat.rmaTexID = -1;
            }
            mats->push_back(t_mat);
        }

        std::cout << "[ObjLoader] Processing vertices and indices..." << std::endl;
        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        for (const auto& shape : shapes) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv != 3) { index_offset += fv; continue; }

                int materialID = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;
                uint32_t matID = (materialID >= 0) ? uint32_t(materialID + *materialOffset) : 0u;
                materialIDs->push_back(matID);

                tinyobj::index_t idx[3] = { shape.mesh.indices[index_offset], shape.mesh.indices[index_offset + 1], shape.mesh.indices[index_offset + 2] };
                XMFLOAT3 p[3], n[3]; XMFLOAT2 uv[3];
                for (int i = 0; i < 3; ++i) {
                    p[i] = { attrib.vertices[3 * idx[i].vertex_index + 0], attrib.vertices[3 * idx[i].vertex_index + 1], attrib.vertices[3 * idx[i].vertex_index + 2] };
                    uv[i] = (idx[i].texcoord_index >= 0) ? XMFLOAT2{ attrib.texcoords[2 * idx[i].texcoord_index + 0], 1.0f - attrib.texcoords[2 * idx[i].texcoord_index + 1] } : XMFLOAT2{0,0};
                    n[i] = (idx[i].normal_index >= 0) ? XMFLOAT3{ attrib.normals[3 * idx[i].normal_index + 0], attrib.normals[3 * idx[i].normal_index + 1], attrib.normals[3 * idx[i].normal_index + 2] } : XMFLOAT3{0,0,0};
                }

                if (idx[0].normal_index < 0) {
                    XMVECTOR p0_v = XMLoadFloat3(&p[0]), p1_v = XMLoadFloat3(&p[1]), p2_v = XMLoadFloat3(&p[2]);
                    XMVECTOR faceNormalVec = XMVector3Normalize(XMVector3Cross(p1_v - p0_v, p2_v - p0_v));
                    XMStoreFloat3(&n[0], faceNormalVec); n[1] = n[2] = n[0];
                }

                for (int i = 0; i < 3; ++i) {
                    Vertex v({p[i]}, {n[i].x, n[i].y, n[i].z, static_cast<float>(matID)}, {uv[i]});
                    if (uniqueVertices.count(v)) {
                        indices->push_back(uniqueVertices[v]);
                    } else {
                        uint32_t newIndex = static_cast<uint32_t>(vertices->size());
                        uniqueVertices[v] = newIndex;
                        indices->push_back(newIndex);
                        vertices->push_back(v);
                    }
                }
                index_offset += fv;
            }
        }

        std::cout << "[ObjLoader] Finished processing geometry for '" << inputfile << "'." << std::endl;
        std::cout << "  - Unique vertices created: " << uniqueVertices.size() << std::endl;
        std::cout << "  - Total indices created: " << indices->size() << std::endl;

        *materialOffset += static_cast<UINT>(materials.size());

        std::cout << "[ObjLoader] COMPLETED loading '" << inputfile << "'." << std::endl;
        std::cout << "  - Total vertices in buffer: " << vertices->size() << std::endl;
        std::cout << "  - Total indices in buffer: " << indices->size() << std::endl;
        std::cout << "  - Total materials in buffer: " << mats->size() << std::endl;
        std::cout << "  - Total albedo textures loaded: " << albedoTextures.size() << std::endl;
        std::cout << "  - Total normal textures loaded: " << normalTextures.size() << std::endl;
        std::cout << "  - Total RMA textures loaded: " << rmaTextures.size() << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;
    }

    static void loadGlbFile(
        const std::string& inputfile,
        std::vector<Vertex>* vertices,
        std::vector<UINT>* indices,
        std::vector<Material>* mats,
        std::vector<UINT>* materialIDs,
        UINT* materialOffset,
        UINT* materialVertexOffset,
        std::map<std::string, uint32_t>& textureMap,
        std::vector<TextureData>& albedoTextures,
        std::vector<TextureData>& normalTextures,
        std::vector<TextureData>& rmaTextures,
        const std::string& material_search_path = "./")
    {
        using namespace DirectX;

        std::cout << "[GlbLoader] Starting to load GLB file: " << inputfile << std::endl;

        // ---- 1. Read file into memory -------------------------------------------
        std::ifstream file(inputfile, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "[GlbLoader] ERROR: Could not open file: " << inputfile << std::endl;
            return;
        }
        size_t fileSize = (size_t)file.tellg();
        file.seekg(0);
        std::vector<uint8_t> fileData(fileSize);
        file.read(reinterpret_cast<char*>(fileData.data()), fileSize);
        file.close();

        // ---- 2. Parse with v3 API -----------------------------------------------
        tg3_model model{};
        tg3_error_stack errors{};
        tg3_error_stack_init(&errors);

        tg3_parse_options opts{};
        tg3_parse_options_init(&opts);
        // Let v3 decode images via its built-in stbi (TINYGLTF3_ENABLE_STB_IMAGE)
        opts.preserve_image_channels = 0; // widen to RGBA
        opts.images_as_is = 0;

        opts.image.load_image = [](tg3_image_result* result,
                           const tg3_image_request* request,
                           void* /*user_data*/) -> int32_t
        {
            std::cout << "    [ImageCallback] Decoding image " << request->image_index
              << " (" << request->data_size << " bytes)" << std::endl;
            int w, h, c;
            unsigned char* data = stbi_load_from_memory(
                request->data, (int)request->data_size,
                &w, &h, &c, 4);  // force RGBA
            if (!data) return 0;  // failure

            result->pixels     = data;
            result->width      = w;
            result->height     = h;
            result->component  = 4;
            result->bits       = 8;
            result->pixel_type = TG3_COMPONENT_TYPE_UNSIGNED_BYTE;
            return 1;  // success
        };

        opts.image.free_image = [](uint8_t* pixels, void* /*user_data*/) {
            stbi_image_free(pixels);
        };

        opts.image.user_data = nullptr;

        tg3_error_code ec = tg3_parse_glb(
            &model, &errors,
            fileData.data(), (uint64_t)fileSize,
            nullptr, 0,  // no base_dir needed for GLB
            &opts);

        struct DecodedImage {
            std::vector<uint8_t> pixels;
            int width, height, channels;
        };
        std::vector<DecodedImage> decodedImages(model.images_count);

        for (uint32_t i = 0; i < model.images_count; ++i) {
            const tg3_image& img = model.images[i];

            // Try already-decoded pixels first (in case callback worked)
            if (img.image.data && img.image.count > 0 && img.width > 0) {
                decodedImages[i].pixels.assign(img.image.data, img.image.data + img.image.count);
                decodedImages[i].width    = img.width;
                decodedImages[i].height   = img.height;
                decodedImages[i].channels = img.component;
                continue;
            }

            // Otherwise, extract raw bytes from the buffer view and decode
            if (img.buffer_view >= 0 && img.buffer_view < (int)model.buffer_views_count) {
                const tg3_buffer_view& bv = model.buffer_views[img.buffer_view];
                if (bv.buffer >= 0 && bv.buffer < (int)model.buffers_count) {
                    const tg3_buffer& buf = model.buffers[bv.buffer];
                    const uint8_t* rawData = buf.data.data + bv.byte_offset;
                    uint64_t rawSize = bv.byte_length;

                    int w, h, c;
                    unsigned char* decoded = stbi_load_from_memory(rawData, (int)rawSize, &w, &h, &c, 4);
                    if (decoded) {
                        decodedImages[i].pixels.assign(decoded, decoded + w * h * 4);
                        decodedImages[i].width    = w;
                        decodedImages[i].height   = h;
                        decodedImages[i].channels = 4;
                        stbi_image_free(decoded);
                        std::cout << "    [Image " << i << "] Decoded " << w << "x" << h << " from buffer view" << std::endl;
                    } else {
                        std::cerr << "    [Image " << i << "] stbi decode failed" << std::endl;
                    }
                }
            }
        }

        if (ec != TG3_OK) {
            std::cerr << "[GlbLoader] Failed to parse GLB (error code " << (int)ec << ")." << std::endl;
            for (uint32_t i = 0; i < tg3_errors_count(&errors); ++i) {
                const tg3_error_entry* e = tg3_errors_get(&errors, i);
                if (e) std::cerr << "  " << (e->message ? e->message : "?") << std::endl;
            }
            tg3_error_stack_free(&errors);
            return;
        }

        // Print any warnings
        for (uint32_t i = 0; i < tg3_errors_count(&errors); ++i) {
            const tg3_error_entry* e = tg3_errors_get(&errors, i);
            if (e && e->severity == TG3_SEVERITY_WARNING)
                std::cout << "[GlbLoader] Warning: " << (e->message ? e->message : "?") << std::endl;
        }

        std::cout << "[GlbLoader] Parsed '" << inputfile << "':" << std::endl;
        std::cout << "  - Meshes: "    << model.meshes_count    << std::endl;
        std::cout << "  - Materials: " << model.materials_count << std::endl;
        std::cout << "  - Images: "    << model.images_count    << std::endl;

        // ---- 3. Texture helpers -------------------------------------------------
        auto getImageIndex = [&](int textureIndex) -> int {
            if (textureIndex < 0 || textureIndex >= (int)model.textures_count) return -1;
            return model.textures[textureIndex].source;
        };

        // Process a tg3_image through the same pipeline as ObjLoader's processTexture
        auto processEmbeddedTexture = [&](
            int imageIndex, std::vector<TextureData>& textureList,
            bool isSrgb, const std::string& label) -> int
        {
            if (imageIndex < 0 || imageIndex >= (int)decodedImages.size()) return -1;

            std::string cacheKey = inputfile + "_img" + std::to_string(imageIndex)
                                 + (isSrgb ? "_srgb" : "_linear");
            if (textureMap.count(cacheKey)) {
                std::cout << "    " << label << " (img " << imageIndex << ") cached as ID " << textureMap[cacheKey] << std::endl;
                return (int)textureMap[cacheKey];
            }

            const DecodedImage& img = decodedImages[imageIndex];
            if (img.pixels.empty()) {
                std::cerr << "    ERROR: image " << imageIndex << " has no pixel data." << std::endl;
                return -1;
            }

            int width  = img.width;
            int height = img.height;
            const uint8_t* pixelData = img.pixels.data();

            std::cout << "    " << label << " (img " << imageIndex << ", "
                      << width << "x" << height << ", " << img.channels << "ch)" << std::endl;

            DXGI_FORMAT format = isSrgb ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
            ScratchImage scratch;
            HRESULT hr = scratch.Initialize2D(format, width, height, 1, 1);
            if (FAILED(hr)) return -1;
            memcpy(scratch.GetPixels(), pixelData, (std::min)(scratch.GetPixelsSize(), (size_t)width * height * 4));

            const TexMetadata& meta = scratch.GetMetadata();
            if (meta.width != TARGET_TEXTURE_DIM || meta.height != TARGET_TEXTURE_DIM) {
                ScratchImage resized;
                hr = Resize(*scratch.GetImage(0,0,0), TARGET_TEXTURE_DIM, TARGET_TEXTURE_DIM, TEX_FILTER_DEFAULT, resized);
                if (FAILED(hr)) return -1;
                scratch = std::move(resized);
            }

            ScratchImage mipChain;
            hr = GenerateMipMaps(*scratch.GetImage(0,0,0), TEX_FILTER_DEFAULT, 0, mipChain);
            if (FAILED(hr)) return -1;

            TextureData texData;
            texData.original_width = width; texData.original_height = height;
            const TexMetadata& fm = mipChain.GetMetadata();
            texData.width = (int)fm.width; texData.height = (int)fm.height;
            texData.channels = 4; texData.image = std::move(mipChain);

            uint32_t id = (uint32_t)textureList.size();
            textureList.push_back(std::move(texData));
            textureMap[cacheKey] = id;
            std::cout << "    " << label << " -> ID " << id << std::endl;
            return (int)id;
        };

        // Build combined RMA from glTF metallicRoughness (G=rough, B=metal) + occlusion
        auto processGltfRMA = [&](
            int mrImgIdx, int aoImgIdx,
            float constR, float constM,
            std::vector<TextureData>& rmaList) -> int
        {
            if (mrImgIdx < 0 && aoImgIdx < 0) return -1;

            std::string cacheKey = inputfile + "_rma_mr" + std::to_string(mrImgIdx)
                                 + "_ao" + std::to_string(aoImgIdx);
            if (textureMap.count(cacheKey)) return (int)textureMap[cacheKey];

            const uint8_t* mrPx = (mrImgIdx >= 0 && !decodedImages[mrImgIdx].pixels.empty())
                ? decodedImages[mrImgIdx].pixels.data() : nullptr;
            const uint8_t* aoPx = (aoImgIdx >= 0 && !decodedImages[aoImgIdx].pixels.empty())
                ? decodedImages[aoImgIdx].pixels.data() : nullptr;

            int w = 0, h = 0;
            if (mrPx) { w = decodedImages[mrImgIdx].width; h = decodedImages[mrImgIdx].height; }
            else if (aoPx) { w = decodedImages[aoImgIdx].width; h = decodedImages[aoImgIdx].height; }
            if (w == 0) return -1;

            ScratchImage combined;
            combined.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM, w, h, 1, 1);
            uint8_t* dest = combined.GetPixels();
            uint8_t rC = (uint8_t)(constR * 255.0f), mC = (uint8_t)(constM * 255.0f);

            for (size_t i = 0; i < (size_t)w * h; ++i) {
                dest[i*4+0] = aoPx ? aoPx[i*4+0] : 255;     // R = AO
                dest[i*4+1] = mrPx ? mrPx[i*4+1] : rC;      // G = Roughness
                dest[i*4+2] = mrPx ? mrPx[i*4+2] : mC;      // B = Metallic
                dest[i*4+3] = 255;
            }

            const TexMetadata& meta = combined.GetMetadata();
            if (meta.width != TARGET_TEXTURE_DIM || meta.height != TARGET_TEXTURE_DIM) {
                ScratchImage resized;
                if (FAILED(Resize(*combined.GetImage(0,0,0), TARGET_TEXTURE_DIM, TARGET_TEXTURE_DIM, TEX_FILTER_DEFAULT, resized))) return -1;
                combined = std::move(resized);
            }

            ScratchImage mipChain;
            GenerateMipMaps(*combined.GetImage(0,0,0), TEX_FILTER_DEFAULT, 0, mipChain);

            TextureData texData;
            texData.original_width = w; texData.original_height = h;
            const TexMetadata& fm = mipChain.GetMetadata();
            texData.width = (int)fm.width; texData.height = (int)fm.height;
            texData.channels = 4; texData.image = std::move(mipChain);

            uint32_t id = (uint32_t)rmaList.size();
            rmaList.push_back(std::move(texData));
            textureMap[cacheKey] = id;
            return (int)id;
        };

        // ---- 4. Materials -------------------------------------------------------
        Material defaultMaterial;
        mats->push_back(defaultMaterial);
        (*materialOffset)++;

        for (uint32_t mi = 0; mi < model.materials_count; ++mi) {
            const tg3_material& gmat = model.materials[mi];
            const tg3_pbr_metallic_roughness& pbr = gmat.pbr_metallic_roughness;
            std::cout << "  [Material] '" << tg3_to_string(gmat.name) << "'" << std::endl;

            Material t_mat{};
            t_mat.Kd = { (float)pbr.base_color_factor[0], (float)pbr.base_color_factor[1],
                         (float)pbr.base_color_factor[2], (float)pbr.base_color_factor[3] };
            t_mat.Ke = { (float)gmat.emissive_factor[0], (float)gmat.emissive_factor[1],
                         (float)gmat.emissive_factor[2] };

            // KHR_materials_emissive_strength — multiplier for emissive values > 1.0
            const tg3_value* emStrExt = tg3_find_extension(gmat.ext, "KHR_materials_emissive_strength");
            if (emStrExt) {
                float strength = (float)tg3_obj_get_double(emStrExt, "emissiveStrength", 1.0);
                t_mat.Ke.x *= strength;
                t_mat.Ke.y *= strength;
                t_mat.Ke.z *= strength;
            }
            t_mat.Ni = 1.5f;

            // KHR_materials_ior
            const tg3_value* iorExt = tg3_find_extension(gmat.ext, "KHR_materials_ior");
            if (iorExt) t_mat.Ni = (float)tg3_obj_get_double(iorExt, "ior", 1.5);

            float roughness = (float)pbr.roughness_factor;
            float metallic  = (float)pbr.metallic_factor;
            t_mat.Pr_Pm_Ps_Pc = { roughness, metallic, 0.0f, 0.0f };
            t_mat.Pcr_aniso_anisor = { 0.0f, 0.0f, 0.0f };
            t_mat.Tf = { 0.0f, 0.0f, 0.0f };

            // KHR_materials_sheen
            const tg3_value* sheenExt = tg3_find_extension(gmat.ext, "KHR_materials_sheen");
            if (sheenExt)
                t_mat.Pr_Pm_Ps_Pc.z = (float)tg3_obj_get_double(sheenExt, "sheenRoughnessFactor", 0.0);

            // KHR_materials_clearcoat
            const tg3_value* ccExt = tg3_find_extension(gmat.ext, "KHR_materials_clearcoat");
            if (ccExt) {
                t_mat.Pr_Pm_Ps_Pc.w    = (float)tg3_obj_get_double(ccExt, "clearcoatFactor", 0.0);
                t_mat.Pcr_aniso_anisor.x = (float)tg3_obj_get_double(ccExt, "clearcoatRoughnessFactor", 0.0);
            }

            // KHR_materials_anisotropy
            const tg3_value* anisoExt = tg3_find_extension(gmat.ext, "KHR_materials_anisotropy");
            if (anisoExt) {
                t_mat.Pcr_aniso_anisor.y = (float)tg3_obj_get_double(anisoExt, "anisotropyStrength", 0.0);
                t_mat.Pcr_aniso_anisor.z = (float)tg3_obj_get_double(anisoExt, "anisotropyRotation", 0.0);
            }

            // KHR_materials_transmission
            const tg3_value* transExt = tg3_find_extension(gmat.ext, "KHR_materials_transmission");
            if (transExt) {
                float tf = (float)tg3_obj_get_double(transExt, "transmissionFactor", 0.0);
                t_mat.Tf = { tf, tf, tf };
            }

            // Albedo (sRGB)
            int baseColorImg = getImageIndex(pbr.base_color_texture.index);
            bool hasMask = tg3_str_eq(gmat.alpha_mode, "MASK");
            if (tg3_str_eq(gmat.alpha_mode, "BLEND") || hasMask) {
                t_mat.albedoTexID    = processEmbeddedTexture(baseColorImg, albedoTextures, true, "albedo+alpha");
                t_mat.alphaThreshold = hasMask ? (float)gmat.alpha_cutoff : 0.5f;
            } else {
                t_mat.albedoTexID    = processEmbeddedTexture(baseColorImg, albedoTextures, true, "albedo");
                t_mat.alphaThreshold = 1.0f;
            }

            // Normal (linear)
            t_mat.normalTexID = processEmbeddedTexture(
                getImageIndex(gmat.normal_texture.index), normalTextures, false, "normal");

            // RMA (linear)
            int mrImg = getImageIndex(pbr.metallic_roughness_texture.index);
            int aoImg = getImageIndex(gmat.occlusion_texture.index);
            t_mat.rmaTexID = (mrImg >= 0 || aoImg >= 0)
                ? processGltfRMA(mrImg, aoImg, roughness, metallic, rmaTextures) : -1;

            mats->push_back(t_mat);
        }

        // ---- 5. Collect mesh instances via scene graph --------------------------
        std::vector<std::pair<int, XMMATRIX>> meshInstances;
        int sceneIdx = model.default_scene >= 0 ? model.default_scene : 0;
        if (sceneIdx < (int)model.scenes_count) {
            const tg3_scene& scene = model.scenes[sceneIdx];
            for (uint32_t i = 0; i < scene.nodes_count; ++i)
                CollectGltfNodesV3(model, scene.nodes[i], XMMatrixIdentity(), meshInstances);
        }

        // ---- 6. Geometry --------------------------------------------------------
        std::cout << "[GlbLoader] Processing geometry (" << meshInstances.size() << " mesh instances)..." << std::endl;
        std::unordered_map<Vertex, uint32_t> uniqueVertices;

        for (const auto& [meshIdx, worldTransform] : meshInstances) {
            if (meshIdx < 0 || meshIdx >= (int)model.meshes_count) continue;
            const tg3_mesh& mesh = model.meshes[meshIdx];
            XMMATRIX normalMatrix = XMMatrixTranspose(XMMatrixInverse(nullptr, worldTransform));

            for (uint32_t pi = 0; pi < mesh.primitives_count; ++pi) {
                const tg3_primitive& prim = mesh.primitives[pi];

                // Only triangle lists (mode 4 or default -1)
                if (prim.mode != TG3_MODE_TRIANGLES && prim.mode != -1) continue;

                // Attributes — look up by name in the key-value pairs
                int posAccessor  = tg3_find_attribute(prim, "POSITION");
                int normAccessor = tg3_find_attribute(prim, "NORMAL");
                int uvAccessor   = tg3_find_attribute(prim, "TEXCOORD_0");

                if (posAccessor < 0) continue;
                std::vector<float> positions = ReadGltfAccessorV3<float>(model, posAccessor);
                std::vector<float> normals;
                if (normAccessor >= 0) normals = ReadGltfAccessorV3<float>(model, normAccessor);
                std::vector<float> texcoords;
                if (uvAccessor >= 0) texcoords = ReadGltfAccessorV3<float>(model, uvAccessor);

                size_t vertexCount = positions.size() / 3;

                // Indices
                std::vector<uint32_t> primIndices;
                if (prim.indices >= 0) {
                    primIndices = ReadGltfAccessorV3<uint32_t>(model, prim.indices);
                } else {
                    primIndices.resize(vertexCount);
                    for (uint32_t i = 0; i < (uint32_t)vertexCount; ++i) primIndices[i] = i;
                }

                // Material
                uint32_t matID = (prim.material >= 0)
                    ? (uint32_t)(prim.material + *materialOffset) : 0u;

                // Triangles
                for (size_t t = 0; t + 2 < primIndices.size(); t += 3) {
                    materialIDs->push_back(matID);

                    uint32_t triIdx[3] = { primIndices[t], primIndices[t+1], primIndices[t+2] };
                    XMFLOAT3 p[3], n[3]; XMFLOAT2 uv[3];

                    for (int i = 0; i < 3; ++i) {
                        uint32_t vi = triIdx[i];

                        XMFLOAT3 lp = { positions[vi*3+0], positions[vi*3+1], positions[vi*3+2] };
                        XMStoreFloat3(&p[i], XMVector3TransformCoord(XMLoadFloat3(&lp), worldTransform));

                        if (!normals.empty()) {
                            XMFLOAT3 ln = { normals[vi*3+0], normals[vi*3+1], normals[vi*3+2] };
                            XMStoreFloat3(&n[i], XMVector3Normalize(XMVector3TransformNormal(XMLoadFloat3(&ln), normalMatrix)));
                        } else {
                            n[i] = { 0, 0, 0 };
                        }

                        // glTF UVs are top-left origin — no V-flip needed (unlike OBJ)
                        uv[i] = !texcoords.empty()
                            ? XMFLOAT2{ texcoords[vi*2+0], texcoords[vi*2+1] }
                            : XMFLOAT2{ 0, 0 };
                    }

                    // Face normal fallback
                    if (normals.empty()) {
                        XMVECTOR fn = XMVector3Normalize(XMVector3Cross(
                            XMLoadFloat3(&p[1]) - XMLoadFloat3(&p[0]),
                            XMLoadFloat3(&p[2]) - XMLoadFloat3(&p[0])));
                        XMStoreFloat3(&n[0], fn); n[1] = n[2] = n[0];
                    }

                    for (int i = 0; i < 3; ++i) {
                        Vertex v({ p[i] }, { n[i].x, n[i].y, n[i].z, (float)matID }, { uv[i] });
                        if (uniqueVertices.count(v)) {
                            indices->push_back(uniqueVertices[v]);
                        } else {
                            uint32_t newIndex = (uint32_t)vertices->size();
                            uniqueVertices[v] = newIndex;
                            indices->push_back(newIndex);
                            vertices->push_back(v);
                        }
                    }
                }
            }
        }

        *materialOffset += (UINT)model.materials_count;

        std::cout << "[GlbLoader] COMPLETED '" << inputfile << "'." << std::endl;
        std::cout << "  - Unique vertices: " << uniqueVertices.size() << std::endl;
        std::cout << "  - Total indices:   " << indices->size()       << std::endl;
        std::cout << "  - Materials:       " << mats->size()          << std::endl;
        std::cout << "  - Albedo textures: " << albedoTextures.size() << std::endl;
        std::cout << "  - Normal textures: " << normalTextures.size() << std::endl;
        std::cout << "  - RMA textures:    " << rmaTextures.size()    << std::endl;
        std::cout << "-----------------------------------------------------" << std::endl;

        // ---- 7. Cleanup v3 model ------------------------------------------------
        tg3_model_free(&model);
        tg3_error_stack_free(&errors);
    }

};

#endif //PATHTRACER_OBJLOADER_H
