//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_OBJLOADER_H
#define PATHTRACER_OBJLOADER_H
#define TINYOBJLOADER_IMPLEMENTATION
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

#include <string>
#include <map>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct TextureData;
constexpr float PI = 3.14159265359f;
constexpr int LUT_SIZE_THETA = 16; // Number of samples for cos(theta)
constexpr int NUM_SAMPLES_MC = 16000; // Monte Carlo samples per integral
//Sheen
constexpr float SHEEN_R_BAKE = 0.10f; // constant for now
constexpr int   NUM_SAMPLES_SHEEN = 4096;

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



void PrintLUTAsVector(const Material& mat) {
    std::wcout << L"1D LUT (Indexed by cosTheta):\n\n";

    // Print column headers for cosTheta values
    for (int idx = 0; idx < LUT_SIZE_THETA; ++idx) {
        float cosTheta = static_cast<float>(idx) / (LUT_SIZE_THETA - 1); // Normalize index

        // Print cosTheta value as label
        std::wcout << L"cosTheta = " << std::fixed << std::setprecision(2) << cosTheta << L": ";

        // Print LUT value at this index
        std::wcout << std::fixed << std::setprecision(3) << 1.0f + (1.0f -mat.LUT[idx]) / mat.LUT[idx] << L"\n";
    }
}

void PrintSheenLUT(const Material& mat) {
    std::wcout << L"Sheen 1D LUT (by cosTheta):\n\n";
    for (int idx = 0; idx < LUT_SIZE_THETA; ++idx) {
        float cosTheta = static_cast<float>(idx) / (LUT_SIZE_THETA - 1);
        std::wcout << L"cosTheta = " << std::fixed << std::setprecision(2) << cosTheta << L": "
                   << std::fixed << std::setprecision(4) << mat.SheenLUT[idx] << L"\n";
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
        float sinTheta = sqrt((std::fmax)(EPSILON, 1.0f - cosTheta * cosTheta));

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

void GenerateSheenLUT(Material& mat)
{
    constexpr float EPSILON = 0.04f;
    XMFLOAT3 N = {0.0f, 0.0f, 1.0f};

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int thetaIdx = 0; thetaIdx < LUT_SIZE_THETA; ++thetaIdx) {
        float cosTheta = EPSILON + static_cast<float>(thetaIdx) / (LUT_SIZE_THETA - 1) * (1.0f - EPSILON);
        float sinTheta = sqrtf((std::fmax)(EPSILON, 1.0f - cosTheta * cosTheta));

        XMFLOAT3 V = { sinTheta, 0.0f, cosTheta };

        mat.SheenLUT[thetaIdx] = ComputeSheenDirectionalAlbedo(N, normalize(V), SHEEN_R_BAKE, NUM_SAMPLES_SHEEN);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::wcout << L"GenerateSheenLUT (1D x " << LUT_SIZE_THETA << L") in " << ms << L" ms\n";
}

class ObjLoader {
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
        tinyobj::ObjReaderConfig reader_config;
        reader_config.mtl_search_path = material_search_path;
        reader_config.triangulate = true;

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

        // Helper lambda to load a texture or return existing ID
        auto loadTexture = [&](
            const std::string& filename,
            const std::string& materialPath,
            std::vector<TextureData>& textureList) -> int
        {
            if (filename.empty()) {
                return -1;
            }

            std::string fullPath = materialPath + filename;

            if (textureMap.count(fullPath)) {
                return textureMap[fullPath];
            }

            TextureData texData;
            int req_comp = 4; // Always load as RGBA
            unsigned char* data = stbi_load(fullPath.c_str(), &texData.width, &texData.height, &texData.channels, req_comp);

            if (!data) {
                std::cerr << "Failed to load texture: " << fullPath << std::endl;
                return -1;
            }

            texData.pixels.assign(data, data + (size_t)texData.width * texData.height * 4);
            stbi_image_free(data);

            uint32_t textureID = static_cast<uint32_t>(textureList.size());
            textureList.push_back(std::move(texData));
            textureMap[fullPath] = textureID;

            std::cout << "Loaded texture " << fullPath << " with ID " << textureID << std::endl;

            return textureID;
        };

        Material defaultMaterial;
        mats->push_back(defaultMaterial);
        (*materialOffset)++;

        for (const auto& mat : materials) {
            Material t_mat;
            t_mat.Kd = { mat.diffuse[0], mat.diffuse[1], mat.diffuse[2], mat.dissolve };
            t_mat.Ke = { mat.emission[0], mat.emission[1], mat.emission[2] };
            t_mat.Ni = mat.ior;
            t_mat.Pr_Pm_Ps_Pc = { mat.roughness, mat.metallic, mat.sheen, mat.clearcoat_thickness };
            t_mat.Pcr_aniso_anisor = { mat.clearcoat_roughness, mat.anisotropy, mat.anisotropy_rotation };

            t_mat.albedoTexID = loadTexture(mat.diffuse_texname, material_search_path, albedoTextures);
            std::string normalTexName = !mat.normal_texname.empty() ? mat.normal_texname : mat.bump_texname;
            t_mat.normalTexID = loadTexture(normalTexName, material_search_path, normalTextures);
            if (mat.unknown_parameter.count("map_rma")) {
                t_mat.rmaTexID = loadTexture(mat.unknown_parameter.at("map_rma"), material_search_path, rmaTextures);
            }

            GenerateEssLUT(t_mat);
            GenerateSheenLUT(t_mat);
            PrintLUTAsVector(t_mat);
            mats->push_back(t_mat);
        }

        std::unordered_map<Vertex, uint32_t> uniqueVertices;
        const float EPS_LEN = 1e-12f;
        const float COS_TOL = 0.9998f;

        for (const auto& shape : shapes) {
            size_t index_offset = 0;
            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                int fv = shape.mesh.num_face_vertices[f];
                if (fv != 3) { index_offset += fv; continue; }

                tinyobj::index_t i0 = shape.mesh.indices[index_offset + 0];
                tinyobj::index_t i1 = shape.mesh.indices[index_offset + 1];
                tinyobj::index_t i2 = shape.mesh.indices[index_offset + 2];

                auto getPos = [&](const tinyobj::index_t& idx) { return XMFLOAT3(attrib.vertices[3 * idx.vertex_index + 0], attrib.vertices[3 * idx.vertex_index + 1], attrib.vertices[3 * idx.vertex_index + 2]); };
                XMFLOAT3 p0 = getPos(i0); XMFLOAT3 p1 = getPos(i1); XMFLOAT3 p2 = getPos(i2);

                auto getUV = [&](const tinyobj::index_t& idx) { if (idx.texcoord_index >= 0) { return XMFLOAT2(attrib.texcoords[2 * idx.texcoord_index + 0], 1.0f - attrib.texcoords[2 * idx.texcoord_index + 1]); } return XMFLOAT2(0.0f, 0.0f); };
                XMFLOAT2 uv0 = getUV(i0); XMFLOAT2 uv1 = getUV(i1); XMFLOAT2 uv2 = getUV(i2);

                XMVECTOR P0 = XMLoadFloat3(&p0); XMVECTOR P1 = XMLoadFloat3(&p1); XMVECTOR P2 = XMLoadFloat3(&p2);
                XMVECTOR faceN_un = XMVector3Cross(P1 - P0, P2 - P0);
                float faceN_len; XMStoreFloat(&faceN_len, XMVector3Length(faceN_un));
                XMFLOAT3 faceN_f(0, 0, 1); if (faceN_len > EPS_LEN) { XMStoreFloat3(&faceN_f, XMVector3Normalize(faceN_un)); }

                auto getObjN = [&](const tinyobj::index_t& idx) -> XMFLOAT3 { if (idx.normal_index >= 0) { return XMFLOAT3(attrib.normals[3 * idx.normal_index + 0], attrib.normals[3 * idx.normal_index + 1], attrib.normals[3 * idx.normal_index + 2]); } return XMFLOAT3(0, 0, 0); };
                XMFLOAT3 n0_src = getObjN(i0); XMFLOAT3 n1_src = getObjN(i1); XMFLOAT3 n2_src = getObjN(i2);

                auto lenSq = [](const XMFLOAT3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; };
                auto aligned = [&](const XMFLOAT3& n) { if (lenSq(n) <= EPS_LEN) return false; float dot; XMStoreFloat(&dot, XMVector3Dot(XMVector3Normalize(XMLoadFloat3(&n)), XMLoadFloat3(&faceN_f))); return dot > COS_TOL; };
                bool all_match_face = aligned(n0_src) && aligned(n1_src) && aligned(n2_src);

                int materialID = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;
                uint32_t matID = (materialID >= 0) ? uint32_t(materialID + *materialOffset) : 0u;
                materialIDs->push_back(matID);

                auto makeNormal = [&](const XMFLOAT3& src) { return all_match_face ? XMFLOAT4(0.0f, 0.0f, 0.0f, *materialVertexOffset) : XMFLOAT4(src.x, src.y, src.z, *materialVertexOffset); };

                Vertex v0(p0, makeNormal(n0_src), uv0);
                Vertex v1(p1, makeNormal(n1_src), uv1);
                Vertex v2(p2, makeNormal(n2_src), uv2);

                auto pushUnique = [&](const Vertex& v) -> uint32_t { auto it = uniqueVertices.find(v); if (it == uniqueVertices.end()) { uint32_t newIndex = static_cast<uint32_t>(vertices->size()); uniqueVertices[v] = newIndex; vertices->push_back(v); return newIndex; } return it->second; };
                indices->push_back(pushUnique(v0));
                indices->push_back(pushUnique(v1));
                indices->push_back(pushUnique(v2));

                index_offset += fv;
            }
        }
        *materialOffset += materials.size();
    }
};


#endif //PATHTRACER_OBJLOADER_H
