#include "../stdafx.h"
#include <cfloat>
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <limits>
#include "AssetLoader.h"
#include "OmmBuilder.h"
#include "BvhQuality.h"
#include "../DXRHelper.h"

static constexpr UINT64 TEXTURE_BATCH_BYTES = 256ull << 20;
static constexpr UINT TEXTURE_BATCH_MAX = 1024;

MeshSplitResult AssetLoader::SplitOpaqueAlpha(const std::vector<UINT>& indices, const std::vector<UINT>& perTriMatIDs,
                                              const MaterialSoA& allMaterials) {
    MeshSplitResult r;
    const UINT triCount = (UINT)indices.size() / 3;
    std::vector<UINT> opaqueIdx, alphaIdx, opaqueMatIDs, alphaMatIDs;

    for (UINT t = 0; t < triCount; ++t) {
        const UINT matID = perTriMatIDs[t];

        const bool isThinGlass = (matID < allMaterials.thinGlass.size()) && (allMaterials.thinGlass[matID] != 0u);
        const bool isTransmissive = allMaterials.Kd[matID].w < 1.0f;
        bool isAlpha = (allMaterials.alphaThreshold[matID] < 1.0f) || isTransmissive || isThinGlass;
        auto& dstIdx = isAlpha ? alphaIdx : opaqueIdx;
        auto& dstMat = isAlpha ? alphaMatIDs : opaqueMatIDs;
        dstIdx.push_back(indices[3 * t + 0]);
        dstIdx.push_back(indices[3 * t + 1]);
        dstIdx.push_back(indices[3 * t + 2]);
        dstMat.push_back(perTriMatIDs[t]);
    }

    r.opaqueTriCount = (UINT)opaqueIdx.size() / 3;
    r.alphaTriCount = (UINT)alphaIdx.size() / 3;
    r.reorderedIndices = std::move(opaqueIdx);
    r.reorderedIndices.insert(r.reorderedIndices.end(), alphaIdx.begin(), alphaIdx.end());
    r.reorderedMaterialIDs = std::move(opaqueMatIDs);
    r.reorderedMaterialIDs.insert(r.reorderedMaterialIDs.end(), alphaMatIDs.begin(), alphaMatIDs.end());
    return r;
}

std::vector<LoadedMesh> AssetLoader::SplitMeshSpatial(LoadedMesh mesh, UINT maxTris) {
    // Split only for a clear cost win or above maxTris.
    const UINT triCount = (UINT)mesh.indices.size() / 3;
    auto keep = [&]() {
        std::vector<LoadedMesh> out;
        out.push_back(std::move(mesh));
        return out;
    };
    if (triCount <= maxTris && triCount < 4096) return keep();
    auto boundsOf = [&](UINT t) {
        bvh::Bounds b;
        for (int k = 0; k < 3; ++k) {
            const auto& p = mesh.vertices[mesh.indices[3*t+k]].position;
            b.point(p.x, p.y, p.z);
        }
        return b;
    };
    const bvh::Split split = bvh::choose_split(triCount, boundsOf);
    if (triCount <= maxTris && (split.axis < 0 || split.ratio > .65)) return keep();

    std::vector<XMFLOAT3> centroids(triCount);
    const float fMax = std::numeric_limits<float>::max();
    XMFLOAT3 cMin{fMax, fMax, fMax};
    XMFLOAT3 cMax{-fMax, -fMax, -fMax};
    for (UINT t = 0; t < triCount; ++t) {
        const auto& p0 = mesh.vertices[mesh.indices[3 * t + 0]].position;
        const auto& p1 = mesh.vertices[mesh.indices[3 * t + 1]].position;
        const auto& p2 = mesh.vertices[mesh.indices[3 * t + 2]].position;
        centroids[t] = {(p0.x + p1.x + p2.x) / 3.0f, (p0.y + p1.y + p2.y) / 3.0f, (p0.z + p1.z + p2.z) / 3.0f};
        cMin.x = std::min(cMin.x, centroids[t].x);
        cMax.x = std::max(cMax.x, centroids[t].x);
        cMin.y = std::min(cMin.y, centroids[t].y);
        cMax.y = std::max(cMax.y, centroids[t].y);
        cMin.z = std::min(cMin.z, centroids[t].z);
        cMax.z = std::max(cMax.z, centroids[t].z);
    }

    const float ex = cMax.x - cMin.x;
    const float ey = cMax.y - cMin.y;
    const float ez = cMax.z - cMin.z;
    int axis = 0;
    if (ey > ex && ey >= ez)
        axis = 1;
    else if (ez > ex && ez > ey)
        axis = 2;
    if (split.axis >= 0) axis = split.axis;

    auto axisVal = [&](UINT t) -> float {
        return axis == 0 ? centroids[t].x : (axis == 1 ? centroids[t].y : centroids[t].z);
    };

    std::vector<UINT> triIdx(triCount);
    for (UINT t = 0; t < triCount; ++t)
        triIdx[t] = t;
    auto midIt = triIdx.begin() + triCount / 2;
    if (split.axis >= 0)
        midIt = std::partition(triIdx.begin(), triIdx.end(), [&](UINT t) { return boundsOf(t).center(axis) < split.position; });
    else
        std::nth_element(triIdx.begin(), midIt, triIdx.end(), [&](UINT a, UINT b) { return axisVal(a) < axisVal(b); });

    // Precision can collapse a split; fall back to the median.
    if (midIt == triIdx.begin() || midIt == triIdx.end()) {
        if (triCount <= maxTris) return keep();
        midIt = triIdx.begin() + triCount / 2;
        std::nth_element(triIdx.begin(), midIt, triIdx.end(), [&](UINT a, UINT b) { return axisVal(a) < axisVal(b); });
    }

    std::vector<uint8_t> isLeft(triCount, 0);
    for (auto it = triIdx.begin(); it != midIt; ++it)
        isLeft[*it] = 1;

    centroids.clear();
    centroids.shrink_to_fit();
    triIdx.clear();
    triIdx.shrink_to_fit();

    auto buildSide = [&](bool wantLeft) {
        LoadedMesh out;
        std::vector<int32_t> vertRemap(mesh.vertices.size(), -1);
        for (UINT t = 0; t < triCount; ++t) {
            if ((isLeft[t] != 0) != wantLeft)
                continue;
            for (int j = 0; j < 3; ++j) {
                const UINT oldVi = mesh.indices[3 * t + j];
                int32_t& newVi = vertRemap[oldVi];
                if (newVi < 0) {
                    newVi = (int32_t)out.vertices.size();
                    out.vertices.push_back(mesh.vertices[oldVi]);
                }
                out.indices.push_back((UINT)newVi);
            }
            out.perTriMaterialIDs.push_back(mesh.perTriMaterialIDs[t]);
        }
        return out;
    };

    LoadedMesh left = buildSide(true);
    LoadedMesh right = buildSide(false);

    mesh = LoadedMesh{};
    isLeft.clear();
    isLeft.shrink_to_fit();

    auto leftPieces = SplitMeshSpatial(std::move(left), maxTris);
    auto rightPieces = SplitMeshSpatial(std::move(right), maxTris);

    leftPieces.insert(leftPieces.end(), std::make_move_iterator(rightPieces.begin()),
                      std::make_move_iterator(rightPieces.end()));
    return leftPieces;
}

namespace {
// A far-field stand-in for an object: three planes crossed at its bounds' centre, facing x, z and y, each textured with
// the object as seen along its normal, orthographically: what covers a texel (alpha-tested) in alpha, their mean albedo
// in colour. The texture holds the three side by side; empty texels take neighbours' colours, for filtering.
struct StandIn {
    std::vector<Vertex> vertices;
    std::vector<UINT> indices;
    TextureData texture;
    UINT dominant = 0; // the material most of its coverage comes from
};

float SrgbToLinear(uint8_t c) {
    const float v = c / 255.0f;
    return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
}
uint8_t LinearToSrgb(float v) {
    v = std::clamp(v, 0.0f, 1.0f);
    const float s = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
    return (uint8_t)std::lround(s * 255.0f);
}

bool BakeStandIn(const std::vector<Vertex>& verts, const std::vector<UINT>& indices, const std::vector<UINT>& mats,
                 const MaterialSoA& materials, const std::vector<TextureData>& albedo, StandIn& out) {
    constexpr int R = 256; // texels per tile side
    if (verts.empty() || indices.empty())
        return false;
    float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX}, hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const Vertex& v : verts) {
        const float p[3] = {v.position.x, v.position.y, v.position.z};
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], p[a]);
            hi[a] = std::max(hi[a], p[a]);
        }
    }
    float ext[3], mid[3];
    for (int a = 0; a < 3; ++a) {
        ext[a] = std::max(hi[a] - lo[a], 1e-4f);
        mid[a] = 0.5f * (lo[a] + hi[a]);
    }
    // Tile k: its plane's normal axis, and the axes of its texture's u and v (v down from the top for upright tiles).
    const int normalAxis[3] = {0, 2, 1}, uAxis[3] = {2, 0, 0}, vAxis[3] = {1, 1, 2};
    const bool upright[3] = {true, true, false};
    std::vector<float> sum((size_t)3 * R * R * 3, 0.0f);
    std::vector<uint32_t> count((size_t)3 * R * R, 0);
    std::vector<uint64_t> byMaterial(materials.size(), 0);

    auto sample = [&](UINT m, float u, float v, float rgb[3]) -> bool {
        const DirectX::XMFLOAT4 kd = materials.Kd[m];
        rgb[0] = kd.x, rgb[1] = kd.y, rgb[2] = kd.z;
        const int tex = materials.albedoTexID[m];
        const bool alphaTested = materials.alphaThreshold[m] < 1.0f;
        if (tex < 0 || tex >= (int)albedo.size() || albedo[tex].image.GetImageCount() == 0)
            return true;
        const DirectX::Image* img = albedo[tex].image.GetImage(0, 0, 0);
        if (!img || DirectX::BitsPerPixel(img->format) != 32)
            return true;
        const DirectX::XMFLOAT2 s = materials.albedoUVScale[m];
        u *= s.x;
        v *= s.y;
        u -= std::floor(u);
        v -= std::floor(v);
        const size_t x = std::min(img->width - 1, (size_t)(u * img->width));
        const size_t y = std::min(img->height - 1, (size_t)(v * img->height));
        const uint8_t* t = img->pixels + y * img->rowPitch + x * 4;
        uint8_t a = t[3];
        if (materials.invertAlpha[m])
            a = 255 - a;
        if (alphaTested && a < materials.alphaThreshold[m] * 255.0f)
            return false;
        const bool srgb = DirectX::IsSRGB(img->format);
        for (int c = 0; c < 3; ++c)
            rgb[c] *= srgb ? SrgbToLinear(t[c]) : t[c] / 255.0f;
        return true;
    };

    for (size_t t = 0; t + 2 < indices.size(); t += 3) {
        const Vertex* v[3] = {&verts[indices[t]], &verts[indices[t + 1]], &verts[indices[t + 2]]};
        const UINT m = mats[t / 3];
        for (int k = 0; k < 3; ++k) {
            float px[3], py[3];
            for (int i = 0; i < 3; ++i) {
                const float p[3] = {v[i]->position.x, v[i]->position.y, v[i]->position.z};
                px[i] = (p[uAxis[k]] - lo[uAxis[k]]) / ext[uAxis[k]] * R;
                py[i] = upright[k] ? (hi[vAxis[k]] - p[vAxis[k]]) / ext[vAxis[k]] * R
                                   : (p[vAxis[k]] - lo[vAxis[k]]) / ext[vAxis[k]] * R;
            }
            const float area = (px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]);
            if (std::fabs(area) < 1e-12f)
                continue;
            const int x0 = std::max(0, (int)std::floor(std::min({px[0], px[1], px[2]})));
            const int x1 = std::min(R - 1, (int)std::ceil(std::max({px[0], px[1], px[2]})));
            const int y0 = std::max(0, (int)std::floor(std::min({py[0], py[1], py[2]})));
            const int y1 = std::min(R - 1, (int)std::ceil(std::max({py[0], py[1], py[2]})));
            for (int y = y0; y <= y1; ++y)
                for (int x = x0; x <= x1; ++x) {
                    const float cx = x + 0.5f, cy = y + 0.5f;
                    const float w0 = ((px[1] - cx) * (py[2] - cy) - (px[2] - cx) * (py[1] - cy)) / area;
                    const float w1 = ((px[2] - cx) * (py[0] - cy) - (px[0] - cx) * (py[2] - cy)) / area;
                    const float w2 = 1.0f - w0 - w1;
                    if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
                        continue;
                    const float u = w0 * v[0]->texCoord.x + w1 * v[1]->texCoord.x + w2 * v[2]->texCoord.x;
                    const float tv = w0 * v[0]->texCoord.y + w1 * v[1]->texCoord.y + w2 * v[2]->texCoord.y;
                    float rgb[3];
                    if (!sample(m, u, tv, rgb))
                        continue;
                    const size_t at = ((size_t)k * R * R + (size_t)y * R + x);
                    for (int c = 0; c < 3; ++c)
                        sum[at * 3 + c] += rgb[c];
                    ++count[at];
                    ++byMaterial[m];
                }
        }
    }
    out.dominant = (UINT)(std::max_element(byMaterial.begin(), byMaterial.end()) - byMaterial.begin());

    // The texture: three tiles in a row, linear colour back to sRGB; empty texels grow their neighbours' colours.
    std::vector<float> rgb((size_t)3 * R * R * 3, 0.0f);
    std::vector<uint8_t> filled((size_t)3 * R * R, 0);
    for (size_t i = 0; i < count.size(); ++i)
        if (count[i]) {
            for (int c = 0; c < 3; ++c)
                rgb[i * 3 + c] = sum[i * 3 + c] / count[i];
            filled[i] = 1;
        }
    for (int pass = 0; pass < 8; ++pass) {
        std::vector<uint8_t> next = filled;
        for (int k = 0; k < 3; ++k)
            for (int y = 0; y < R; ++y)
                for (int x = 0; x < R; ++x) {
                    const size_t at = (size_t)k * R * R + (size_t)y * R + x;
                    if (filled[at])
                        continue;
                    float acc[3] = {0, 0, 0};
                    int n = 0;
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int nx = x + dx, ny = y + dy;
                            if (nx < 0 || ny < 0 || nx >= R || ny >= R)
                                continue;
                            const size_t nb = (size_t)k * R * R + (size_t)ny * R + nx;
                            if (!filled[nb])
                                continue;
                            for (int c = 0; c < 3; ++c)
                                acc[c] += rgb[nb * 3 + c];
                            ++n;
                        }
                    if (!n)
                        continue;
                    for (int c = 0; c < 3; ++c)
                        rgb[at * 3 + c] = acc[c] / n;
                    next[at] = 1;
                }
        filled.swap(next);
    }
    if (FAILED(out.texture.image.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 3 * R, R, 1, 1)))
        return false;
    const DirectX::Image* img = out.texture.image.GetImage(0, 0, 0);
    for (int k = 0; k < 3; ++k)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const size_t at = (size_t)k * R * R + (size_t)y * R + x;
                uint8_t* t = img->pixels + (size_t)y * img->rowPitch + ((size_t)k * R + x) * 4;
                for (int c = 0; c < 3; ++c)
                    t[c] = LinearToSrgb(rgb[at * 3 + c]);
                t[3] = count[at] ? 255 : 0;
            }
    out.texture.width = 3 * R;
    out.texture.height = R;
    out.texture.channels = 4;
    out.texture.original_width = 3 * R;
    out.texture.original_height = R;

    // The planes: two triangles each, over the bounds' cross-section, mapped to their tiles.
    for (int k = 0; k < 3; ++k) {
        const UINT base = (UINT)out.vertices.size();
        for (int c = 0; c < 4; ++c) {
            const float fu = (c & 1) ? 1.0f : 0.0f, fv = (c & 2) ? 1.0f : 0.0f;
            float p[3];
            p[normalAxis[k]] = mid[normalAxis[k]];
            p[uAxis[k]] = lo[uAxis[k]] + fu * ext[uAxis[k]];
            p[vAxis[k]] = upright[k] ? hi[vAxis[k]] - fv * ext[vAxis[k]] : lo[vAxis[k]] + fv * ext[vAxis[k]];
            float n[3] = {0, 0, 0};
            n[normalAxis[k]] = 1.0f;
            out.vertices.push_back(Vertex({p[0], p[1], p[2]}, {n[0], n[1], n[2], 0.0f}, {(k + fu) / 3.0f, fv}));
        }
        for (UINT i : {0u, 1u, 2u, 2u, 1u, 3u})
            out.indices.push_back(base + i);
    }
    return true;
}
} // namespace

void AssetLoader::SplitOversizedMeshes(LoadedScene& scene, UINT maxTris) {
    const size_t meshesIn = scene.meshes.size();
    const size_t instancesIn = scene.instances.size();
    std::wcout << L"[Split] SplitOversizedMeshes ENTER: meshes=" << meshesIn << L" instances=" << instancesIn
               << L" threshold=" << maxTris << L" tris" << std::endl;

    std::vector<LoadedMesh> newMeshes;
    std::vector<std::vector<UINT>> oldToNew(scene.meshes.size());
    newMeshes.reserve(scene.meshes.size());

    for (size_t i = 0; i < scene.meshes.size(); ++i) {
        const UINT triCount = (UINT)scene.meshes[i].indices.size() / 3;
        const UINT vtxCount = (UINT)scene.meshes[i].vertices.size();
        std::wcout << L"[Split]   mesh[" << i << L"] tris=" << triCount << L" verts=" << vtxCount
                   << (triCount > maxTris ? L"  (limit requires splitting)" : L"  (checking spatial cost)") << std::endl;

        const std::string name = scene.meshes[i].name;
        auto pieces = SplitMeshSpatial(std::move(scene.meshes[i]), maxTris);

        if (pieces.size() > 1) {
            std::wcout << L"[Split]   -> produced " << pieces.size() << L" pieces:" << std::endl;
            for (size_t p = 0; p < pieces.size(); ++p) {
                std::wcout << L"[Split]      piece[" << p << L"] tris=" << (pieces[p].indices.size() / 3) << L" verts="
                           << pieces[p].vertices.size() << std::endl;
            }
        }

        oldToNew[i].reserve(pieces.size());
        for (auto& p : pieces) {
            p.name = name;
            oldToNew[i].push_back((UINT)newMeshes.size());
            newMeshes.push_back(std::move(p));
        }
    }
    scene.meshes = std::move(newMeshes);

    std::vector<std::pair<UINT, XMMATRIX>> newInstances;
    newInstances.reserve(scene.instances.size());
    for (const auto& [oldIdx, xform] : scene.instances) {
        for (UINT newIdx : oldToNew[oldIdx])
            newInstances.push_back({newIdx, xform});
    }
    scene.instances = std::move(newInstances);

    std::wcout << L"[Split] SplitOversizedMeshes EXIT: meshes " << meshesIn << L" -> " << scene.meshes.size()
               << L"  instances " << instancesIn << L" -> " << scene.instances.size() << std::endl;
}

void AssetLoader::LoadModels(const std::vector<ModelEntry>& modelEntries, Scene& scene, ID3D12Device* device,
                             ID3D12GraphicsCommandList* cmdList, const FlushFn& flushAndReset) {
    std::map<std::string, uint32_t> textureMap;
    std::vector<TextureData> albedoTextures, normalTextures, rmaTextures;
    // Streamed cluster hierarchies need the page pool in front of all mesh geometry.
    if (clusters::paging_enabled() && !scene.pagePoolTris && !scene.ReservePagePool())
        std::printf("[pager] meshes loaded before; this load keeps every hierarchy level resident\n");

    // Micromaps for alpha pages, baked as the pages are written.
    OmmPageBaker ommPages(scene.materials, [&albedoTextures](int texID) -> const DirectX::Image* {
        return texID < (int)albedoTextures.size() && albedoTextures[texID].image.GetImageCount() > 0
                   ? albedoTextures[texID].image.GetImage(0, 0, 0)
                   : nullptr;
    });

    for (const auto& entry : modelEntries) {
        const auto& modelPath = entry.path;
        const XMMATRIX& modelXform = entry.transform;

        SceneModel model;
        model.filePath = modelPath;
        model.name = entry.name.empty() ? modelPath : entry.name;
        model.worldTransform = modelXform;
        model.meshStart = (UINT)scene.meshes.size();
        model.instanceStart = (UINT)scene.instances.size();

        XMVECTOR scaleV, rotQ, transV;
        XMMatrixDecompose(&scaleV, &rotQ, &transV, modelXform);
        XMStoreFloat3(&model.position, transV);
        XMStoreFloat3(&model.scale, scaleV);
        XMFLOAT4 q;
        XMStoreFloat4(&q, rotQ);
        float sinr = 2.0f * (q.w * q.x + q.y * q.z);
        float cosr = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
        model.rotation.x = XMConvertToDegrees(atan2f(sinr, cosr));
        float sinp = 2.0f * (q.w * q.y - q.z * q.x);
        model.rotation.y = XMConvertToDegrees(fabsf(sinp) >= 1.0f ? copysignf(XM_PIDIV2, sinp) : asinf(sinp));
        float siny = 2.0f * (q.w * q.z + q.x * q.y);
        float cosy = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
        model.rotation.z = XMConvertToDegrees(atan2f(siny, cosy));

        std::string matSearchPath = "./";
        auto lastSlash = modelPath.find_last_of("/\\");
        if (lastSlash != std::string::npos)
            matSearchPath = modelPath.substr(0, lastSlash + 1);

        std::string ext = modelPath.substr(modelPath.find_last_of('.') + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        LoadedScene loaded;
        if (ext == "glb" || ext == "gltf")
            loaded = ObjLoader::loadGlbFile(modelPath, textureMap, albedoTextures, normalTextures, rmaTextures,
                                            matSearchPath);
        else
            loaded = ObjLoader::loadObjFile(modelPath, textureMap, albedoTextures, normalTextures, rmaTextures,
                                            matSearchPath);

        SplitOversizedMeshes(loaded, MAX_TRIS_PER_MESH);

        const UINT globalMatBase = (UINT)scene.materials.size();
        scene.materials.append(loaded.materials);
        scene.materialNames.insert(scene.materialNames.end(), loaded.materialNames.begin(), loaded.materialNames.end());

        uint32_t lodMeshes = 0;
        uint64_t lodTrisBefore = 0, lodTrisAfter = 0;
        double lodMs = 0.0;
        auto finalizeMesh = [&](std::vector<Vertex> verts, std::vector<UINT> idxOpaqueFirst,
                                std::vector<UINT> matOpaqueFirst, UINT opaqueTris, UINT alphaTris) -> UINT {
            MeshGPU gpu{};
            if (!verts.empty()) {
                gpu.boundsMin = gpu.boundsMax = verts[0].position;
                for (const Vertex& v : verts) {
                    gpu.boundsMin = {std::min(gpu.boundsMin.x, v.position.x), std::min(gpu.boundsMin.y, v.position.y),
                                     std::min(gpu.boundsMin.z, v.position.z)};
                    gpu.boundsMax = {std::max(gpu.boundsMax.x, v.position.x), std::max(gpu.boundsMax.y, v.position.y),
                                     std::max(gpu.boundsMax.z, v.position.z)};
                }
            }
            gpu.cpuVertices = std::move(verts);
            gpu.cpuIndices = std::move(idxOpaqueFirst);
            gpu.cpuMaterialIDs = std::move(matOpaqueFirst);
            gpu.vertexCount = (UINT)gpu.cpuVertices.size();
            gpu.indexCount = (UINT)gpu.cpuIndices.size();
            gpu.opaqueTriCount = opaqueTris;
            gpu.alphaTriCount = alphaTris;
            gpu.materialIDBase = (UINT)scene.materialIDs.size();
            if (clusters::lod_enabled()) {
                // A hierarchy per material over the triangles that emit nothing; emitters stay as they are, as the
                // lights are their triangles. Alpha triangles join when paged, with micromaps baked per page.
                const auto t0 = std::chrono::steady_clock::now();
                const bool paged = scene.pagePoolTris && clusters::paging_enabled();
                const UINT lodTris = paged ? opaqueTris + alphaTris : opaqueTris;
                std::vector<uint32_t> regions(lodTris, clusters::NO_REGION);
                for (UINT t = 0; t < lodTris; ++t) {
                    const UINT m = gpu.cpuMaterialIDs[t];
                    const XMFLOAT3& ke = scene.materials.Ke[m];
                    if (ke.x + ke.y + ke.z <= 0.0f) regions[t] = m;
                }
                const UINT before = gpu.indexCount / 3;
                const auto& mats = scene.materials;
                auto cutout = [&mats](uint32_t m) {
                    const bool glass = m < mats.thinGlass.size() && mats.thinGlass[m] != 0u;
                    return mats.alphaThreshold[m] < 1.0f && mats.Kd[m].w >= 1.0f && !glass;
                };
                auto alpha = [&mats](uint32_t m) {
                    const bool glass = m < mats.thinGlass.size() && mats.thinGlass[m] != 0u;
                    return mats.alphaThreshold[m] < 1.0f || mats.Kd[m].w < 1.0f || glass;
                };
                auto opacity = [&](uint32_t m, int maxLevel, const float* uvs, uint32_t vertexCount,
                                   const uint32_t* indices, uint32_t triCount, clusters::PageOmm& out) {
                    if (!alpha(m))
                        return false;
                    ommPages.Bake(m, maxLevel, uvs, vertexCount, indices, triCount, out);
                    return true;
                };
                if (paged) {
                    if (gpu.BuildPagedLod(std::move(regions), globalMatBase, cutout, opacity)) {
                        ++lodMeshes;
                        lodTrisBefore += before;
                        lodTrisAfter += gpu.indexCount / 3;
                    }
                } else if (gpu.BuildClusterLod(std::move(regions))) {
                    ++lodMeshes;
                    lodTrisBefore += before;
                    lodTrisAfter += gpu.indexCount / 3;
                }
                lodMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            } else {
                gpu.OrderForClusters();
            }

            if (gpu.paged) {
                // Its own range of the page pool; its material IDs at the same triangle positions as its indices.
                uint64_t v = 0, t = 0;
                if (!scene.pagePoolVertexAlloc.allocate(gpu.vertexCount, v) ||
                    !scene.pagePoolTriAlloc.allocate(gpu.indexCount / 3, t))
                    throw std::runtime_error("The page pool cannot hold the streamed meshes' fixed triangles; raise "
                                             "RDN_PAGE_POOL_VERTS / RDN_PAGE_POOL_TRIS");
                gpu.poolVertexBase = (UINT)v;
                gpu.poolTriBase = (UINT)t;
                gpu.materialIDBase = (UINT)t;
                std::copy(gpu.cpuMaterialIDs.begin(), gpu.cpuMaterialIDs.end(), scene.materialIDs.begin() + (ptrdiff_t)t);
            } else {
                scene.materialIDs.insert(scene.materialIDs.end(), gpu.cpuMaterialIDs.begin(), gpu.cpuMaterialIDs.end());
            }

            const UINT idx = (UINT)scene.meshes.size();
            scene.meshes.push_back(std::move(gpu));
            return idx;
        };

        std::vector<UINT> instCount(loaded.meshes.size(), 0);
        for (const auto& [meshIdx, xform] : loaded.instances)
            instCount[meshIdx]++;

        // Meshes used once merge into spatial pieces; those the test scatter (RDN_SCATTER=<count>:<name part>) names
        // merge apart, into the object it copies.
        std::string scatterPart;
        {
            char v[256] = {};
            if (GetEnvironmentVariableA("RDN_SCATTER", v, sizeof(v)) > 0 && std::strchr(v, ':'))
                scatterPart = std::strchr(v, ':') + 1;
        }
        auto lower = [](std::string s) {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            return s;
        };
        scatterPart = lower(scatterPart);
        // Decided up front: a mesh's source is cleared once it is used.
        std::vector<uint8_t> scatterMesh(loaded.meshes.size());
        for (size_t mi = 0; mi < loaded.meshes.size(); ++mi)
            scatterMesh[mi] = !scatterPart.empty() && lower(loaded.meshes[mi].name).find(scatterPart) != std::string::npos;
        auto isMergeCandidate = [&](size_t mi) -> bool { return instCount[mi] <= 1; };

        int subIdx = 0;

        std::vector<int> keptSceneMesh(loaded.meshes.size(), -1);
        std::vector<std::string> meshNames(loaded.meshes.size());
        for (size_t mi = 0; mi < loaded.meshes.size(); ++mi) {
            if (isMergeCandidate(mi))
                continue;
            auto& srcMesh = loaded.meshes[mi];
            meshNames[mi] = srcMesh.name;

            std::vector<UINT> globalMatIDs = srcMesh.perTriMaterialIDs;
            for (auto& mid : globalMatIDs)
                mid += globalMatBase;

            auto split = SplitOpaqueAlpha(srcMesh.indices, globalMatIDs, scene.materials);
            keptSceneMesh[mi] =
                (int)finalizeMesh(std::move(srcMesh.vertices), std::move(split.reorderedIndices),
                                  std::move(split.reorderedMaterialIDs), split.opaqueTriCount, split.alphaTriCount);
            srcMesh = LoadedMesh{};
        }
        for (const auto& [meshIdx, xform] : loaded.instances) {
            if (isMergeCandidate(meshIdx))
                continue;
            SceneInstance si{};
            si.meshIndex = (UINT)keptSceneMesh[meshIdx];
            si.modelIndex = (UINT)scene.models.size();
            si.localTransform = xform;
            si.worldTransform = xform * modelXform;
            si.prevWorldTransform = si.worldTransform;
            si.name = model.name + "_sub" + std::to_string(subIdx++) +
                      (meshNames[meshIdx].empty() ? std::string() : "_" + meshNames[meshIdx]);
            scene.instances.push_back(si);
        }

        std::vector<Vertex> mVerts;
        std::vector<UINT> mOpaqueIdx, mAlphaIdx, mOpaqueMat, mAlphaMat;
        int mergedCount = 0;
        std::string mergedLabel = "_merged";

        auto flushMerged = [&]() {
            if (mVerts.empty())
                return;
            std::vector<UINT> idx = std::move(mOpaqueIdx);
            idx.insert(idx.end(), mAlphaIdx.begin(), mAlphaIdx.end());
            std::vector<UINT> mat = std::move(mOpaqueMat);
            mat.insert(mat.end(), mAlphaMat.begin(), mAlphaMat.end());

            LoadedMesh merged;
            merged.vertices = std::move(mVerts);
            merged.indices = std::move(idx);
            merged.perTriMaterialIDs = std::move(mat);
            // Split so distant parts don't bloat BLAS bounds; the scatter's object stays whole.
            std::vector<LoadedMesh> pieces;
            if (mergedLabel == "_merged")
                pieces = SplitMeshSpatial(std::move(merged), MAX_TRIS_PER_MESH);
            else
                pieces.push_back(std::move(merged));
            for (auto& piece : pieces) {
                auto split = SplitOpaqueAlpha(piece.indices, piece.perTriMaterialIDs, scene.materials);
                const UINT sceneMesh = finalizeMesh(std::move(piece.vertices), std::move(split.reorderedIndices),
                    std::move(split.reorderedMaterialIDs), split.opaqueTriCount, split.alphaTriCount);
                SceneInstance si{};
                si.meshIndex = sceneMesh;
                si.modelIndex = (UINT)scene.models.size();
                si.localTransform = XMMatrixIdentity();
                si.worldTransform = modelXform;
                si.prevWorldTransform = si.worldTransform;
                si.name = model.name + mergedLabel + std::to_string(mergedCount++);
                scene.instances.push_back(si);
            }

            mVerts.clear();
            mOpaqueIdx.clear();
            mAlphaIdx.clear();
            mOpaqueMat.clear();
            mAlphaMat.clear();
        };

        auto mergeInto = [&](UINT meshIdx, const XMMATRIX& localXform) {
            auto& srcMesh = loaded.meshes[meshIdx];
            const UINT thisTris = (UINT)srcMesh.indices.size() / 3;
            const UINT curTris = (UINT)(mOpaqueIdx.size() + mAlphaIdx.size()) / 3;

            if (curTris > 0 && curTris + thisTris > MAX_TRIS_PER_MESH)
                flushMerged();

            const UINT vbase = (UINT)mVerts.size();
            XMVECTOR det;
            const XMMATRIX nrmMat = XMMatrixTranspose(XMMatrixInverse(&det, localXform));
            const bool mirrored = XMVectorGetX(det) < 0.0f;
            mVerts.reserve(mVerts.size() + srcMesh.vertices.size());
            for (const auto& sv : srcMesh.vertices) {
                Vertex v = sv;
                XMVECTOR p = XMVector3TransformCoord(XMLoadFloat3(&sv.position), localXform);
                XMStoreFloat3(&v.position, p);
                XMVECTOR n = XMVector3TransformNormal(
                    XMVectorSet(sv.normal_material.x, sv.normal_material.y, sv.normal_material.z, 0.0f), nrmMat);
                n = XMVector3Normalize(n);
                v.normal_material.x = XMVectorGetX(n);
                v.normal_material.y = XMVectorGetY(n);
                v.normal_material.z = XMVectorGetZ(n);
                mVerts.push_back(v);
            }

            for (UINT t = 0; t < thisTris; ++t) {
                const UINT gMat = srcMesh.perTriMaterialIDs[t] + globalMatBase;

                const bool isThinGlass =
                    (gMat < scene.materials.thinGlass.size()) && (scene.materials.thinGlass[gMat] != 0u);
                const bool isTransmissive = scene.materials.Kd[gMat].w < 1.0f;
                const bool isAlpha = scene.materials.alphaThreshold[gMat] < 1.0f || isTransmissive || isThinGlass;
                auto& dstIdx = isAlpha ? mAlphaIdx : mOpaqueIdx;
                auto& dstMat = isAlpha ? mAlphaMat : mOpaqueMat;
                dstIdx.push_back(srcMesh.indices[3 * t + 0] + vbase);

                dstIdx.push_back(srcMesh.indices[3 * t + (mirrored ? 2 : 1)] + vbase);
                dstIdx.push_back(srcMesh.indices[3 * t + (mirrored ? 1 : 2)] + vbase);
                dstMat.push_back(gMat);
            }

            srcMesh = LoadedMesh{};
        };
        if (!scatterPart.empty()) {
            // Its parts, each a mesh of its own as well as part of the whole.
            const XMVECTOR shift = XMVectorSet(scene.sceneOriginWorld.x, scene.sceneOriginWorld.y,
                                               scene.sceneOriginWorld.z, 0.0f);
            mergedLabel = "_scatter_" + scatterPart + "_";
            for (const auto& [meshIdx, localXform] : loaded.instances) {
                if (!isMergeCandidate(meshIdx) || !scatterMesh[meshIdx])
                    continue;
                LoadedMesh part = loaded.meshes[meshIdx];
                for (auto& mid : part.perTriMaterialIDs)
                    mid += globalMatBase;
                auto split = SplitOpaqueAlpha(part.indices, part.perTriMaterialIDs, scene.materials);
                const UINT partMesh = finalizeMesh(std::move(part.vertices), std::move(split.reorderedIndices),
                                                   std::move(split.reorderedMaterialIDs), split.opaqueTriCount,
                                                   split.alphaTriCount);
                XMMATRIX placed = localXform * modelXform;
                placed.r[3] = XMVectorSubtract(placed.r[3], shift);
                scene.scatter.parts.push_back({partMesh, placed});
                mergeInto(meshIdx, localXform);
            }
            // Its far-field stand-in, from the whole before it is finalized.
            StandIn stand;
            std::vector<UINT> standIdx = mOpaqueIdx, standMat = mOpaqueMat;
            standIdx.insert(standIdx.end(), mAlphaIdx.begin(), mAlphaIdx.end());
            standMat.insert(standMat.end(), mAlphaMat.begin(), mAlphaMat.end());
            if (BakeStandIn(mVerts, standIdx, standMat, scene.materials, albedoTextures, stand)) {
                Material m;
                m.albedoTexID = (int)albedoTextures.size();
                m.alphaThreshold = 0.5f;
                m.Pr_Pm_Ps_Pc = scene.materials.Pr_Pm_Ps_Pc[stand.dominant];
                albedoTextures.push_back(std::move(stand.texture));
                scene.materials.push_back(m);
                scene.materialNames.push_back("scatter stand-in");
                const UINT mat = (UINT)scene.materials.size() - 1;
                std::vector<UINT> mats(stand.indices.size() / 3, mat);
                auto split = SplitOpaqueAlpha(stand.indices, mats, scene.materials);
                scene.scatter.standIn = finalizeMesh(std::move(stand.vertices), std::move(split.reorderedIndices),
                                                     std::move(split.reorderedMaterialIDs), split.opaqueTriCount,
                                                     split.alphaTriCount);
            }
            const size_t before = scene.instances.size();
            flushMerged();
            if (scene.instances.size() > before) {
                scene.scatter.wholeInstance = (UINT)before;
                scene.scatter.whole = scene.instances[before].meshIndex;
            }
            mergedLabel = "_merged";
            std::printf("[sets] RDN_SCATTER: \"%s\" is %zu parts%s\n", scatterPart.c_str(), scene.scatter.parts.size(),
                        scene.scatter.standIn != ~0u ? ", with a stand-in" : "");
        }
        for (const auto& [meshIdx, localXform] : loaded.instances)
            if (isMergeCandidate(meshIdx) && !scatterMesh[meshIdx])
                mergeInto(meshIdx, localXform);
        flushMerged();

        if (lodMeshes)
            std::printf(scene.pagePoolTris ? "[lod] %s: %u meshes paged, %.2fM triangles -> %.2fM kept in memory, %.0f ms\n"
                                           : "[lod] %s: %u meshes with a cluster hierarchy, %.2fM triangles -> %.2fM over all levels, %.0f ms\n",
                        model.name.c_str(), lodMeshes, (double)lodTrisBefore / 1e6, (double)lodTrisAfter / 1e6, lodMs);

        model.meshCount = (UINT)scene.meshes.size() - model.meshStart;
        model.instanceCount = (UINT)scene.instances.size() - model.instanceStart;
        std::vector<bvh::Bounds> meshBounds(model.meshCount);
        for (UINT i = 0; i < model.meshCount; ++i) {
            const auto& mesh = scene.meshes[model.meshStart + i];
            for (UINT index : mesh.cpuIndices) {
                const auto& p = mesh.cpuVertices[index].position;
                meshBounds[i].point(p.x, p.y, p.z);
            }
        }
        std::vector<bvh::Bounds> bounds;
        std::vector<uint32_t> triangleCounts;
        for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
            const auto& instance = scene.instances[i];
            const auto& mesh = scene.meshes[instance.meshIndex];
            bvh::Bounds b;
            // Box corners, not vertices: the TLAS bounds the BLAS box.
            const auto& local = meshBounds[instance.meshIndex - model.meshStart];
            for (UINT corner = 0; local.valid() && corner < 8; ++corner) {
                const XMFLOAT3 v{(corner & 1) ? local.hi[0] : local.lo[0],
                    (corner & 2) ? local.hi[1] : local.lo[1], (corner & 4) ? local.hi[2] : local.lo[2]};
                XMFLOAT3 p;
                XMStoreFloat3(&p, XMVector3TransformCoord(XMLoadFloat3(&v), instance.worldTransform));
                b.point(p.x, p.y, p.z);
            }
            bounds.push_back(b); triangleCounts.push_back(mesh.indexCount / 3);
        }
        model.bvhSurvey = bvh::survey(bounds, triangleCounts);
        std::wcout << L"[BVH survey] " << std::wstring(model.name.begin(), model.name.end())
            << L": " << model.bvhSurvey.instances << L" instances, " << model.bvhSurvey.triangles
            << L" triangles, " << model.bvhSurvey.overlapPairs << L" overlapping root-bound pairs" << std::endl;

        UINT modelIdx = (UINT)scene.models.size();
        for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i)
            scene.instances[i].modelIndex = modelIdx;

        std::wcout << L"[AssetLoader] Model '" << std::wstring(model.name.begin(), model.name.end())
                   << L"' registered: meshes=" << model.meshCount << L" instances=" << model.instanceCount
                   << L" (merged " << loaded.instances.size() << L" sub-objects -> " << mergedCount << L" blas)"
                   << std::endl;

        scene.models.push_back(std::move(model));
    }

    if (scene.pagePoolTris &&
        std::none_of(scene.meshes.begin(), scene.meshes.end(), [](const MeshGPU& m) { return m.paged; }))
        scene.ReleasePagePool();

    {
        SCOPE_TIMER("OMM Bake");

        std::vector<DirectX::ScratchImage*> albedoImgPtrs(albedoTextures.size(), nullptr);
        std::vector<DirectX::XMFLOAT2> albedoScales(albedoTextures.size(), {1.0f, 1.0f});
        for (size_t i = 0; i < albedoTextures.size(); ++i) {
            if (albedoTextures[i].image.GetImageCount() > 0)
                albedoImgPtrs[i] = &albedoTextures[i].image;
        }

        for (size_t i = 0; i < scene.materials.size(); ++i) {
            int id = scene.materials.albedoTexID[i];
            if (id >= 0 && (uint32_t)id < albedoScales.size())
                albedoScales[id] = scene.materials.albedoUVScale[i];
        }

        OmmBuilder::BakeAll(scene.meshes, scene.materials, albedoImgPtrs);
    }

    scene.bindlessAlbedoBase = BINDLESS_HEAP_START;
    scene.bindlessNormalBase = scene.bindlessAlbedoBase + (UINT)albedoTextures.size();
    scene.bindlessRmaBase = scene.bindlessNormalBase + (UINT)normalTextures.size();
    scene.totalBindlessTextures = (UINT)(albedoTextures.size() + normalTextures.size() + rmaTextures.size());

    for (size_t i = 0; i < scene.materials.size(); ++i) {
        if (scene.materials.albedoTexID[i] >= 0)
            scene.materials.albedoTexID[i] += (int)scene.bindlessAlbedoBase;
        if (scene.materials.normalTexID[i] >= 0)
            scene.materials.normalTexID[i] += (int)scene.bindlessNormalBase;
        if (scene.materials.rmaTexID[i] >= 0)
            scene.materials.rmaTexID[i] += (int)scene.bindlessRmaBase;
    }

    CreateBindlessTextures(albedoTextures, scene.bindlessAlbedoBase, L"Albedo", device, cmdList,
                           scene.bindlessGpuTextures, flushAndReset);
    CreateBindlessTextures(normalTextures, scene.bindlessNormalBase, L"Normal", device, cmdList,
                           scene.bindlessGpuTextures, flushAndReset);
    CreateBindlessTextures(rmaTextures, scene.bindlessRmaBase, L"RMA", device, cmdList, scene.bindlessGpuTextures,
                           flushAndReset);
}

void AssetLoader::CreateBindlessTextures(std::vector<TextureData>& textures, UINT heapBaseSlot,
                                         const std::wstring& debugPrefix, ID3D12Device* device,
                                         ID3D12GraphicsCommandList* cmdList,
                                         std::vector<ComPtr<ID3D12Resource>>& outGpuTextures,
                                         const FlushFn& flushAndReset) {
    // Kept alive until each batch is flushed.
    std::vector<ComPtr<ID3D12Resource>> batchUploadHeaps;
    UINT batchCount = 0;
    UINT64 batchBytes = 0;
    size_t batchStart = 0;

    for (size_t i = 0; i < textures.size(); ++i) {
        auto& tex = textures[i];
        const auto& meta = tex.image.GetMetadata();

        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = meta.width;
        d.Height = (UINT)meta.height;
        d.DepthOrArraySize = 1;
        d.MipLevels = (UINT16)meta.mipLevels;
        d.Format = meta.format;
        d.SampleDesc.Count = 1;

        ComPtr<ID3D12Resource> gpuTex;
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kDefaultHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpuTex)));

        const UINT subCount = (UINT)meta.mipLevels;
        std::vector<D3D12_SUBRESOURCE_DATA> sr(subCount);
        for (UINT m = 0; m < subCount; ++m) {
            const auto* img = tex.image.GetImage(m, 0, 0);
            sr[m].pData = img->pixels;
            sr[m].RowPitch = (LONG_PTR)img->rowPitch;
            sr[m].SlicePitch = (LONG_PTR)img->slicePitch;
        }

        UINT64 uploadSize = GetRequiredIntermediateSize(gpuTex.Get(), 0, subCount);
        ComPtr<ID3D12Resource> upload;
        auto ud = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &ud,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&upload)));

        UpdateSubresources(cmdList, gpuTex.Get(), upload.Get(), 0, 0, subCount, sr.data());

        auto b = CD3DX12_RESOURCE_BARRIER::Transition(gpuTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, kSRV);
        cmdList->ResourceBarrier(1, &b);

        outGpuTextures.push_back(gpuTex);
        batchUploadHeaps.push_back(std::move(upload));
        ++batchCount;
        batchBytes += uploadSize;

        if (batchBytes >= TEXTURE_BATCH_BYTES || batchCount >= TEXTURE_BATCH_MAX) {
            std::wcout << L"[AssetLoader] Flushing " << debugPrefix << L" texture batch (" << batchCount
                       << L" textures, " << (i + 1) << L"/" << textures.size() << L")" << std::endl;

            flushAndReset();
            batchUploadHeaps.clear();

            for (size_t j = batchStart; j <= i; ++j) {
                textures[j].image.Release();
            }
            batchCount = 0;
            batchBytes = 0;
            batchStart = i + 1;
        }
    }

    if (batchCount > 0) {
        std::wcout << L"[AssetLoader] Flushing final " << debugPrefix << L" batch (" << batchCount << L" textures)"
                   << std::endl;

        flushAndReset();
        batchUploadHeaps.clear();

        for (size_t j = batchStart; j < textures.size(); ++j) {
            textures[j].image.Release();
        }
    }
}
