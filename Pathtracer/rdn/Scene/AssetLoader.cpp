#include "../stdafx.h"
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
    // Survey every sufficiently large mesh and split only when its predicted
    // traversal cost improves substantially, or the hard triangle limit requires it.
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

    // Coordinates far from the origin can round a bin boundary onto an endpoint.
    // Always make progress, even when a candidate split collapses numerically.
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

        auto finalizeMesh = [&](std::vector<Vertex> verts, std::vector<UINT> idxOpaqueFirst,
                                std::vector<UINT> matOpaqueFirst, UINT opaqueTris, UINT alphaTris) -> UINT {
            MeshGPU gpu{};
            gpu.cpuVertices = std::move(verts);
            gpu.cpuIndices = std::move(idxOpaqueFirst);
            gpu.cpuMaterialIDs = std::move(matOpaqueFirst);
            gpu.vertexCount = (UINT)gpu.cpuVertices.size();
            gpu.indexCount = (UINT)gpu.cpuIndices.size();
            gpu.opaqueTriCount = opaqueTris;
            gpu.alphaTriCount = alphaTris;
            gpu.materialIDBase = (UINT)scene.materialIDs.size();

            scene.materialIDs.insert(scene.materialIDs.end(), gpu.cpuMaterialIDs.begin(), gpu.cpuMaterialIDs.end());

            const UINT idx = (UINT)scene.meshes.size();
            scene.meshes.push_back(std::move(gpu));
            return idx;
        };

        std::vector<UINT> instCount(loaded.meshes.size(), 0);
        for (const auto& [meshIdx, xform] : loaded.instances)
            instCount[meshIdx]++;

        auto isMergeCandidate = [&](size_t mi) -> bool { return instCount[mi] <= 1; };

        int subIdx = 0;

        std::vector<int> keptSceneMesh(loaded.meshes.size(), -1);
        for (size_t mi = 0; mi < loaded.meshes.size(); ++mi) {
            if (isMergeCandidate(mi))
                continue;
            auto& srcMesh = loaded.meshes[mi];

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
            si.name = model.name + "_sub" + std::to_string(subIdx++);
            scene.instances.push_back(si);
        }

        std::vector<Vertex> mVerts;
        std::vector<UINT> mOpaqueIdx, mAlphaIdx, mOpaqueMat, mAlphaMat;
        int mergedCount = 0;

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
            // Static submeshes that move with this model remain mergeable. Survey
            // their combined geometry so distant parts do not create enormous,
            // mostly empty BLAS bounds. Shared meshes retain instancing above.
            auto pieces = SplitMeshSpatial(std::move(merged), MAX_TRIS_PER_MESH);
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
                si.name = model.name + "_merged" + std::to_string(mergedCount++);
                scene.instances.push_back(si);
            }

            mVerts.clear();
            mOpaqueIdx.clear();
            mAlphaIdx.clear();
            mOpaqueMat.clear();
            mAlphaMat.clear();
        };

        for (const auto& [meshIdx, localXform] : loaded.instances) {
            if (!isMergeCandidate(meshIdx))
                continue;
            auto& srcMesh = loaded.meshes[meshIdx];
            const UINT thisTris = (UINT)srcMesh.indices.size() / 3;
            const UINT curTris = (UINT)(mOpaqueIdx.size() + mAlphaIdx.size()) / 3;

            // Flush merged geometry before crossing the per-BLAS triangle cap.
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
        }
        flushMerged();

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
            // Transform the local root's eight corners, not just mesh vertices:
            // a rotated BLAS can have a larger TLAS bound than its geometry.
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
    // Upload heaps remain retained until each submitted batch is flushed.
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
