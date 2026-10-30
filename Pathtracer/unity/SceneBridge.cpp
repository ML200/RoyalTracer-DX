#include "../rdn/stdafx.h"
#include <fstream>
#include "SceneBridge.h"
#include "../rdn/DXRHelper.h"
#include "../rdn/Scene/AssetLoader.h"
#include "../rdn/Scene/OmmBuilder.h"
#include <DirectXTex.h>

namespace scene_bridge {

XMMATRIX ToMatrix(const float* m) {
    return XMMATRIX(m[0], m[4], m[8], 0.0f, //
                    m[1], m[5], m[9], 0.0f, //
                    m[2], m[6], m[10], 0.0f, //
                    m[3], m[7], m[11], 1.0f);
}

static TextureData MakeTexture(StagedScene::Texture& t) {
    using namespace DirectX;
    const DXGI_FORMAT format =
        t.kind == RT_TEXTURE_ALBEDO ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
    ScratchImage base;
    ThrowIfFailed(base.Initialize2D(format, t.width, t.height, 1, 1));
    const Image* img = base.GetImage(0, 0, 0);
    for (uint32_t y = 0; y < t.height; ++y)
        memcpy(img->pixels + y * img->rowPitch, t.rgba.data() + (size_t)y * t.width * 4, (size_t)t.width * 4);
    TextureData td;
    // The renderer picks mips from ray cones, so every texture needs a full chain.
    ThrowIfFailed(GenerateMipMaps(*img, TEX_FILTER_DEFAULT, 0, td.image));
    td.width = td.original_width = (int)t.width;
    td.height = td.original_height = (int)t.height;
    td.channels = 4;
    t.rgba = {};
    return td;
}

void Build(StagedScene& staged, Scene& scene, ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
           const std::function<void()>& flush) {
    // Bindless ranges come in kind order: albedo, normal, RMA.
    std::vector<TextureData> lists[3];
    std::vector<int> localIndex(staged.textures.size(), -1);
    for (size_t i = 0; i < staged.textures.size(); ++i) {
        auto& t = staged.textures[i];
        const uint32_t kind = std::min(t.kind, (uint32_t)RT_TEXTURE_RMA);
        localIndex[i] = (int)lists[kind].size();
        lists[kind].push_back(MakeTexture(t));
    }
    auto texture = [&](int32_t id, uint32_t kind) {
        const bool valid = id >= 0 && (size_t)id < staged.textures.size() &&
                           std::min(staged.textures[id].kind, (uint32_t)RT_TEXTURE_RMA) == kind;
        return valid ? localIndex[id] : -1;
    };

    for (size_t i = 0; i < staged.materials.size(); ++i) {
        const RT_MaterialDesc& d = staged.materials[i];
        Material m;
        m.Kd = {d.baseColor[0], d.baseColor[1], d.baseColor[2], d.baseColor[3]};
        m.Ke = {d.emission[0], d.emission[1], d.emission[2]};
        m.Ni = d.ior > 0.0f ? d.ior : 1.5f;
        m.Pr_Pm_Ps_Pc = {std::clamp(d.roughness, 0.0f, 1.0f), std::clamp(d.metallic, 0.0f, 1.0f), 0.0f,
                         std::clamp(d.coat, 0.0f, 1.0f)};
        m.Pcr_aniso_anisor = {std::clamp(d.coatRoughness, 0.0f, 1.0f), 0.0f, 0.0f};
        m.Tf = {0.0f, 0.0f, 0.0f}; // opaque; only glass reads it
        m.albedoTexID = texture(d.albedoTexture, RT_TEXTURE_ALBEDO);
        m.normalTexID = texture(d.normalTexture, RT_TEXTURE_NORMAL);
        m.rmaTexID = texture(d.rmaTexture, RT_TEXTURE_RMA);
        const XMFLOAT2 uvScale = {d.uvScale[0] != 0.0f ? d.uvScale[0] : 1.0f, d.uvScale[1] != 0.0f ? d.uvScale[1] : 1.0f};
        m.albedoUVScale = m.normalUVScale = m.rmaUVScale = uvScale;
        m.alphaThreshold = m.albedoTexID >= 0 ? std::clamp(d.alphaCutoff, 0.0f, 1.0f) : 1.0f;
        scene.materials.push_back(m);
        scene.materialNames.push_back("host_" + std::to_string(i));
    }

    const UINT materialCount = (UINT)scene.materials.size();
    for (auto& sm : staged.meshes) {
        for (UINT& m : sm.triangleMaterials)
            m = std::min(m, materialCount - 1);
        auto split = AssetLoader::SplitOpaqueAlpha(sm.indices, sm.triangleMaterials, scene.materials);
        MeshGPU gpu{};
        gpu.cpuVertices = std::move(sm.vertices);
        gpu.cpuIndices = std::move(split.reorderedIndices);
        gpu.cpuMaterialIDs = std::move(split.reorderedMaterialIDs);
        gpu.vertexCount = (UINT)gpu.cpuVertices.size();
        gpu.indexCount = (UINT)gpu.cpuIndices.size();
        gpu.opaqueTriCount = split.opaqueTriCount;
        gpu.alphaTriCount = split.alphaTriCount;
        gpu.materialIDBase = (UINT)scene.materialIDs.size();
        scene.materialIDs.insert(scene.materialIDs.end(), gpu.cpuMaterialIDs.begin(), gpu.cpuMaterialIDs.end());
        scene.meshes.push_back(std::move(gpu));
    }

    SceneModel model;
    model.name = "Host scene";
    model.meshCount = (UINT)scene.meshes.size();
    model.instanceCount = (UINT)staged.instances.size();
    for (size_t i = 0; i < staged.instances.size(); ++i) {
        SceneInstance si{};
        si.meshIndex = std::min(staged.instances[i].mesh, (UINT)scene.meshes.size() - 1);
        si.modelIndex = 0;
        si.localTransform = staged.instances[i].world;
        si.worldTransform = si.localTransform;
        si.prevWorldTransform = si.localTransform;
        si.name = "host_" + std::to_string(i);
        scene.instances.push_back(si);
    }
    scene.models.push_back(std::move(model));

    // OMMs read albedo alpha through list-local texture ids, before those become heap slots.
    std::vector<DirectX::ScratchImage*> albedoImages(lists[0].size(), nullptr);
    for (size_t i = 0; i < lists[0].size(); ++i)
        albedoImages[i] = &lists[0][i].image;
    OmmBuilder::BakeAll(scene.meshes, scene.materials, albedoImages);

    scene.bindlessAlbedoBase = BINDLESS_HEAP_START;
    scene.bindlessNormalBase = scene.bindlessAlbedoBase + (UINT)lists[0].size();
    scene.bindlessRmaBase = scene.bindlessNormalBase + (UINT)lists[1].size();
    scene.totalBindlessTextures = (UINT)(lists[0].size() + lists[1].size() + lists[2].size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        if (scene.materials.albedoTexID[i] >= 0)
            scene.materials.albedoTexID[i] += (int)scene.bindlessAlbedoBase;
        if (scene.materials.normalTexID[i] >= 0)
            scene.materials.normalTexID[i] += (int)scene.bindlessNormalBase;
        if (scene.materials.rmaTexID[i] >= 0)
            scene.materials.rmaTexID[i] += (int)scene.bindlessRmaBase;
    }
    AssetLoader::CreateBindlessTextures(lists[0], scene.bindlessAlbedoBase, L"Albedo", device, cmdList,
                                        scene.bindlessGpuTextures, flush);
    AssetLoader::CreateBindlessTextures(lists[1], scene.bindlessNormalBase, L"Normal", device, cmdList,
                                        scene.bindlessGpuTextures, flush);
    AssetLoader::CreateBindlessTextures(lists[2], scene.bindlessRmaBase, L"RMA", device, cmdList,
                                        scene.bindlessGpuTextures, flush);
    staged = StagedScene{};
}

} // namespace scene_bridge
