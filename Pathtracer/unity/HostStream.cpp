#include "../rdn/stdafx.h"
#include "HostStream.h"
#include "../rdn/DXRHelper.h"
#include "../rdn/Scene/AssetLoader.h"
#include "../rdn/planet/tlas_builder.h"
#include "../src/Components/Vertex.h"
#include <DirectXPackedVector.h>
#include <DirectXTex.h>
#include "../rdn/Lighting/LightTreePacking.h"
#include <sstream>

namespace {
constexpr uint64_t kRetireFrames = 8; // more than the frames a queue can hold
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr uint64_t kAsAlign = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
constexpr uint32_t kLightDense = 0x80000000u; // InstanceProperties.triToLightBase: records follow per primitive


uint64_t AlignUp(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

// Deforming meshes refit every frame; the rest are built once.
D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS BlasFlags(bool deforming) {
    return deforming ? D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                           D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD
                     : D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
}
} // namespace

void HostStream::WriteVertices(BTriVertex* v, uint32_t count, const float* positions, const float* normals,
                               const float* uvs, const DirectX::PackedVector::XMHALF2* packedUvs) {
    using namespace DirectX;
    for (uint32_t i = 0; i < count; ++i) {
        const float* p = positions + 3 * i;
        const float* n = normals + 3 * i;
        v[i].vertex = {p[0], p[1], p[2]};
        XMVECTOR nv = XMVectorSet(n[0], n[1], n[2], 0.0f);
        nv = XMVectorGetX(XMVector3LengthSq(nv)) > 1e-12f ? XMVector3Normalize(nv) : XMVectorSet(0, 1, 0, 0);
        v[i].packedNormal = EncodeNormalOct(nv);
        v[i].texCoord = packedUvs ? packedUvs[i]
                        : uvs     ? PackedVector::XMHALF2(uvs[2 * i], uvs[2 * i + 1])
                                  : PackedVector::XMHALF2(0.0f, 0.0f);
    }
}

HostStream::HostStream(const Config& config) : m_cfg(config) {
    m_instances.resize(m_cfg.instances);
    m_materialLive.resize(m_cfg.materials, 0);
    m_materialCone.resize(m_cfg.materials, {0.0f, 0.0f});
}

// The bound buffers belong to the renderer, which goes first; their mappings end with them.
HostStream::~HostStream() = default;

IHostStream::Reservation HostStream::reservation() const {
    Reservation r;
    r.vertexElems = m_cfg.vertices;
    r.indexElems = m_cfg.indices;
    r.matIdElems = m_cfg.triangles;
    r.instanceSlots = m_cfg.instances;
    r.materials = m_cfg.materials;
    return r;
}

void HostStream::bind(const Binding& b) {
    m_b = b;
    m_vertexVa = b.vertexGlobal->GetGPUVirtualAddress();
    m_indexVa = b.indexGlobal->GetGPUVirtualAddress();
    m_descriptorSize = b.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    void* p = nullptr;
    D3D12_RANGE noRead{0, 0};
    ThrowIfFailed(b.materialIds->Map(0, &noRead, &p));
    m_materialIdsMapped = static_cast<uint8_t*>(p);
    m_vtxAlloc.init(m_cfg.vertices, 64);
    m_idxAlloc.init(m_cfg.indices, 192);
    m_matAlloc.init(m_cfg.triangles, 64);
    m_textures.resize(b.textureCount);
    m_bound = true;
}

int32_t HostStream::Allocate(std::vector<uint32_t>& freeList, uint32_t& next, uint32_t capacity) {
    if (!freeList.empty()) {
        const uint32_t id = freeList.back();
        freeList.pop_back();
        return (int32_t)id;
    }
    return next < capacity ? (int32_t)next++ : -1;
}

void HostStream::Retire(std::function<void()> action) { m_retired.emplace_back(m_frame + kRetireFrames, std::move(action)); }

// ---------------------------------------------------------------------------------------------------------- textures

int32_t HostStream::CreateTexture(const RT_TextureDesc& d) {
    using namespace DirectX;
    if (!m_bound || !d.rgba8 || !d.width || !d.height) {
        m_error = "invalid texture";
        return -1;
    }
    const int32_t id = Allocate(m_freeTextures, m_nextTexture, (uint32_t)m_textures.size());
    if (id < 0) {
        m_error = "out of texture slots";
        return -1;
    }
    const DXGI_FORMAT format = d.kind == RT_TEXTURE_ALBEDO ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : DXGI_FORMAT_R8G8B8A8_UNORM;
    ScratchImage base, mips;
    ThrowIfFailed(base.Initialize2D(format, d.width, d.height, 1, 1));
    const Image* img = base.GetImage(0, 0, 0);
    for (uint32_t y = 0; y < d.height; ++y)
        memcpy(img->pixels + y * img->rowPitch, d.rgba8 + (size_t)y * d.width * 4, (size_t)d.width * 4);
    // The renderer picks mips from ray cones, so every texture needs a full chain.
    ThrowIfFailed(GenerateMipMaps(*img, TEX_FILTER_DEFAULT, 0, mips));
    const UINT mipCount = (UINT)mips.GetMetadata().mipLevels;

    Texture& t = m_textures[id];
    t = Texture{};
    auto desc = CD3DX12_RESOURCE_DESC::Tex2D(format, d.width, d.height, 1, (UINT16)mipCount);
    auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(m_b.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                      nullptr, IID_PPV_ARGS(&t.resource)));
    t.resource->SetName(L"HostTexture");
    t.footprints.resize(mipCount);
    std::vector<UINT> rows(mipCount);
    std::vector<UINT64> rowBytes(mipCount);
    UINT64 total = 0;
    m_b.device->GetCopyableFootprints(&desc, 0, mipCount, 0, t.footprints.data(), rows.data(), rowBytes.data(), &total);
    t.staging = nv_helpers_dx12::CreateBuffer(m_b.device, total, D3D12_RESOURCE_FLAG_NONE,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    uint8_t* dst = nullptr;
    ThrowIfFailed(t.staging->Map(0, nullptr, (void**)&dst));
    for (UINT m = 0; m < mipCount; ++m) {
        const Image* mip = mips.GetImage(m, 0, 0);
        for (UINT y = 0; y < rows[m]; ++y)
            memcpy(dst + t.footprints[m].Offset + (size_t)y * t.footprints[m].Footprint.RowPitch,
                   mip->pixels + (size_t)y * mip->rowPitch, (size_t)rowBytes[m]);
    }
    t.staging->Unmap(0, nullptr);

    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = format;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = mipCount;
    CD3DX12_CPU_DESCRIPTOR_HANDLE handle(m_b.heap->GetCPUDescriptorHandleForHeapStart(), (INT)(m_b.textureBase + id),
                                         m_descriptorSize);
    m_b.device->CreateShaderResourceView(t.resource.Get(), &srv, handle);
    t.live = true;
    m_pendingTextures.push_back((uint32_t)id);
    return id;
}

void HostStream::ReleaseTexture(int32_t id) {
    if (id < 0 || (size_t)id >= m_textures.size() || !m_textures[id].live)
        return;
    m_textures[id].live = false;
    Retire([this, id] {
        m_textures[id] = Texture{};
        m_freeTextures.push_back((uint32_t)id);
    });
}

// --------------------------------------------------------------------------------------------------------- materials

Material HostStream::ToMaterial(const RT_MaterialDesc& d) const {
    auto texture = [&](int32_t id) {
        return id >= 0 && (size_t)id < m_textures.size() && m_textures[id].live ? (int)(m_b.textureBase + id) : -1;
    };
    Material m;
    m.Kd = {d.baseColor[0], d.baseColor[1], d.baseColor[2], d.baseColor[3]};
    m.Ke = {d.emission[0], d.emission[1], d.emission[2]};
    m.Ni = d.ior > 0.0f ? d.ior : 1.5f;
    m.Pr_Pm_Ps_Pc = {std::clamp(d.roughness, 0.0f, 1.0f), std::clamp(d.metallic, 0.0f, 1.0f), 0.0f,
                     std::clamp(d.coat, 0.0f, 1.0f)};
    m.Pcr_aniso_anisor = {std::clamp(d.coatRoughness, 0.0f, 1.0f), 0.0f, 0.0f};
    m.Tf = {0.0f, 0.0f, 0.0f}; // opaque; only glass reads it
    m.albedoTexID = texture(d.albedoTexture);
    m.normalTexID = texture(d.normalTexture);
    m.rmaTexID = texture(d.rmaTexture);
    const DirectX::XMFLOAT2 uvScale = {d.uvScale[0] != 0.0f ? d.uvScale[0] : 1.0f,
                                       d.uvScale[1] != 0.0f ? d.uvScale[1] : 1.0f};
    m.albedoUVScale = m.normalUVScale = m.rmaUVScale = uvScale;
    m.alphaThreshold = m.albedoTexID >= 0 ? std::clamp(d.alphaCutoff, 0.0f, 1.0f) : 1.0f;
    if (d.flags & RT_MATERIAL_THIN_GLASS) { // as the voxel world's glass (mc_materials.cpp)
        m.thinGlass = 1;
        m.Tf = {d.baseColor[0], d.baseColor[1], d.baseColor[2]};
        m.Kd = {d.baseColor[0], d.baseColor[1], d.baseColor[2], 0.0f};
        m.Pr_Pm_Ps_Pc.y = 0.0f;
        m.albedoTexID = -1;
        m.alphaThreshold = 1.0f;
    }
    return m;
}

void HostStream::WriteMaterial(uint32_t slot, const Material& m) {
    m_b.materials->Set(m_b.materialBase + slot, m);
    m_pendingMaterials.push_back(slot);
}

// HDRP's spot attenuation as scale and offset on the cosine.
static DirectX::XMFLOAT2 ConeOf(const RT_MaterialDesc& d) {
    if (!(d.flags & RT_MATERIAL_SPOT_EMISSION))
        return {0.0f, 0.0f};
    const float outer = std::clamp(d.spotCosOuter, 0.0f, 1.0f);
    const float scale = 1.0f / std::max(std::clamp(d.spotCosInner, outer, 1.0f) - outer, 1e-4f);
    return {scale, -outer * scale};
}

void HostStream::ApplyCone(LightTriangle& r, UINT material) const {
    const DirectX::XMFLOAT2 cone = m_materialCone[material - m_b.materialBase];
    r.coneScale = cone.x;
    r.coneOffset = cone.y;
}

int32_t HostStream::CreateMaterial(const RT_MaterialDesc& d) {
    const int32_t id = m_bound ? Allocate(m_freeMaterials, m_nextMaterial, m_cfg.materials) : -1;
    if (id < 0) {
        m_error = "out of material slots";
        return -1;
    }
    m_materialLive[id] = 1;
    m_materialCone[id] = ConeOf(d);
    WriteMaterial((uint32_t)id, ToMaterial(d));
    return id;
}

bool HostStream::UpdateMaterial(int32_t id, const RT_MaterialDesc& d) {
    if (id <= 0 || (uint32_t)id >= m_cfg.materials || !m_materialLive[id]) {
        m_error = "unknown material";
        return false;
    }
    m_materialCone[id] = ConeOf(d);
    WriteMaterial((uint32_t)id, ToMaterial(d));
    // Light records carry their emission: meshes lit by this material rebuild theirs (same sizes, same spans).
    const UINT index = m_b.materialBase + (UINT)id;
    for (uint32_t mesh = 0; mesh < (uint32_t)m_meshes.size(); ++mesh) {
        Mesh& m = m_meshes[mesh];
        if (!m.live || !m.lights || std::find(m.lights->materials.begin(), m.lights->materials.end(), index) ==
                                        m.lights->materials.end())
            continue;
        const size_t nodes = m.lights->blas.nodes.size();
        for (size_t k = 0; k < m.lights->records.size(); ++k)
            if (m.lights->materials[k] == index) {
                m.lights->records[k].emission = m_b.materials->Ke[index];
                ApplyCone(m.lights->records[k], index);
            }
        // Same sizes: the upload refreshes the spans in place and the mesh stays lit meanwhile (lights that flicker
        // update every frame). Otherwise new spans; frames in flight still read the old ones.
        if (!BuildLightTree(*m.lights) || m.lights->blas.nodes.size() != nodes) {
            if (m.lights->allocated) {
                const uint64_t recOff = m.lights->recOff, nodeOff = m.lights->nodeOff, recs = m.lights->records.size();
                Retire([this, recOff, recs, nodeOff, nodes] {
                    m_recAlloc.free(recOff, recs);
                    m_nodeAlloc.free(nodeOff, nodes);
                });
            }
            m.lights->allocated = m.lights->uploaded = false;
        }
        if (std::find(m_pendingLights.begin(), m_pendingLights.end(), mesh) == m_pendingLights.end())
            m_pendingLights.push_back(mesh);
        m_lightSetDirty = true; // the slots' power changed
    }
    return true;
}

// Emission and power from the current materials, then the light BLAS. Records keep their area in cdf.
bool HostStream::BuildLightTree(MeshLights& l) {
    float total = 0.0f;
    for (size_t k = 0; k < l.records.size(); ++k) {
        LightTriangle& r = l.records[k];
        r.weight = r.cdf * std::max(Luminance(r.emission), 1e-6f); // a dimmed light keeps a valid tree
        if (r.coneScale > 0.0f) // a spot's flux: its cone's solid angle instead of a hemisphere's pi
            r.weight *= 2.0f * (1.0f + r.coneOffset / r.coneScale);
        total += r.weight;
    }
    for (LightTriangle& r : l.records) {
        r.triCount = (UINT)l.records.size();
        r.totalWeight = total;
    }
    try {
        lt::LightTreeBuilder::BuildSingleBLAS(l.records, l.blas, 32u);
    } catch (...) {
        l.blas.nodes.clear();
    }
    return !l.blas.nodes.empty() && l.blas.leafTriLocal.size() == l.records.size();
}

void HostStream::ReleaseMaterial(int32_t id) {
    if (id <= 0 || (uint32_t)id >= m_cfg.materials || !m_materialLive[id])
        return;
    m_materialLive[id] = 0;
    Retire([this, id] { m_freeMaterials.push_back((uint32_t)id); });
}

// ------------------------------------------------------------------------------------------------------------ meshes

void HostStream::BuildGeometryDescs(const Mesh& m, D3D12_RAYTRACING_GEOMETRY_DESC out[2], uint32_t& count) const {
    count = 0;
    uint64_t idx = m_b.indexBase + m.idxOff;
    auto geom = [&](uint32_t tris, bool opaque) {
        D3D12_RAYTRACING_GEOMETRY_DESC& g = out[count++];
        g = {};
        g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g.Flags = opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        g.Triangles.VertexBuffer.StartAddress = m_vertexVa;
        g.Triangles.VertexBuffer.StrideInBytes = sizeof(BTriVertex);
        g.Triangles.VertexCount = m_b.combinedVertexCount;
        g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        g.Triangles.IndexBuffer = m_indexVa + idx * sizeof(uint32_t);
        g.Triangles.IndexCount = tris * 3;
        g.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        idx += (uint64_t)tris * 3;
    };
    if (m.opaqueTris)
        geom(m.opaqueTris, true);
    if (m.triCount > m.opaqueTris)
        geom((uint32_t)m.triCount - m.opaqueTris, false);
}

int32_t HostStream::CreateMesh(const RT_MeshDesc& d) {
    using namespace DirectX;
    if (!m_bound || !d.positions || !d.normals || !d.indices || d.indexCount < 3 || !d.vertexCount) {
        m_error = "invalid mesh";
        return -1;
    }
    const uint32_t tris = d.indexCount / 3;
    std::vector<UINT> indices(d.indices, d.indices + (size_t)tris * 3);
    for (UINT i : indices)
        if (i >= d.vertexCount) {
            m_error = "mesh index out of range";
            return -1;
        }
    std::vector<UINT> materials(tris, m_b.materialBase);
    if (d.triangleMaterials)
        for (uint32_t t = 0; t < tris; ++t) {
            const int32_t id = d.triangleMaterials[t];
            materials[t] = m_b.materialBase +
                           (id > 0 && (uint32_t)id < m_cfg.materials && m_materialLive[id] ? (uint32_t)id : 0u);
        }
    MeshSplitResult split = AssetLoader::SplitOpaqueAlpha(indices, materials, *m_b.materials);
    // Emissive triangles first within the opaque and the alpha part: their light records are dense.
    auto emissive = [&](UINT material) { return Luminance(m_b.materials->Ke[material]) > 0.0f; };
    uint32_t litCount[2] = {0, 0};
    for (int part = 0; part < 2; ++part) {
        const uint32_t first = part ? split.opaqueTriCount : 0u, last = part ? tris : split.opaqueTriCount;
        std::vector<uint32_t> order;
        for (uint32_t t = first; t < last; ++t)
            if (emissive(split.reorderedMaterialIDs[t]))
                order.push_back(t);
        litCount[part] = (uint32_t)order.size();
        if (order.empty())
            continue;
        for (uint32_t t = first; t < last; ++t)
            if (!emissive(split.reorderedMaterialIDs[t]))
                order.push_back(t);
        std::vector<UINT> idx(order.size() * 3), mat(order.size());
        for (size_t k = 0; k < order.size(); ++k) {
            for (int c = 0; c < 3; ++c)
                idx[k * 3 + c] = split.reorderedIndices[order[k] * 3 + c];
            mat[k] = split.reorderedMaterialIDs[order[k]];
        }
        std::copy(idx.begin(), idx.end(), split.reorderedIndices.begin() + first * 3);
        std::copy(mat.begin(), mat.end(), split.reorderedMaterialIDs.begin() + first);
    }
    std::shared_ptr<MeshLights> lights;
    if (litCount[0] + litCount[1]) {
        lights = std::make_shared<MeshLights>();
        lights->litOpaque = litCount[0];
        lights->litAlpha = litCount[1];
        float total = 0.0f;
        auto add = [&](uint32_t t) {
            const UINT* v = &split.reorderedIndices[t * 3];
            const float* a = d.positions + 3 * v[0];
            const float* b = d.positions + 3 * v[1];
            const float* c = d.positions + 3 * v[2];
            LightTriangle r{};
            r.x = {a[0], a[1], a[2]};
            r.y = {b[0], b[1], b[2]};
            r.z = {c[0], c[1], c[2]};
            r.emission = m_b.materials->Ke[split.reorderedMaterialIDs[t]];
            ApplyCone(r, split.reorderedMaterialIDs[t]);
            lights->materials.push_back(split.reorderedMaterialIDs[t]);
            const XMVECTOR cr = XMVector3Cross(XMVectorSet(b[0] - a[0], b[1] - a[1], b[2] - a[2], 0),
                                               XMVectorSet(c[0] - a[0], c[1] - a[1], c[2] - a[2], 0));
            r.weight = std::max(0.5f * XMVectorGetX(XMVector3Length(cr)), 1e-10f); // area; BuildLightTree adds power
            lights->records.push_back(r);
        };
        for (uint32_t t = 0; t < litCount[0]; ++t)
            add(t);
        for (uint32_t t = 0; t < litCount[1]; ++t)
            add(split.opaqueTriCount + t);
        for (LightTriangle& r : lights->records)
            r.cdf = r.weight; // area, kept for emission updates
        if (!BuildLightTree(*lights))
            lights.reset(); // they still glow when hit, but do not light the scene
    }

    Mesh m;
    m.vtxCount = d.vertexCount;
    m.triCount = tris;
    m.opaqueTris = split.opaqueTriCount;
    if (!m_vtxAlloc.allocate(m.vtxCount, m.vtxOff)) {
        m_error = "out of host vertex space";
        return -1;
    }
    if (!m_idxAlloc.allocate(m.triCount * 3, m.idxOff)) {
        m_vtxAlloc.free(m.vtxOff, m.vtxCount);
        m_error = "out of host index space";
        return -1;
    }
    if (!m_matAlloc.allocate(m.triCount, m.matOff)) {
        m_vtxAlloc.free(m.vtxOff, m.vtxCount);
        m_idxAlloc.free(m.idxOff, m.triCount * 3);
        m_error = "out of host triangle space";
        return -1;
    }
    const int32_t id = Allocate(m_freeMeshes, m_nextMesh, 0x7FFFFFFF);
    if ((size_t)id >= m_meshes.size())
        m_meshes.resize((size_t)id + 1);

    // Vertices then absolute indices, staged for the copy in record_gpu_work.
    const uint64_t vtxBytes = m.vtxCount * sizeof(BTriVertex), idxBytes = m.triCount * 3 * sizeof(uint32_t);
    m.stagingIndexOffset = AlignUp(vtxBytes, 16);
    m.staging = nv_helpers_dx12::CreateBuffer(m_b.device, m.stagingIndexOffset + idxBytes, D3D12_RESOURCE_FLAG_NONE,
                                              D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    uint8_t* dst = nullptr;
    ThrowIfFailed(m.staging->Map(0, nullptr, (void**)&dst));
    WriteVertices(reinterpret_cast<BTriVertex*>(dst), d.vertexCount, d.positions, d.normals, d.uvs, nullptr);
    m.deforming = (d.flags & RT_MESH_DEFORMING) != 0;
    if (m.deforming) {
        m.uvs.resize(d.vertexCount);
        for (uint32_t i = 0; i < d.vertexCount; ++i)
            m.uvs[i] = reinterpret_cast<BTriVertex*>(dst)[i].texCoord;
    }
    auto* idx = reinterpret_cast<uint32_t*>(dst + m.stagingIndexOffset);
    const uint32_t vtxBase = (uint32_t)(m_b.vertexBase + m.vtxOff);
    for (size_t i = 0; i < split.reorderedIndices.size(); ++i)
        idx[i] = split.reorderedIndices[i] + vtxBase;
    m.staging->Unmap(0, nullptr);
    memcpy(m_materialIdsMapped + (m_b.matIdBase + m.matOff) * sizeof(UINT), split.reorderedMaterialIDs.data(),
           split.reorderedMaterialIDs.size() * sizeof(UINT));

    D3D12_RAYTRACING_GEOMETRY_DESC geoms[2];
    uint32_t geomCount = 0;
    BuildGeometryDescs(m, geoms, geomCount);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
    in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    in.Flags = BlasFlags(m.deforming);
    in.NumDescs = geomCount;
    in.pGeometryDescs = geoms;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    m_b.device->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
    m.blas = nv_helpers_dx12::CreateBuffer(m_b.device, AlignUp(info.ResultDataMaxSizeInBytes, kAsAlign),
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                           nv_helpers_dx12::kDefaultHeapProps);
    m.blas->SetName(L"HostBLAS");
    m.scratchBytes = AlignUp(info.ScratchDataSizeInBytes, kAsAlign);
    m.updateScratchBytes = AlignUp(info.UpdateScratchDataSizeInBytes, kAsAlign);
    m.live = true;
    m.lights = std::move(lights);
    if (m.lights)
        m_pendingLights.push_back((uint32_t)id);
    m_meshes[id] = std::move(m);
    m_pendingMeshes.push_back((uint32_t)id);
    return id;
}

void HostStream::DropMesh(int32_t id) {
    Mesh& m = m_meshes[id];
    const uint64_t vtxOff = m.vtxOff, vtxCount = m.vtxCount, idxOff = m.idxOff, tris = m.triCount, matOff = m.matOff;
    m.live = false;
    std::shared_ptr<MeshLights> lights = m.lights;
    Retire([this, id, vtxOff, vtxCount, idxOff, tris, matOff, lights] {
        if (lights && lights->allocated) {
            m_recAlloc.free(lights->recOff, lights->records.size());
            m_nodeAlloc.free(lights->nodeOff, lights->blas.nodes.size());
        }
        m_vtxAlloc.free(vtxOff, vtxCount);
        m_idxAlloc.free(idxOff, tris * 3);
        m_matAlloc.free(matOff, tris);
        m_meshes[id] = Mesh{};
        m_freeMeshes.push_back((uint32_t)id);
    });
}

bool HostStream::UpdateVertices(int32_t id, const float* positions, const float* normals) {
    if (id < 0 || (size_t)id >= m_meshes.size() || !m_meshes[id].live || !m_meshes[id].deforming || !positions ||
        !normals) {
        m_error = "not a deforming mesh";
        return false;
    }
    Mesh& m = m_meshes[id];
    // Not built yet: the first upload takes the new vertices. Otherwise they wait for the next refit.
    uint8_t* dst = nullptr;
    if (!m.built) {
        ThrowIfFailed(m.staging->Map(0, nullptr, (void**)&dst));
        WriteVertices(reinterpret_cast<BTriVertex*>(dst), (uint32_t)m.vtxCount, positions, normals, nullptr, m.uvs.data());
        m.staging->Unmap(0, nullptr);
        return true;
    }
    if (!m.update) {
        m.update = nv_helpers_dx12::CreateBuffer(m_b.device, m.vtxCount * sizeof(BTriVertex), D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        m_pendingUpdates.push_back((uint32_t)id);
    }
    ThrowIfFailed(m.update->Map(0, nullptr, (void**)&dst));
    WriteVertices(reinterpret_cast<BTriVertex*>(dst), (uint32_t)m.vtxCount, positions, normals, nullptr, m.uvs.data());
    m.update->Unmap(0, nullptr);
    return true;
}

void HostStream::ReleaseMesh(int32_t id) {
    if (id < 0 || (size_t)id >= m_meshes.size() || !m_meshes[id].live || m_meshes[id].released)
        return;
    m_meshes[id].released = true;
    if (!m_meshes[id].users)
        DropMesh(id);
}

// --------------------------------------------------------------------------------------------------------- instances

int32_t HostStream::CreateInstance(int32_t mesh, const float xform[12]) {
    if (mesh < 0 || (size_t)mesh >= m_meshes.size() || !m_meshes[mesh].live || m_meshes[mesh].released) {
        m_error = "invalid mesh";
        return -1;
    }
    const int32_t id = Allocate(m_freeInstances, m_nextInstance, m_cfg.instances);
    if (id < 0) {
        m_error = "out of instance slots";
        return -1;
    }
    Instance& i = m_instances[id];
    i = Instance{};
    i.mesh = mesh;
    i.live = true;
    memcpy(i.xform, xform, sizeof(i.xform));
    ++m_meshes[mesh].users;
    m_instancesChanged = true;
    return id;
}

void HostStream::ReleaseInstance(int32_t id) {
    if (id < 0 || (uint32_t)id >= m_cfg.instances || !m_instances[id].live)
        return;
    Instance& i = m_instances[id];
    ExcludeLightSlot(i);
    i.live = false;
    m_instancesChanged = true;
    Mesh& m = m_meshes[i.mesh];
    if (--m.users == 0 && m.released)
        DropMesh(i.mesh);
    Retire([this, id] { m_freeInstances.push_back((uint32_t)id); });
}

void HostStream::SetTransform(int32_t id, const float xform[12]) {
    if (id >= 0 && (uint32_t)id < m_cfg.instances && m_instances[id].live)
        memcpy(m_instances[id].xform, xform, sizeof(m_instances[id].xform));
}

void HostStream::SetFlags(int32_t id, uint32_t flags) {
    if (id >= 0 && (uint32_t)id < m_cfg.instances && m_instances[id].live && m_instances[id].flags != flags) {
        m_instances[id].flags = flags;
        m_instancesChanged = true;
    }
}

void HostStream::SetVisible(int32_t id, bool visible) {
    if (id >= 0 && (uint32_t)id < m_cfg.instances && m_instances[id].live && m_instances[id].visible != visible) {
        m_instances[id].visible = visible;
        if (!visible)
            ExcludeLightSlot(m_instances[id]);
        m_instancesChanged = true;
    }
}

// ---------------------------------------------------------------------------------------------------- emitter sets

namespace {
constexpr uint32_t kOctaVerts = 6, kOctaTris = 8;
const float kOctaAxes[kOctaVerts][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
const uint32_t kOctaFaces[kOctaTris][3] = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
                                           {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}}; // counter-clockwise outside
// An octahedron of circumradius r shows sqrt(3) r^2 on average, a sphere pi r^2: its radiance makes up the difference.
constexpr float kSphereRadiance = 1.8137994f;

// An emitter set's light tree: a fixed four-way tree over emitter slots, each slot a node of two four-face groups.
// Emitters fill the slots in Morton order, so updates keep the topology, leaf order and trails and refit the rest.
void BuildEmitterTopology(uint32_t capacity, lt::LightTreeBuilder::SingleBLAS& out) {
    struct Node {
        uint32_t firstChild = 0, childCount = 0, triFirst = 0, triCount = 0;
    };
    std::vector<Node> nodes(1);
    std::vector<uint32_t> leaves;
    std::vector<lt::LightTreeTrail> trails((size_t)capacity * kOctaTris, 0);
    auto children = [&](uint32_t node, uint32_t count) {
        nodes[node].firstChild = (uint32_t)nodes.size();
        nodes[node].childCount = count;
        nodes.resize(nodes.size() + count);
        return nodes[node].firstChild;
    };
    std::function<void(uint32_t, uint32_t, uint32_t, lt::LightTreeTrail, uint32_t)> build =
        [&](uint32_t node, uint32_t b, uint32_t e, lt::LightTreeTrail trail, uint32_t depth) {
            if (e - b == 1) {
                const uint32_t groups = children(node, 2);
                for (uint32_t g = 0; g < 2; ++g) {
                    const uint32_t faces = children(groups + g, 4);
                    nodes[groups + g].triFirst = (uint32_t)leaves.size();
                    nodes[groups + g].triCount = 4;
                    for (uint32_t f = 0; f < 4; ++f) {
                        const uint32_t tri = b * kOctaTris + g * 4 + f;
                        nodes[faces + f].triFirst = (uint32_t)leaves.size();
                        nodes[faces + f].triCount = 1;
                        leaves.push_back(tri);
                        trails[tri] = lt::AppendLightTreeTrail(lt::AppendLightTreeTrail(trail, g, depth), f, depth + 1);
                    }
                }
            } else {
                const uint32_t count = std::min(4u, e - b), first = children(node, count);
                for (uint32_t c = 0; c < count; ++c)
                    build(first + c, b + (e - b) * c / count, b + (e - b) * (c + 1) / count,
                          lt::AppendLightTreeTrail(trail, c, depth), depth + 1);
            }
            nodes[node].triFirst = nodes[nodes[node].firstChild].triFirst;
            nodes[node].triCount = (e - b) * kOctaTris;
        };
    build(0, 0, capacity, 0, 0);
    // Breadth-first, as BuildSingleBLAS stores its trees.
    std::vector<uint32_t> order{0}, remap(nodes.size(), 0);
    for (size_t q = 0; q < order.size(); ++q)
        for (uint32_t c = 0; c < nodes[order[q]].childCount; ++c) {
            remap[nodes[order[q]].firstChild + c] = (uint32_t)order.size();
            order.push_back(nodes[order[q]].firstChild + c);
        }
    out.nodes.assign(nodes.size(), lt::LightBLASNodeGpu{});
    for (size_t k = 0; k < order.size(); ++k) {
        const Node& n = nodes[order[k]];
        lt::LightBLASNodeGpu& g = out.nodes[k];
        g.firstChild = n.childCount ? remap[n.firstChild] : 0xFFFFFFFFu;
        g.childCount = n.childCount;
        g.triFirst = n.triFirst;
        g.triCount = n.triCount;
    }
    out.leafTriLocal = std::move(leaves);
    out.trails = std::move(trails);
}

// Bounds, power and orientation from the records, leaves up (children follow their parents breadth-first).
void RefitEmitterTree(std::vector<LightTriangle>& records, lt::LightTreeBuilder::SingleBLAS& tree) {
    float total = 0.0f;
    for (LightTriangle& r : records) {
        r.weight = r.cdf * std::max(Luminance(r.emission), 1e-6f);
        total += r.weight;
    }
    for (LightTriangle& r : records) {
        r.triCount = (UINT)records.size();
        r.totalWeight = total;
    }
    std::vector<lt::Cone> cones(tree.nodes.size());
    for (size_t k = tree.nodes.size(); k-- > 0;) {
        lt::LightBLASNodeGpu& g = tree.nodes[k];
        lt::Cone& cone = cones[k];
        if (!g.childCount) {
            const LightTriangle& r = records[tree.leafTriLocal[g.triFirst]];
            g.bmin = lt::min3(r.x, lt::min3(r.y, r.z));
            g.bmax = lt::max3(r.x, lt::max3(r.y, r.z));
            g.power = r.weight;
            const DirectX::XMFLOAT3 n = lt::cross3(lt::sub3(r.y, r.x), lt::sub3(r.z, r.x));
            if (lt::length3(n) < 1e-12f) {
                cone.axis = {0.0f, 0.0f, 1.0f};
                cone.theta_o = lt::LT_PI;
            } else {
                cone.axis = lt::normalize3(n);
                cone.theta_o = 0.0f;
            }
        } else {
            const lt::LightBLASNodeGpu& c0 = tree.nodes[g.firstChild];
            g.bmin = c0.bmin;
            g.bmax = c0.bmax;
            g.power = c0.power;
            cone = cones[g.firstChild];
            for (uint32_t c = 1; c < g.childCount; ++c) {
                const lt::LightBLASNodeGpu& child = tree.nodes[g.firstChild + c];
                g.bmin = lt::min3(g.bmin, child.bmin);
                g.bmax = lt::max3(g.bmax, child.bmax);
                g.power += child.power;
                cone = lt::coneUnion(cone, cones[g.firstChild + c]);
            }
        }
        g.axis = cone.axis;
        g.cosTheta_o = std::cos(std::clamp(cone.theta_o, 0.0f, lt::LT_PI));
        g.sinTheta_o = std::sqrt(std::max(0.0f, 1.0f - g.cosTheta_o * g.cosTheta_o));
    }
}

// 30-bit Morton code of a point in [0,1)^3.
uint32_t Morton3(float x, float y, float z) {
    auto spread = [](float v) {
        uint32_t b = (uint32_t)std::clamp(v * 1024.0f, 0.0f, 1023.0f);
        b = (b | (b << 16)) & 0x030000FFu;
        b = (b | (b << 8)) & 0x0300F00Fu;
        b = (b | (b << 4)) & 0x030C30C3u;
        return (b | (b << 2)) & 0x09249249u;
    };
    return spread(x) | (spread(y) << 1) | (spread(z) << 2);
}
} // namespace

int32_t HostStream::CreateEmitterSet(uint32_t capacity) {
    if (!m_bound || capacity == 0 || capacity > (1u << 16)) {
        m_error = "invalid emitter set capacity";
        return -1;
    }
    RT_MaterialDesc md{};
    md.baseColor[3] = 1.0f;
    md.emission[0] = md.emission[1] = md.emission[2] = 1.0f; // marks every triangle emissive; records carry the rest
    md.roughness = 1.0f;
    md.ior = 1.5f;
    md.alphaCutoff = 1.0f;
    md.albedoTexture = md.normalTexture = md.rmaTexture = -1;
    md.uvScale[0] = md.uvScale[1] = 1.0f;
    EmitterSet s;
    s.capacity = capacity;
    s.material = CreateMaterial(md);
    if (s.material < 0)
        return -1;
    // Unit octahedra side by side, so the first light tree is well formed; the set stays hidden until it has emitters.
    s.positions.resize((size_t)capacity * kOctaVerts * 3);
    s.normals.resize(s.positions.size());
    std::vector<uint32_t> indices((size_t)capacity * kOctaTris * 3);
    std::vector<int32_t> materials((size_t)capacity * kOctaTris, s.material);
    std::vector<float> uvs((size_t)capacity * kOctaVerts * 2, 0.0f);
    for (uint32_t e = 0; e < capacity; ++e) {
        for (uint32_t v = 0; v < kOctaVerts; ++v)
            for (int k = 0; k < 3; ++k) {
                s.positions[(e * kOctaVerts + v) * 3 + k] = kOctaAxes[v][k] + (k == 0 ? 3.0f * (float)e : 0.0f);
                s.normals[(e * kOctaVerts + v) * 3 + k] = kOctaAxes[v][k];
            }
        for (uint32_t t = 0; t < kOctaTris; ++t)
            for (int c = 0; c < 3; ++c)
                indices[(e * kOctaTris + t) * 3 + c] = e * kOctaVerts + kOctaFaces[t][c];
    }
    RT_MeshDesc d{s.positions.data(), s.normals.data(), uvs.data(), capacity * kOctaVerts, indices.data(),
                  (uint32_t)indices.size(), materials.data(), RT_MESH_DEFORMING};
    s.mesh = CreateMesh(d);
    if (s.mesh >= 0 && (!m_meshes[s.mesh].lights || m_meshes[s.mesh].lights->records.size() != indices.size() / 3)) {
        ReleaseMesh(s.mesh);
        s.mesh = -1;
        m_error = "emitter set without light records";
    }
    if (s.mesh < 0) {
        ReleaseMaterial(s.material);
        return -1;
    }
    m_meshes[s.mesh].rebuild = true;
    MeshLights& l = *m_meshes[s.mesh].lights; // not uploaded yet: the fixed tree replaces the built one
    BuildEmitterTopology(capacity, l.blas);
    RefitEmitterTree(l.records, l.blas);
    s.order.resize(capacity);
    const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    s.instance = CreateInstance(s.mesh, identity);
    ReleaseMesh(s.mesh); // lives with its instance
    if (s.instance < 0) {
        ReleaseMaterial(s.material);
        return -1;
    }
    SetFlags(s.instance, RT_INSTANCE_EMITTER);
    SetVisible(s.instance, false);
    uint32_t id;
    if (!m_freeEmitterSets.empty()) {
        id = m_freeEmitterSets.back();
        m_freeEmitterSets.pop_back();
    } else {
        id = (uint32_t)m_emitterSets.size();
        m_emitterSets.emplace_back();
    }
    m_emitterSets[id] = std::move(s);
    return (int32_t)id;
}

bool HostStream::UpdateEmitterSet(int32_t id, const RT_Emitter* emitters, uint32_t count) {
    using namespace DirectX;
    if (id < 0 || (size_t)id >= m_emitterSets.size() || m_emitterSets[id].instance < 0 || (count && !emitters)) {
        m_error = "unknown emitter set";
        return false;
    }
    EmitterSet& s = m_emitterSets[id];
    if (count > s.capacity) {
        m_error = "more emitters than the set holds";
        return false;
    }
    if ((count > 0) != s.visible) {
        s.visible = count > 0;
        SetVisible(s.instance, s.visible);
    }
    if (!count)
        return true;
    // Slots in Morton order; unused octahedra collapse onto the last emitter: no area, no power, no extent in the tree.
    XMFLOAT3 lo = {1e30f, 1e30f, 1e30f}, hi = {-1e30f, -1e30f, -1e30f};
    for (uint32_t e = 0; e < count; ++e) {
        const XMFLOAT3 p = {emitters[e].position[0], emitters[e].position[1], emitters[e].position[2]};
        lo = lt::min3(lo, p);
        hi = lt::max3(hi, p);
    }
    const XMFLOAT3 scale = {1.0f / std::max(hi.x - lo.x, 1e-6f), 1.0f / std::max(hi.y - lo.y, 1e-6f),
                            1.0f / std::max(hi.z - lo.z, 1e-6f)};
    for (uint32_t e = 0; e < count; ++e) {
        const float* p = emitters[e].position;
        s.order[e] = (uint64_t)Morton3((p[0] - lo.x) * scale.x, (p[1] - lo.y) * scale.y, (p[2] - lo.z) * scale.z) << 32 | e;
    }
    std::sort(s.order.begin(), s.order.begin() + count);
    MeshLights& l = *m_meshes[s.mesh].lights;
    for (uint32_t e = 0; e < s.capacity; ++e) {
        const RT_Emitter& src = emitters[(uint32_t)s.order[std::min(e, count - 1)]];
        const float r = e < count ? std::max(src.radius, 1e-3f) : 0.0f;
        float* p = &s.positions[(size_t)e * kOctaVerts * 3];
        for (uint32_t v = 0; v < kOctaVerts; ++v)
            for (int k = 0; k < 3; ++k)
                p[v * 3 + k] = src.position[k] + kOctaAxes[v][k] * r;
        const XMFLOAT3 radiance = e < count ? XMFLOAT3{src.radiance[0] * kSphereRadiance, src.radiance[1] * kSphereRadiance,
                                                       src.radiance[2] * kSphereRadiance}
                                            : XMFLOAT3{0.0f, 0.0f, 0.0f};
        const float area = std::max(0.8660254f * r * r, 1e-10f); // an equilateral face with edge r sqrt(2)
        for (uint32_t t = 0; t < kOctaTris; ++t) {
            LightTriangle& rec = l.records[(size_t)e * kOctaTris + t];
            const float* a = p + kOctaFaces[t][0] * 3;
            const float* b = p + kOctaFaces[t][1] * 3;
            const float* c = p + kOctaFaces[t][2] * 3;
            rec.x = {a[0], a[1], a[2]};
            rec.y = {b[0], b[1], b[2]};
            rec.z = {c[0], c[1], c[2]};
            rec.emission = radiance;
            rec.cdf = area;
        }
    }
    if (!UpdateVertices(s.mesh, s.positions.data(), s.normals.data()))
        return false;
    RefitEmitterTree(l.records, l.blas);
    // Same spans and sizes: the upload refreshes them in place and the set stays lit meanwhile.
    if (std::find(m_pendingLights.begin(), m_pendingLights.end(), (uint32_t)s.mesh) == m_pendingLights.end())
        m_pendingLights.push_back((uint32_t)s.mesh);
    m_lightSetDirty = true; // the slot's bounds and power changed
    return true;
}

void HostStream::ReleaseEmitterSet(int32_t id) {
    if (id < 0 || (size_t)id >= m_emitterSets.size() || m_emitterSets[id].instance < 0)
        return;
    EmitterSet& s = m_emitterSets[id];
    ReleaseInstance(s.instance);
    ReleaseMaterial(s.material);
    s = EmitterSet{};
    m_freeEmitterSets.push_back((uint32_t)id);
}

void HostStream::record_gpu_work(ID3D12GraphicsCommandList*, ID3D12GraphicsCommandList4* cl) {
    if (!m_bound)
        return;
    ++m_frame;
    while (!m_retired.empty() && m_retired.front().first <= m_frame) {
        m_retired.front().second();
        m_retired.pop_front();
    }

    for (size_t k = 0; k < m_retiredLightSlots.size();) {
        const RetiredSlot& r = m_retiredLightSlots[k];
        if (r.frame <= m_frame && (!m_lightTlasHasHost || r.excludedAt <= m_lightLive)) {
            m_lightSlots[r.slot] = LightSlot{};
            m_freeLightSlots.push_back(r.slot);
            m_retiredLightSlots[k] = m_retiredLightSlots.back();
            m_retiredLightSlots.pop_back();
        } else
            ++k;
    }
    RecordLightUploads(cl);

    // Material writes wait for this point: their textures upload below, before the frame that reads them.
    if (!m_pendingMaterials.empty()) {
        uint8_t* packed = nullptr;
        D3D12_RANGE noRead{0, 0};
        ThrowIfFailed(m_b.materialBuffer->Map(0, &noRead, (void**)&packed));
        for (uint32_t slot : m_pendingMaterials) {
            const uint32_t index = m_b.materialBase + slot;
            uint32_t words[MaterialPack::kMatPackedU32];
            MaterialPack::PackOne(m_b.materials->Get(index), words);
            memcpy(packed + (size_t)index * sizeof(words), words, sizeof(words));
        }
        m_b.materialBuffer->Unmap(0, nullptr);
        m_pendingMaterials.clear();
    }

    std::vector<D3D12_RESOURCE_BARRIER> barriers;
    for (uint32_t id : m_pendingTextures) {
        Texture& t = m_textures[id];
        if (!t.live || !t.resource)
            continue;
        for (UINT m = 0; m < (UINT)t.footprints.size(); ++m) {
            CD3DX12_TEXTURE_COPY_LOCATION dst(t.resource.Get(), m), src(t.staging.Get(), t.footprints[m]);
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(t.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
        Retire([staging = std::move(t.staging)] {});
        t.uploaded = true;
    }
    m_pendingTextures.clear();
    if (!barriers.empty())
        cl->ResourceBarrier((UINT)barriers.size(), barriers.data());

    std::vector<uint32_t> meshes, updates;
    uint64_t scratchBytes = 0;
    for (uint32_t id : m_pendingMeshes)
        if (m_meshes[id].live) {
            meshes.push_back(id);
            scratchBytes += m_meshes[id].scratchBytes;
        }
    m_pendingMeshes.clear();
    for (uint32_t id : m_pendingUpdates) {
        const Mesh& m = m_meshes[id];
        if (m.live && m.built && m.update) {
            updates.push_back(id);
            scratchBytes += m.rebuild ? m.scratchBytes : m.updateScratchBytes;
        }
    }
    m_pendingUpdates.clear();
    if (meshes.empty() && updates.empty())
        return;

    const D3D12_RESOURCE_BARRIER toCopy[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_b.vertexGlobal, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                             D3D12_RESOURCE_STATE_COPY_DEST),
        CD3DX12_RESOURCE_BARRIER::Transition(m_b.indexGlobal, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                             D3D12_RESOURCE_STATE_COPY_DEST),
    };
    cl->ResourceBarrier(2, toCopy);
    for (uint32_t id : meshes) {
        const Mesh& m = m_meshes[id];
        cl->CopyBufferRegion(m_b.vertexGlobal, (m_b.vertexBase + m.vtxOff) * sizeof(BTriVertex), m.staging.Get(), 0,
                             m.vtxCount * sizeof(BTriVertex));
        cl->CopyBufferRegion(m_b.indexGlobal, (m_b.indexBase + m.idxOff) * sizeof(uint32_t), m.staging.Get(),
                             m.stagingIndexOffset, m.triCount * 3 * sizeof(uint32_t));
    }
    for (uint32_t id : updates) {
        const Mesh& m = m_meshes[id];
        cl->CopyBufferRegion(m_b.vertexGlobal, (m_b.vertexBase + m.vtxOff) * sizeof(BTriVertex), m.update.Get(), 0,
                             m.vtxCount * sizeof(BTriVertex));
    }
    const D3D12_RESOURCE_BARRIER toRead[2] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_b.vertexGlobal, D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_b.indexGlobal, D3D12_RESOURCE_STATE_COPY_DEST,
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    cl->ResourceBarrier(2, toRead);

    ComPtr<ID3D12Resource> scratch = nv_helpers_dx12::CreateBuffer(
        m_b.device, scratchBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nv_helpers_dx12::kDefaultHeapProps);
    uint64_t scratchUsed = 0;
    for (uint32_t id : meshes) {
        Mesh& m = m_meshes[id];
        D3D12_RAYTRACING_GEOMETRY_DESC geoms[2];
        uint32_t geomCount = 0;
        BuildGeometryDescs(m, geoms, geomCount);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
        desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        desc.Inputs.Flags = BlasFlags(m.deforming);
        desc.Inputs.NumDescs = geomCount;
        desc.Inputs.pGeometryDescs = geoms;
        desc.DestAccelerationStructureData = m.blas->GetGPUVirtualAddress();
        desc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress() + scratchUsed;
        cl->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
        scratchUsed += m.scratchBytes;
        Retire([staging = std::move(m.staging)] {});
        m.built = true;
    }
    for (uint32_t id : updates) {
        Mesh& m = m_meshes[id];
        D3D12_RAYTRACING_GEOMETRY_DESC geoms[2];
        uint32_t geomCount = 0;
        BuildGeometryDescs(m, geoms, geomCount);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
        desc.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        desc.Inputs.Flags = BlasFlags(true);
        if (!m.rebuild) {
            desc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            desc.SourceAccelerationStructureData = m.blas->GetGPUVirtualAddress();
        }
        desc.Inputs.NumDescs = geomCount;
        desc.Inputs.pGeometryDescs = geoms;
        desc.DestAccelerationStructureData = m.blas->GetGPUVirtualAddress();
        desc.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress() + scratchUsed;
        cl->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
        scratchUsed += m.rebuild ? m.scratchBytes : m.updateScratchBytes;
        Retire([update = std::move(m.update)] {});
    }
    // The TLAS build on this list reads the new BLASes.
    const D3D12_RESOURCE_BARRIER uav = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    cl->ResourceBarrier(1, &uav);
    Retire([scratch] {});
}

void HostStream::append_instances(planet::TlasBuilder& tlas, InstanceProperties* props, const planet::DVec3& origin,
                                  uint32_t hitGroup, bool& forceRebuild, bool&) {
    using namespace DirectX;
    if (!m_bound)
        return;
    const bool originChanged = origin.x != m_lastOrigin.x || origin.y != m_lastOrigin.y || origin.z != m_lastOrigin.z;
    m_lastOrigin = origin;
    if (m_instancesChanged || originChanged)
        forceRebuild = true;
    m_instancesChanged = false;

    // Scene space: world minus the origin, as the TLAS and the instance records use it.
    auto relative = [&](const float in[12], float out[12]) {
        memcpy(out, in, 12 * sizeof(float));
        out[3] = (float)(in[3] - origin.x);
        out[7] = (float)(in[7] - origin.y);
        out[11] = (float)(in[11] - origin.z);
    };
    auto toMatrix = [](const float m[12]) {
        return XMMATRIX(m[0], m[4], m[8], 0.0f, m[1], m[5], m[9], 0.0f, m[2], m[6], m[10], 0.0f, m[3], m[7], m[11],
                        1.0f);
    };

    for (uint32_t id = 0; id < m_nextInstance; ++id) {
        Instance& i = m_instances[id];
        if (!i.live || !i.visible)
            continue;
        const Mesh& mesh = m_meshes[i.mesh];
        if (!mesh.built)
            continue;
        if (!i.seen) {
            memcpy(i.lastFrame, i.xform, sizeof(i.xform)); // no motion on the first frame
            i.seen = true;
        }
        float rel[12];
        relative(i.xform, rel);
        const uint32_t instId = m_b.propsBase + id;
        const bool alpha = mesh.triCount > mesh.opaqueTris;
        const D3D12_RAYTRACING_INSTANCE_FLAGS flags =
            alpha ? D3D12_RAYTRACING_INSTANCE_FLAG_NONE : D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE;
        const uint32_t hg = mesh.opaqueTris == 0 ? hitGroup + 1u : hitGroup;
        const uint8_t mask = (i.flags & RT_INSTANCE_EMITTER) ? INSTANCE_MASK_EMITTER : 0xFF;
        tlas.add_instance(mesh.blas->GetGPUVirtualAddress(), rel, instId, hg, flags, mask);

        const bool moved = memcmp(i.xform, i.lastFrame, sizeof(i.xform)) != 0;
        const bool prevStale = memcmp(i.writtenPrev, i.lastFrame, sizeof(i.xform)) != 0;
        const bool lit = mesh.lights && mesh.lights->uploaded;
        if (lit && i.lightSlot == kNone) {
            uint32_t slot = kNone;
            if (!m_freeLightSlots.empty()) {
                slot = m_freeLightSlots.back();
                m_freeLightSlots.pop_back();
            } else if (m_lightSlots.size() < m_cfg.lightSlots) {
                slot = (uint32_t)m_lightSlots.size();
                m_lightSlots.emplace_back();
            }
            if (slot != kNone) {
                m_lightSlots[slot] = LightSlot{(int32_t)id, m_lightVersion + 1, 0xFFFFFFFFu, true};
                i.lightSlot = slot;
                m_lightSetDirty = true;
            }
        }
        if (lit && moved)
            m_lightSetDirty = true; // the tree's bounds follow the instance
        const uint32_t lightRecord = lit ? kLightDense | (m_lights.recordBase + (uint32_t)mesh.lights->recOff) : kNone;
        const uint32_t lightSlot =
            i.lightSlot != kNone && LightSlotLive(i.lightSlot) ? m_lights.slotBase + i.lightSlot : kNone;
        if (props && (!i.propsValid || originChanged || moved || prevStale || lightRecord != i.writtenLightRecord ||
                      lightSlot != i.writtenLightSlot)) {
            float prevRel[12];
            relative(i.lastFrame, prevRel);
            const XMMATRIX M = toMatrix(rel);
            XMVECTOR det;
            XMMATRIX upper = M;
            upper.r[3] = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
            InstanceProperties& p = props[instId];
            p.objectToWorld = MakeFloat3x4(M);
            p.objectToWorldInverse = MakeFloat3x4(XMMatrixInverse(&det, M));
            p.objectToWorldNormal = MakeFloat3x4(XMMatrixTranspose(XMMatrixInverse(&det, upper)));
            p.prevObjectToWorld = MakeFloat3x4(toMatrix(prevRel));
            p.indexBase = (UINT)(m_b.indexBase + mesh.idxOff);
            p.vertexBase = (UINT)(m_b.vertexBase + mesh.vtxOff);
            p.materialBase = (UINT)(m_b.matIdBase + mesh.matOff);
            p.opaqueTriCount = mesh.opaqueTris;
            p.triToLightBase = lightRecord;
            p._pad[0] = lit ? mesh.lights->litOpaque : 0u;
            p._pad[1] = lit ? mesh.lights->litAlpha : 0u;
            p.lightSlot = lightSlot;
            i.writtenLightRecord = lightRecord;
            i.writtenLightSlot = lightSlot;
            memcpy(i.writtenPrev, i.lastFrame, sizeof(i.xform));
            i.propsValid = true;
        }
        memcpy(i.lastFrame, i.xform, sizeof(i.xform));
    }
    if (m_lightSetDirty) {
        ++m_lightVersion;
        m_lightSetDirty = false;
    }
}

// ------------------------------------------------------------------------------------------------------------ lights

void HostStream::bind_lights(const DynamicLightBinding& b) {
    m_lights = b;
    m_lightsBound = b.records != nullptr;
    m_recAlloc.init(b.recordCapacity, 64);
    m_nodeAlloc.init(b.nodeCapacity, 64);
    m_lightTlasHasHost = false;
    m_lightSetDirty = true;
    // New buffers: every mesh's light data goes up again, every instance record is rewritten.
    m_pendingLights.clear();
    for (uint32_t id = 0; id < (uint32_t)m_meshes.size(); ++id)
        if (m_meshes[id].live && m_meshes[id].lights) {
            m_meshes[id].lights->allocated = m_meshes[id].lights->uploaded = false;
            m_pendingLights.push_back(id);
        }
    for (Instance& i : m_instances)
        i.propsValid = false;
}

void HostStream::set_light_slot_base(uint32_t base) {
    if (m_lights.slotBase == base)
        return;
    m_lights.slotBase = base;
    m_lightTlasHasHost = false;
    m_lightSetDirty = true;
    for (Instance& i : m_instances)
        i.propsValid = false;
}

void HostStream::on_light_tlas_published(uint32_t version, bool hasLeaves) {
    if (hasLeaves)
        m_lightLive = std::max(m_lightLive, version);
    m_lightTlasHasHost = hasLeaves;
}

bool HostStream::LightSlotLive(uint32_t slot) const {
    if (!m_lightTlasHasHost || slot >= m_lightSlots.size())
        return false;
    const LightSlot& s = m_lightSlots[slot];
    return s.includedAt <= m_lightLive && m_lightLive < s.excludedAt;
}

void HostStream::ExcludeLightSlot(Instance& i) {
    if (i.lightSlot == kNone)
        return;
    LightSlot& s = m_lightSlots[i.lightSlot];
    s.active = false;
    s.excludedAt = m_lightVersion + 1;
    m_retiredLightSlots.push_back({i.lightSlot, m_frame + kRetireFrames, s.excludedAt});
    i.lightSlot = kNone;
    m_lightSetDirty = true;
}

void HostStream::light_snapshot(const planet::DVec3& origin, std::vector<lt::TLASExtraLeaf>& out, uint32_t& version,
                                uint32_t& slotCount) const {
    using namespace DirectX;
    out.clear();
    version = m_lightVersion;
    slotCount = m_lights.slotBase + (uint32_t)m_lightSlots.size();
    if (!m_lightsBound)
        return;
    for (uint32_t s = 0; s < (uint32_t)m_lightSlots.size(); ++s) {
        const LightSlot& slot = m_lightSlots[s];
        if (!slot.active || slot.includedAt > m_lightVersion)
            continue;
        const Instance& i = m_instances[slot.instance];
        const Mesh& mesh = m_meshes[i.mesh];
        if (!mesh.lights || !mesh.lights->uploaded)
            continue;
        const float* x = i.xform;
        const XMMATRIX world(x[0], x[4], x[8], 0.0f, x[1], x[5], x[9], 0.0f, x[2], x[6], x[10], 0.0f,
                             (float)(x[3] - origin.x), (float)(x[7] - origin.y), (float)(x[11] - origin.z), 1.0f);
        const lt::LightBLASNodeGpu& root = mesh.lights->blas.nodes[0];
        lt::TLASExtraLeaf leaf;
        leaf.slot = m_lights.slotBase + s;
        XMVECTOR mn = XMVectorReplicate(1e30f), mx = XMVectorReplicate(-1e30f);
        for (int k = 0; k < 8; ++k) {
            const XMVECTOR corner = XMVectorSet((k & 1) ? root.bmax.x : root.bmin.x, (k & 2) ? root.bmax.y : root.bmin.y,
                                                (k & 4) ? root.bmax.z : root.bmin.z, 1.0f);
            const XMVECTOR p = XMVector3TransformCoord(corner, world);
            mn = XMVectorMin(mn, p);
            mx = XMVectorMax(mx, p);
        }
        XMStoreFloat3(&leaf.aabb.mn, mn);
        XMStoreFloat3(&leaf.aabb.mx, mx);
        // Emitted power scales with area: |det|^(2/3) for a uniform scale.
        const float areaScale = std::pow(std::abs(XMVectorGetX(XMMatrixDeterminant(world))), 2.0f / 3.0f);
        leaf.power = root.power * areaScale;
        XMStoreFloat3(&leaf.cone.axis,
                      XMVector3Normalize(XMVector3TransformNormal(XMVectorSet(root.axis.x, root.axis.y, root.axis.z, 0), world)));
        leaf.cone.theta_o = std::acos(std::clamp(root.cosTheta_o, -1.0f, 1.0f));
        leaf.cone.theta_e = lt::LT_HALF_PI;
        leaf.primCount = (uint32_t)mesh.lights->records.size();
        leaf.sumPower = leaf.power;
        leaf.sumPowerSq = leaf.power * leaf.power;
        XMFLOAT4X4 worldF;
        XMStoreFloat4x4(&worldF, world);
        leaf.record = lt::makeSlotRecord(m_b.propsBase + (uint32_t)slot.instance,
                                         m_lights.nodeBase + (uint32_t)mesh.lights->nodeOff, worldF);
        out.push_back(leaf);
    }
}

// Light data of new meshes into the bound regions: records, light BLAS nodes, leaf indices and trails.
void HostStream::RecordLightUploads(ID3D12GraphicsCommandList4* cl) {
    if (!m_lightsBound || m_pendingLights.empty())
        return;
    std::vector<uint32_t> ready, waiting;
    for (uint32_t id : m_pendingLights) {
        Mesh& m = m_meshes[id];
        if (!m.live || !m.lights)
            continue;
        MeshLights& l = *m.lights;
        if (!l.allocated) {
            if (!m_recAlloc.allocate(l.records.size(), l.recOff))
                l.recOff = ~0ull;
            else if (!m_nodeAlloc.allocate(l.blas.nodes.size(), l.nodeOff)) {
                m_recAlloc.free(l.recOff, l.records.size());
                l.recOff = ~0ull;
            }
            if (l.recOff == ~0ull) {
                if (!m_warnedLightSpace)
                    std::cout << "Host light region full; further emissive meshes glow but do not light" << std::endl;
                m_warnedLightSpace = true;
                continue;
            }
            l.allocated = true;
        }
        ready.push_back(id);
    }
    m_pendingLights.clear();
    if (ready.empty())
        return;

    struct Part {
        uint64_t recBytes, nodeOffset, nodeBytes, leafOffset, leafBytes, trailOffset, trailBytes, size;
    };
    std::vector<Part> parts;
    uint64_t total = 0;
    for (uint32_t id : ready) {
        const MeshLights& l = *m_meshes[id].lights;
        Part p{};
        p.recBytes = l.records.size() * sizeof(LightTriangle);
        p.nodeOffset = AlignUp(p.recBytes, 16);
        p.nodeBytes = l.blas.nodes.size() * sizeof(lt::LightBLASNodeGpu);
        p.leafOffset = AlignUp(p.nodeOffset + p.nodeBytes, 16);
        p.leafBytes = l.blas.leafTriLocal.size() * sizeof(uint32_t);
        p.trailOffset = AlignUp(p.leafOffset + p.leafBytes, 16);
        p.trailBytes = l.blas.trails.size() * sizeof(lt::LightTreeTrail);
        p.size = AlignUp(p.trailOffset + p.trailBytes, 256);
        parts.push_back(p);
        total += p.size;
    }
    ComPtr<ID3D12Resource> staging = nv_helpers_dx12::CreateBuffer(
        m_b.device, total, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    uint8_t* base = nullptr;
    ThrowIfFailed(staging->Map(0, nullptr, (void**)&base));
    uint64_t at = 0;
    for (size_t k = 0; k < ready.size(); ++k) {
        const MeshLights& l = *m_meshes[ready[k]].lights;
        const Part& p = parts[k];
        const uint32_t recBase = m_lights.recordBase + (uint32_t)l.recOff;
        uint8_t* dst = base + at;
        memcpy(dst, l.records.data(), p.recBytes);
        const auto words = lt::EncodeLightBLAS(l.blas.nodes, recBase);
        memcpy(dst + p.nodeOffset, words.data(), std::min<uint64_t>(p.nodeBytes, words.size() * sizeof(uint32_t)));
        auto* leaf = reinterpret_cast<uint32_t*>(dst + p.leafOffset);
        for (size_t t = 0; t < l.blas.leafTriLocal.size(); ++t)
            leaf[t] = recBase + l.blas.leafTriLocal[t];
        memcpy(dst + p.trailOffset, l.blas.trails.data(), p.trailBytes);
        at += p.size;
    }
    staging->Unmap(0, nullptr);

    ID3D12Resource* targets[4] = {m_lights.records, m_lights.nodes, m_lights.leafIndex, m_lights.trails};
    D3D12_RESOURCE_BARRIER toCopy[4], toRead[4];
    for (int t = 0; t < 4; ++t) {
        toCopy[t] = CD3DX12_RESOURCE_BARRIER::Transition(targets[t], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                         D3D12_RESOURCE_STATE_COPY_DEST);
        toRead[t] = CD3DX12_RESOURCE_BARRIER::Transition(targets[t], D3D12_RESOURCE_STATE_COPY_DEST,
                                                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    cl->ResourceBarrier(4, toCopy);
    at = 0;
    for (size_t k = 0; k < ready.size(); ++k) {
        MeshLights& l = *m_meshes[ready[k]].lights;
        const Part& p = parts[k];
        const uint64_t rec = m_lights.recordBase + l.recOff;
        cl->CopyBufferRegion(m_lights.records, rec * sizeof(LightTriangle), staging.Get(), at, p.recBytes);
        cl->CopyBufferRegion(m_lights.nodes, (m_lights.nodeBase + l.nodeOff) * sizeof(lt::LightBLASNodeGpu),
                             staging.Get(), at + p.nodeOffset, p.nodeBytes);
        cl->CopyBufferRegion(m_lights.leafIndex, rec * sizeof(uint32_t), staging.Get(), at + p.leafOffset, p.leafBytes);
        cl->CopyBufferRegion(m_lights.trails, rec * sizeof(lt::LightTreeTrail), staging.Get(), at + p.trailOffset,
                             p.trailBytes);
        l.uploaded = true;
        at += p.size;
    }
    cl->ResourceBarrier(4, toRead);
    Retire([staging] {});
}

std::string HostStream::Stats() const {
    size_t meshes = 0, instances = 0, textures = 0;
    for (const Mesh& m : m_meshes)
        meshes += m.live ? 1 : 0;
    for (uint32_t i = 0; i < m_nextInstance; ++i)
        instances += m_instances[i].live ? 1 : 0;
    for (const Texture& t : m_textures)
        textures += t.live ? 1 : 0;
    std::ostringstream s;
    s << meshes << " meshes, " << instances << " instances, " << textures << " textures; vertices "
      << m_vtxAlloc.used() << "/" << m_vtxAlloc.capacity() << ", indices " << m_idxAlloc.used() << "/"
      << m_idxAlloc.capacity();
    return s.str();
}
