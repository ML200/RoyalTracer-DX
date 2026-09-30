#include "../stdafx.h"
#include "mc_omm_bake.h"
#include <algorithm>
#include <map>
#include <omm.h>
#include "block_registry.h"

namespace mc {

bool bake_omm_table(const BlockRegistry& reg, const std::vector<OmmBakeTri>& tris,
                    OmmTable& table, OmmBakeResult& out, std::string* err) {
    table.index.clear();
    out = OmmBakeResult{};
    if (tris.empty()) return true;

    ommBakerCreationDesc bakerDesc{};
    bakerDesc.type = ommBakerType_CPU;
    ommBaker baker = nullptr;
    if (ommCreateBaker(&bakerDesc, &baker) != ommResult_SUCCESS) {
        if (err) *err = "OMM baker creation failed";
        return false;
    }

    std::map<std::pair<uint32_t, uint8_t>, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < (uint32_t)tris.size(); ++i) groups[{ tris[i].tex, tris[i].level }].push_back(i);

    std::vector<ommCpuTexture> textures(reg.textureAlpha.size(), nullptr);
    auto texture_of = [&](uint32_t tex) -> ommCpuTexture {
        if (tex >= textures.size()) return nullptr;
        if (textures[tex]) return textures[tex];
        const BlockRegistry::AlphaMask& am = reg.textureAlpha[tex];
        if (am.empty()) return nullptr;
        ommCpuTextureMipDesc mip{};
        mip.width       = (uint32_t)am.width;
        mip.height      = (uint32_t)am.height;
        mip.rowPitch    = (uint32_t)am.width;
        mip.textureData = am.alpha.data();
        ommCpuTextureDesc td{};
        td.format      = ommCpuTextureFormat_UNORM8;
        td.flags       = ommCpuTextureFlags_None;
        td.mips        = &mip;
        td.mipCount    = 1;
        td.alphaCutoff = 0.5f;
        ommCpuTexture t = nullptr;
        if (ommCpuCreateTexture(baker, &td, &t) != ommResult_SUCCESS) return nullptr;
        textures[tex] = t;
        return t;
    };

    const int32_t unknownOpaque = D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE;
    std::vector<float>    uvs;
    std::vector<uint32_t> indices;
    uint32_t bakes = 0, failed = 0;
    for (const auto& [group, list] : groups) {
        ommCpuTexture texture = texture_of(group.first);
        if (!texture) { for (uint32_t i : list) table.index[tris[i].key] = unknownOpaque; continue; }
        uvs.clear(); indices.clear();
        for (uint32_t i : list)
            for (int k = 0; k < 3; ++k) {
                indices.push_back((uint32_t)(uvs.size() / 2));
                uvs.push_back(tris[i].uv[k][0]);
                uvs.push_back(tris[i].uv[k][1]);
            }
        ommCpuBakeInputDesc in = ommCpuBakeInputDescDefault();
        in.bakeFlags = (ommCpuBakeFlags)(ommCpuBakeFlags_EnableInternalThreads | ommCpuBakeFlags_Force32BitIndices);
        in.texture   = texture;
        in.runtimeSamplerDesc.addressingMode = ommTextureAddressMode_Wrap;
        in.runtimeSamplerDesc.filter         = ommTextureFilterMode_Nearest;
        in.runtimeSamplerDesc.borderAlpha    = 0.0f;
        in.alphaMode             = ommAlphaMode_Test;
        in.texCoordFormat        = ommTexCoordFormat_UV32_FLOAT;
        in.texCoords             = uvs.data();
        in.texCoordStrideInBytes = 0;
        in.indexFormat           = ommIndexFormat_UINT_32;
        in.indexBuffer           = indices.data();
        in.indexCount            = (uint32_t)indices.size();
        in.dynamicSubdivisionScale = 0.0f;
        in.maxSubdivisionLevel     = group.second;
        in.rejectionThreshold      = 0.0f;
        in.alphaCutoff             = 0.5f;
        in.format                  = ommFormat_OC1_4_State;
        in.unknownStatePromotion   = ommUnknownStatePromotion_Nearest;
        ommCpuBakeResult result = nullptr;
        if (ommCpuBake(baker, &in, &result) != ommResult_SUCCESS) {
            ++failed;
            for (uint32_t i : list) table.index[tris[i].key] = unknownOpaque;
            continue;
        }
        ++bakes;
        const ommCpuBakeResultDesc* d = nullptr;
        ommCpuGetBakeResultDesc(result, &d);
        const size_t rawOffset = (out.rawData.size() + 255) & ~(size_t)255;
        out.rawData.resize(rawOffset);
        out.rawData.insert(out.rawData.end(), (const uint8_t*)d->arrayData, (const uint8_t*)d->arrayData + d->arrayDataSize);
        const uint32_t descBase = (uint32_t)out.ommDescs.size();
        for (uint32_t k = 0; k < d->descArrayCount; ++k) {
            D3D12_RAYTRACING_OPACITY_MICROMAP_DESC dx{};
            dx.ByteOffset       = (UINT)(d->descArray[k].offset + rawOffset);
            dx.SubdivisionLevel = d->descArray[k].subdivisionLevel;
            dx.Format           = (D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)d->descArray[k].format;
            out.ommDescs.push_back(dx);
        }
        for (uint32_t h = 0; h < d->descArrayHistogramCount; ++h) {
            const auto& sh = d->descArrayHistogram[h];
            bool merged = false;
            for (auto& e : out.histogram) {
                if (e.SubdivisionLevel == sh.subdivisionLevel && e.Format == (D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)sh.format) {
                    e.Count += sh.count; merged = true; break;
                }
            }
            if (!merged) {
                D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY e{};
                e.Count = sh.count; e.SubdivisionLevel = sh.subdivisionLevel;
                e.Format = (D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)sh.format;
                out.histogram.push_back(e);
            }
        }
        for (size_t n = 0; n < list.size(); ++n) {
            int32_t v;
            if (d->indexFormat == ommIndexFormat_UINT_32) v = ((const int32_t*)d->indexBuffer)[n];
            else if (d->indexFormat == ommIndexFormat_UINT_16) { const uint16_t u = ((const uint16_t*)d->indexBuffer)[n]; v = u >= 0xFFFC ? (int32_t)(int16_t)u : (int32_t)u; }
            else { const uint8_t u = ((const uint8_t*)d->indexBuffer)[n]; v = u >= 0xFC ? (int32_t)(int8_t)u : (int32_t)u; }
            table.index[tris[list[n]].key] = v < 0 ? v : (int32_t)(descBase + v);
        }
        ommCpuDestroyBakeResult(result);
    }
    for (ommCpuTexture t : textures) if (t) ommCpuDestroyTexture(baker, t);
    ommDestroyBaker(baker);
    out.alphaTriCount = (uint32_t)tris.size();
    if (err) *err = failed ? std::to_string(failed) + " of " + std::to_string(bakes + failed) + " bakes failed" : std::string();
    return true;
}

}
