#include "../stdafx.h"
#include "OceanSystem.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include <random>
#include <bit>

namespace ocean {

namespace {

constexpr uint32_t N = OCEAN_FFT_SIZE;
constexpr double kEarthRadius = 6371000.0;

// Deterministic hash so that the Gaussian draw at -k can be reproduced without depending on the
// order the grid is visited. Both halves of a conjugate pair must agree exactly or the synthesised
// surface picks up an imaginary component.
uint32_t Hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint32_t HashCoord(int32_t nx, int32_t nz, uint32_t cascade, uint32_t seed) {
    return Hash(Hash(uint32_t(nx) * 73856093u ^ uint32_t(nz) * 19349663u) ^ Hash(cascade * 83492791u + seed));
}

void GaussianPair(uint32_t h, double& g0, double& g1) {
    const double u1 = std::max(1e-7, (double)(h & 0xFFFFFFu) / (double)0x1000000u);
    const double u2 = (double)(Hash(h) & 0xFFFFFFu) / (double)0x1000000u;
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = 6.283185307179586 * u2;
    g0 = r * std::cos(theta);
    g1 = r * std::sin(theta);
}

ComPtr<ID3D12Resource> CreateTexArray(ID3D12Device* dev, DXGI_FORMAT fmt, uint32_t mips, const wchar_t* name,
                                      bool unorderedAccess = true) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = N;
    d.Height = N;
    d.DepthOrArraySize = OCEAN_CASCADES;
    d.MipLevels = (UINT16)mips;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = unorderedAccess ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

    // Read-only arrays start in COMMON: the spectrum is staged from the copy queue and read from
    // the compute queue, and implicit promotion covers both without a cross-queue transition.
    const D3D12_RESOURCE_STATES initial =
        unorderedAccess ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON;

    ComPtr<ID3D12Resource> r;
    const CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
    ThrowIfFailed(dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, initial, nullptr, IID_PPV_ARGS(&r)));
    r->SetName(name);
    return r;
}

} // namespace

void OceanSystem::Configure(const Params& p) {
    ValidateParams(p);
    const bool respec = !m_initialised || p.windSpeed != m_params.windSpeed || p.fetch != m_params.fetch ||
                        p.windDirectionDeg != m_params.windDirectionDeg || p.swell != m_params.swell ||
                        p.windAlign != m_params.windAlign || p.amplitudeScale != m_params.amplitudeScale ||
                        p.shortWaveAmplitude != m_params.shortWaveAmplitude ||
                        p.foamCoverage != m_params.foamCoverage || p.seaLevelY != m_params.seaLevelY || p.keepAboveZero != m_params.keepAboveZero || p.minClearance != m_params.minClearance || p.seed != m_params.seed || p.choppiness != m_params.choppiness ||
                        p.significantHeight != m_params.significantHeight || p.peakPeriod != m_params.peakPeriod ||
                        p.swellHeight != m_params.swellHeight || p.swellPeriod != m_params.swellPeriod ||
                        p.swellDirectionDeg != m_params.swellDirectionDeg || p.swellSpreadDeg != m_params.swellSpreadDeg;
    m_params = p;
    if (respec)
        m_bakePending = true;
}

OceanSystem::Reservation OceanSystem::GetReservation() const {
    Reservation r;
    if (!m_params.enabled)
        return r;
    r.vertexElems = OCEAN_MAX_TILES * OCEAN_TILE_VERTS;
    r.indexElems = OCEAN_MAX_TILES * OCEAN_TILE_INDICES;
    // Every ocean triangle shares one material, so one tile's worth of identifiers is enough for
    // all of them.
    r.matIDElems = OCEAN_TILE_TRIS;
    r.instanceSlots = OCEAN_MAX_TILES;
    return r;
}

Material OceanSystem::MakeMaterial(const Params& p) {
    Material m;
    // SSS fields describe a participating medium, independent of surface transmission.
    m.Kd = {1.0f, 1.0f, 1.0f, std::clamp(p.bodyWeight, 0.0f, 1.0f)};
    m.Ke = {0.0f, 0.0f, 0.0f};
    m.Ni = 1.333f; // sea water at visible wavelengths
    m.Pr_Pm_Ps_Pc = {0.0f, 0.0f, 0.0f, 0.0f};
    m.diffuseRoughness = 0.0f;
    m.Tf = WaterAbsorptionRGB(p.chlorophyll);
    const auto scatter = WaterScatteringRGB(p.chlorophyll, p.turbidity);
    const float peakScatter = std::max({scatter.x, scatter.y, scatter.z, 1e-5f});
    m.sssAlbedo = {scatter.x / peakScatter, scatter.y / peakScatter, scatter.z / peakScatter};
    m.sssRadius = std::clamp(p.subsurfaceRadiusScale / peakScatter, 0.01f, 1000.0f);
    m.sssPhaseG = p.subsurfacePhaseG;
    m.sssWeight = std::clamp(p.subsurfaceStrength, 0.0f, 1.0f);
    m.sssEnable = m.sssWeight > 0.0f ? 1u : 0u;
    m.albedoTexID = -1;
    m.normalTexID = -1;
    m.rmaTexID = -1;
    m.alphaThreshold = 1.0f;
    (void)p;
    return m;
}

void OceanSystem::Init(ID3D12Device5* device, DeviceContext* ctx) {
    if (!m_params.enabled)
        return;
    m_device = device;
    m_ctx = ctx;

    m_h0 = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, L"OceanH0", false);
    m_wave = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, L"OceanWaveData", false);
    m_disp = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, OCEAN_MIP_LEVELS, L"OceanDisplacement");
    m_deriv = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, OCEAN_MIP_LEVELS, L"OceanDerivatives");
    m_moments = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, OCEAN_MIP_LEVELS, L"OceanRawMoments");
    m_surface = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, OCEAN_MIP_LEVELS, L"OceanCoverageFreshness");
    m_previousDisp = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, OCEAN_MIP_LEVELS, L"OceanPreviousDisplacement");
    m_foam[0] = CreateTexArray(device, DXGI_FORMAT_R32G32_FLOAT, 1, L"OceanFoamA");
    m_foam[1] = CreateTexArray(device, DXGI_FORMAT_R32G32_FLOAT, 1, L"OceanFoamB");

    {
        // Two packed complex fields per slice, two slices per cascade.
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = N;
        d.Height = N;
        d.DepthOrArraySize = OCEAN_CASCADES * 2;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        const CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
        ThrowIfFailed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                      IID_PPV_ARGS(&m_fft)));
        m_fft->SetName(L"OceanFFT");
    }

    m_paramsBuffer = nv_helpers_dx12::CreateBuffer(device, sizeof(OceanParamsGPU), D3D12_RESOURCE_FLAG_NONE,
                                                   D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);
    m_paramsBuffer->SetName(L"OceanParams");
    m_tilesBuffer =
        nv_helpers_dx12::CreateBuffer(device, sizeof(OceanTileGPU) * OCEAN_MAX_TILES, D3D12_RESOURCE_FLAG_NONE,
                                      D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);
    m_tilesBuffer->SetName(L"OceanTiles");

    m_statsBuffer = nv_helpers_dx12::CreateBuffer(
        device, sizeof(XMFLOAT4) * OCEAN_CASCADES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
    m_statsBuffer->SetName(L"OceanStats");
    for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
        m_statsReadback[i] =
            nv_helpers_dx12::CreateBuffer(device, sizeof(XMFLOAT4) * OCEAN_CASCADES, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kReadbackHeapProps);
    }

    for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
        m_paramsUpload[i] =
            nv_helpers_dx12::CreateBuffer(device, sizeof(OceanParamsGPU), D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        m_tilesUpload[i] =
            nv_helpers_dx12::CreateBuffer(device, sizeof(OceanTileGPU) * OCEAN_MAX_TILES, D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    }

    // One staging buffer large enough for both baked arrays.
    const uint64_t bakeBytes = (uint64_t)N * N * OCEAN_CASCADES * sizeof(XMFLOAT4);
    m_bakeUpload = nv_helpers_dx12::CreateBuffer(device, bakeBytes * 2, D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    m_bakeUpload->SetName(L"OceanBakeUpload");

    // Acceleration structure pool. Every tile has identical topology, so one size query covers all
    // of them.
    {
        D3D12_RAYTRACING_GEOMETRY_DESC g = {};
        g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        g.Triangles.VertexCount = VertexSpan();
        g.Triangles.VertexBuffer.StrideInBytes = sizeof(BTriVertex);
        g.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        g.Triangles.IndexCount = OCEAN_TILE_INDICES;

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
        in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        // The surface is re-tessellated every frame, so build speed dominates trace speed here,
        // and allowing refits turns most frames into a fraction of a full build.
        in.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE);
        in.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        in.NumDescs = 1;
        in.pGeometryDescs = &g;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
        m_blasSlotSize = planet::align_up(info.ResultDataMaxSizeInBytes,
                                          D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
        m_blasScratchSize = planet::align_up(info.ScratchDataSizeInBytes,
                                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
        m_blasUpdateScratchSize = planet::align_up(std::max<uint64_t>(info.UpdateScratchDataSizeInBytes, 256),
                                                   D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

        m_blasBuffer = planet::create_buffer(device, m_blasSlotSize * OCEAN_MAX_TILES,
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                             planet::HEAP_DEFAULT);
        m_blasBuffer->SetName(L"OceanBlasPool");
        m_blasScratch = planet::create_buffer(device, m_blasScratchSize * kScratchSlots,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS, planet::HEAP_DEFAULT);
        m_blasScratch->SetName(L"OceanBlasScratch");

        m_stats.blasBytes = m_blasSlotSize * OCEAN_MAX_TILES;
        LOG(L"[ocean] tiles=" << OCEAN_MAX_TILES << L" tris/tile=" << OCEAN_TILE_TRIS << L" blas="
                              << (m_blasSlotSize >> 10) << L" KiB each, pool " << (m_stats.blasBytes >> 20)
                              << L" MiB; scratch " << (m_blasScratchSize >> 10) << L" KiB x " << kScratchSlots
                              << L" = " << ((m_blasScratchSize * kScratchSlots) >> 20) << L" MiB");
    }

    m_blasBuilt.assign(OCEAN_MAX_TILES, 0);

    // Root signature: root constants plus direct heap indexing, so no descriptor tables are needed
    // for any of the ocean's own resources.
    {
        CD3DX12_ROOT_PARAMETER1 rp[1];
        rp[0].InitAsConstants(8, 0, 0, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_STATIC_SAMPLER_DESC samp[1];
        samp[0].Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                     D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        // Only the resource heap is indexed directly; the one sampler the tessellator needs is
        // static, so no sampler heap has to be bound on the streaming queue.
        desc.Init_1_1(1, rp, 1, samp, D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);
        ComPtr<ID3DBlob> sig, err;
        HRESULT hr = D3D12SerializeVersionedRootSignature(&desc, &sig, &err);
        if (FAILED(hr)) {
            if (err)
                OutputDebugStringA((char*)err->GetBufferPointer());
            ThrowIfFailed(hr);
        }
        ThrowIfFailed(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                                  IID_PPV_ARGS(&m_rootSig)));
        m_rootSig->SetName(L"OceanRootSignature");
    }

    auto makePso = [&](const wchar_t* file, const wchar_t* entry, ComPtr<ID3D12PipelineState>& out) {
        ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(file, entry);
        D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
        d.pRootSignature = m_rootSig.Get();
        d.CS = {cs->GetBufferPointer(), cs->GetBufferSize()};
        ThrowIfFailed(device->CreateComputePipelineState(&d, IID_PPV_ARGS(&out)));
        out->SetName(entry);
    };
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanEvolve", m_psoEvolve);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanFftH", m_psoFftH);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanFftV", m_psoFftV);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanAssemble", m_psoAssemble);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanCondition", m_psoCondition);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanFoam", m_psoFoam);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanMip", m_psoMip);
    makePso(L"Ocean_Tiles_v8.hlsl", L"OceanTiles", m_psoTiles);

    if (const char* path = std::getenv("RT_OCEAN_PERF_CSV")) {
        m_profile.open(path);
        m_profile << "frame,fft_ms,assemble_ms,foam_ms,mips_ms,mesh_ms,blas_ms,total_ms,tiles,triangles,allocated_bytes\n";
        D3D12_QUERY_HEAP_DESC q{};
        q.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        q.Count = FRAME_COUNT * 7;
        ThrowIfFailed(device->CreateQueryHeap(&q, IID_PPV_ARGS(&m_timestamps)));
        m_timestampReadback = planet::create_buffer(device, FRAME_COUNT * 7 * sizeof(uint64_t),
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, planet::HEAP_READBACK);
        ThrowIfFailed(ctx->PlanetComputeQueue()->GetTimestampFrequency(&m_timestampFrequency));
    }
    // Actual committed allocation sizes; global geometry's ocean reservation is added separately.
    ID3D12Resource* resources[] = {m_h0.Get(),m_wave.Get(),m_fft.Get(),m_disp.Get(),m_deriv.Get(),
        m_moments.Get(),m_surface.Get(),m_previousDisp.Get(),m_foam[0].Get(),m_foam[1].Get(),
        m_bakeUpload.Get(),m_paramsBuffer.Get(),m_tilesBuffer.Get(),m_statsBuffer.Get(),
        m_blasBuffer.Get(),m_blasScratch.Get(),m_timestampReadback.Get()};
    m_stats.resourceBytes = 0;
    auto account = [&](ID3D12Resource* r) { if (r) { const auto d=r->GetDesc();
        m_stats.resourceBytes += device->GetResourceAllocationInfo(0,1,&d).SizeInBytes; } };
    for (auto* r : resources) account(r);
    for (uint32_t i=0;i<FRAME_COUNT;++i) {
        account(m_paramsUpload[i].Get()); account(m_tilesUpload[i].Get()); account(m_statsReadback[i].Get());
    }
    const auto reservation = GetReservation();
    m_stats.resourceBytes += uint64_t(reservation.vertexElems)*sizeof(BTriVertex) + uint64_t(reservation.indexElems)*4;
    LOG(L"[ocean] resources including reserved geometry=" << m_stats.resourceBytes << L" bytes");

    m_initialised = true;
    Bake();
}

void OceanSystem::BindScene(ID3D12Resource* globalVertex, ID3D12Resource* globalIndex, uint32_t vertexBase,
                            uint32_t indexBase, uint32_t materialBase, uint32_t propsBase, uint32_t materialIndex) {
    m_globalVertex = globalVertex;
    m_globalIndex = globalIndex;
    m_vertexBase = vertexBase;
    m_indexBase = indexBase;
    m_materialBase = materialBase;
    m_propsBase = propsBase;
    m_materialIndex = materialIndex;
}

void OceanSystem::CreateDescriptors(ID3D12Device* device, ID3D12DescriptorHeap* heap) {
    if (!m_initialised || !m_params.enabled)
        return;

    // Kept so the simulation can bind it on the streaming compute queue. Every ocean shader reads
    // its resources through ResourceDescriptorHeap, and that is only defined while the heap is
    // bound on the list doing the work - the graphics list binding it does not carry over.
    m_srvHeap = heap;

    const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto at = [&](uint32_t slot) {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(heap->GetCPUDescriptorHandleForHeapStart(), (INT)slot, inc);
    };

    auto texSrv = [&](ID3D12Resource* res, uint32_t mips, uint32_t slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Format = res->GetDesc().Format;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        s.Texture2DArray.MipLevels = mips;
        s.Texture2DArray.ArraySize = OCEAN_CASCADES;
        device->CreateShaderResourceView(res, &s, at(slot));
    };
    auto texUav = [&](ID3D12Resource* res, uint32_t mip, uint32_t slot) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = res->GetDesc().Format;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        u.Texture2DArray.MipSlice = mip;
        u.Texture2DArray.ArraySize = OCEAN_CASCADES;
        device->CreateUnorderedAccessView(res, nullptr, &u, at(slot));
    };
    auto structuredSrv = [&](ID3D12Resource* res, uint32_t count, uint32_t stride, uint32_t slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Format = DXGI_FORMAT_UNKNOWN;
        s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        s.Buffer.NumElements = count;
        s.Buffer.StructureByteStride = stride;
        device->CreateShaderResourceView(res, &s, at(slot));
    };

    texSrv(m_disp.Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_DISP);
    texSrv(m_deriv.Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_DERIV);
    texSrv(m_moments.Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_MOMENTS);
    texSrv(m_surface.Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_SURFACE);
    texSrv(m_previousDisp.Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_PREV_DISP);
    texSrv(m_h0.Get(), 1, OCEAN_SRV_H0);
    texSrv(m_wave.Get(), 1, OCEAN_SRV_WAVE);
    structuredSrv(m_paramsBuffer.Get(), 1, sizeof(OceanParamsGPU), OCEAN_SRV_PARAMS);
    structuredSrv(m_tilesBuffer.Get(), OCEAN_MAX_TILES, sizeof(OceanTileGPU), OCEAN_SRV_TILES);

    const uint32_t vertexCount = m_vertexBase + OCEAN_MAX_TILES * OCEAN_TILE_VERTS;
    structuredSrv(m_globalVertex, vertexCount, sizeof(BTriVertex), OCEAN_SRV_VERTS);
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Format = DXGI_FORMAT_R32_UINT;
        s.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        s.Buffer.NumElements = m_indexBase + OCEAN_MAX_TILES * OCEAN_TILE_INDICES;
        device->CreateShaderResourceView(m_globalIndex, &s, at(OCEAN_SRV_INDICES));
    }

    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = DXGI_FORMAT_UNKNOWN;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        u.Texture2DArray.ArraySize = OCEAN_CASCADES * 2;
        u.Format = m_fft->GetDesc().Format;
        device->CreateUnorderedAccessView(m_fft.Get(), nullptr, &u, at(OCEAN_UAV_FFT));
    }
    texUav(m_foam[0].Get(), 0, OCEAN_UAV_FOAM_A);
    texUav(m_foam[1].Get(), 0, OCEAN_UAV_FOAM_B);
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = DXGI_FORMAT_UNKNOWN;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = OCEAN_CASCADES;
        u.Buffer.StructureByteStride = sizeof(XMFLOAT4);
        device->CreateUnorderedAccessView(m_statsBuffer.Get(), nullptr, &u, at(OCEAN_UAV_STATS));
    }
    for (uint32_t m = 0; m < OCEAN_MIP_LEVELS; ++m) {
        texUav(m_disp.Get(), m, OCEAN_UAV_DISP_MIPS + m);
        texUav(m_deriv.Get(), m, OCEAN_UAV_DERIV_MIPS + m);
        texUav(m_moments.Get(), m, OCEAN_UAV_MOMENT_MIPS + m);
        texUav(m_surface.Get(), m, OCEAN_UAV_SURFACE_MIPS + m);
    }

    // The tessellator writes vertices straight into the scene's global buffer, addressing it with
    // the same element indices the hit evaluator later reads.
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = DXGI_FORMAT_UNKNOWN;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = vertexCount;
        u.Buffer.StructureByteStride = sizeof(BTriVertex);
        device->CreateUnorderedAccessView(m_globalVertex, nullptr, &u, at(OCEAN_UAV_VERTS));
    }
}

void OceanSystem::Bake() {
    if (!m_initialised || !m_params.enabled)
        return;
    const auto t0 = std::chrono::high_resolution_clock::now();

    m_spectrum.Init(m_params);

    m_h0Data.assign((size_t)N * N * OCEAN_CASCADES, XMFLOAT4{0, 0, 0, 0});
    m_waveData.assign((size_t)N * N * OCEAN_CASCADES, XMFLOAT4{0, 0, 0, 0});

    const double windRad = (double)m_params.windDirectionDeg * 0.017453292519943295;
    const double wc = std::cos(windRad), ws = std::sin(windRad);
    const double amp = std::max(0.0, (double)m_params.amplitudeScale);

    double varAlong[OCEAN_CASCADES] = {};
    double varCross[OCEAN_CASCADES] = {};
    double elevationVar = 0.0;

    // Gaussian amplitudes use half sqrt(PSD * bin area); evolution preserves Hermitian pairs.
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        const double L = CascadeLengths()[c];
        const double dk = 6.283185307179586 / L;

        for (uint32_t m = 0; m < N; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                const int32_t inx = (int32_t)n - (int32_t)(N / 2);
                const int32_t inz = (int32_t)m - (int32_t)(N / 2);
                const double kx = (double)inx * dk;
                const double kz = (double)inz * dk;
                const double k = std::sqrt(kx * kx + kz * kz);

                const size_t idx = ((size_t)c * N + m) * N + n;
                XMFLOAT4& wave = m_waveData[idx];
                wave = XMFLOAT4{(float)kx, (float)kz, 0.0f, 0.0f};

                if (k < 1e-6 || n == 0 || m == 0)
                    continue;
                // Share of this wavelength that belongs to this cascade. Amplitude carries the
                // square root because it is power that partitions.
                const double weight = CascadeWeight((int)c, k);
                if (weight <= 1e-6)
                    continue;
                const double ampWeight = std::sqrt(weight);

                const double omega = std::sqrt(kGravity * k);
                wave.z = (float)(1.0 / k);
                wave.w = (float)omega;

                // Rotate into the wind frame: the spectrum's direction is measured from the wind.
                const double kxw = kx * ws + kz * wc;
                const double kzw = kx * wc - kz * ws;

                const double detailGain = ShortWaveAmplitude(m_params, k);
                const double detailPower = detailGain * detailGain;
                const double A = 0.5 * std::sqrt(std::max(0.0, detailPower * m_spectrum.S2D(kxw, kzw) +
                    SwellDensity(m_params, kx, kz)) * dk * dk) * amp * ampWeight;
                double g0, g1;
                GaussianPair(HashCoord(inx, inz, c, m_params.seed), g0, g1);

                const double An = 0.5 * std::sqrt(std::max(0.0, detailPower * m_spectrum.S2D(-kxw, -kzw) +
                    SwellDensity(m_params, -kx, -kz)) * dk * dk) * amp * ampWeight;
                double h0, h1;
                GaussianPair(HashCoord(-inx, -inz, c, m_params.seed), h0, h1);

                m_h0Data[idx] = XMFLOAT4{(float)(A * g0), (float)(A * g1), (float)(An * h0), (float)(-An * h1)};

                // Expected power of this mode, and the slope variance it carries.
                const double power = 4.0 * A * A;
                elevationVar += power;
                varAlong[c] += kxw * kxw * power;
                varCross[c] += kzw * kzw * power;
            }
        }
    }

    // The GPU bounds the actual bilinear map each frame before geometry or foam consumes it.
    m_effectiveChoppiness = m_params.choppiness;
    m_phaseEpoch = 0.0;
    m_resetFoam = true;

    double cmAlong = 0.0, cmCross = 0.0;
    CoxMunkSlopeVariance(m_params.windSpeed, cmAlong, cmCross);
    double sumAlong = 0.0, sumCross = 0.0;
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        sumAlong += varAlong[c];
        sumCross += varCross[c];
    }

    // Unrepresented high-frequency energy from the SAME wind spectrum, with an explicit
    // gravity-wave validity cutoff. No subtraction from unrelated Cox-Munk totals.
    // This bounded gravity tail is an approximation, not a capillary-wave spectrum.
    double tailAlong = 0.0, tailCross = 0.0;
    const double k0 = CascadeNyquist(OCEAN_CASCADES - 1);
    const double k1 = 370.0; // rad/m; gravity/capillary crossover, deliberately no extrapolation above
    const double dlog = std::max(0.0, std::log(k1 / k0)) / 512.0;
    for (int i = 0; i < 512 && dlog > 0.0; ++i) {
        const double k = k0 * std::exp((i + 0.5) * dlog);
        for (int j = 0; j < 128; ++j) {
            const double theta = (j + 0.5) * (6.283185307179586 / 128.0);
            const double x = k * std::cos(theta), z = k * std::sin(theta);
            const double detailGain = ShortWaveAmplitude(m_params, k);
            const double power = m_spectrum.S2D(x, z) * k * k * dlog * (6.283185307179586 / 128.0) * amp * amp * detailGain * detailGain;
            tailAlong += x * x * power;
            tailCross += z * z * power;
        }
    }

    m_gpuParams.slopeVarTailAlong = (float)tailAlong;
    m_gpuParams.slopeVarTailCross = (float)tailCross;
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        double kMin, kMax;
        CascadeBand((int)c, kMin, kMax);
        ((float*)&m_gpuParams.cascadeLength)[c] = (float)CascadeLengths()[c];
        ((float*)&m_gpuParams.cascadeKMin)[c] = (float)kMin;
        ((float*)&m_gpuParams.cascadeKMax)[c] = (float)kMax;
        ((float*)&m_gpuParams.slopeVarAlong)[c] = (float)varAlong[c];
        ((float*)&m_gpuParams.slopeVarCross)[c] = (float)varCross[c];
    }

    m_stats.slopeVarSpectrum = sumAlong + sumCross;
    m_stats.slopeVarCoxMunk = cmAlong + cmCross;
    m_stats.whitecapCoverage = 0.0;

    // Taken from the analytic prediction rather than from the discrete sum above, so that a scene
    // can ask for the water line before the field is baked and get the same answer.
    m_waveDepth = PredictWaveDepth(m_params);
    m_surfaceY = PredictSurfaceLevel(m_params);
    m_stats.surfaceY = m_surfaceY;
    m_stats.significantWaveHeight = 4.0 * std::sqrt(std::max(0.0, elevationVar));
    m_stats.bakeMs =
        std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    LOG(L"[ocean] sea level " << m_surfaceY << L" m, deepest trough " << (m_surfaceY - m_waveDepth) << L" m");
    LOG(L"[ocean] wind=" << m_params.windSpeed << L" m/s fetch=" << (m_params.fetch / 1000.0f) << L" km Hs="
                         << m_stats.significantWaveHeight << L" m  slope var: spectrum="
                         << m_stats.slopeVarSpectrum << L" Cox-Munk=" << m_stats.slopeVarCoxMunk << L" whitecap=" << (m_stats.whitecapCoverage * 100.0) << L"% tail="
                         << (tailAlong + tailCross) << L" (bake " << m_stats.bakeMs << L" ms)");

    m_bakePending = false;
    m_paramsDirty = true;
    m_h0Dirty = true;
}

void OceanSystem::BeginFrame(float dt, const planet::CameraView& cam, uint32_t frameIndex) {
    if (!m_initialised || !m_params.enabled)
        return;

    m_dt = m_params.paused ? 0.0f : std::clamp(m_params.fixedTimeStep > 0.0f ? m_params.fixedTimeStep : dt, 0.0f, 0.1f);
    m_time += m_dt;
    m_frameIndex = frameIndex % FRAME_COUNT;
    m_previousOrigin = m_sceneOrigin;
    m_sceneOrigin = cam.scene_origin;

    if (m_bakePending)
        Bake();

    // Fold a double-precision epoch into the complex amplitudes. Reducing time alone would
    // jump phase because ocean frequencies are not integer multiples of a common period.
    const double epoch = std::floor(m_time / 128.0) * 128.0;
    if (epoch != m_phaseEpoch) {
        const double elapsed = epoch - m_phaseEpoch;
        for (size_t i = 0; i < m_h0Data.size(); ++i) {
            auto& h = m_h0Data[i];
            const double phase = std::remainder(double(m_waveData[i].w) * elapsed, 6.283185307179586);
            const double c = std::cos(phase), s = std::sin(phase);
            h = {(float)(h.x*c + h.y*s), (float)(h.y*c - h.x*s),
                 (float)(h.z*c - h.w*s), (float)(h.w*c + h.z*s)};
        }
        m_phaseEpoch = epoch;
        m_h0Dirty = true;
    }

    // Read deformation conditioning and timing statistics from the completed frame.
    {
        const uint32_t slot = (m_frameIndex + 1u) % FRAME_COUNT;
        if (m_statsReadback[slot] && m_statsFence[slot] != 0 &&
            m_ctx->PlanetComputeCompleted() >= m_statsFence[slot]) {
            if (m_timestamps) {
                const uint64_t* ticks = nullptr;
                const CD3DX12_RANGE range(slot*7*sizeof(uint64_t),(slot+1)*7*sizeof(uint64_t));
                ThrowIfFailed(m_timestampReadback->Map(0,&range,(void**)&ticks));
                ticks += slot*7;
                m_profile << m_profileFrame++;
                for (uint32_t i=1;i<7;++i) m_profile << ',' << double(ticks[i]-ticks[i-1])*1000.0/m_timestampFrequency;
                m_profile << ',' << double(ticks[6]-ticks[0])*1000.0/m_timestampFrequency << ','
                    << m_stats.tiles << ',' << m_stats.triangles << ',' << m_stats.resourceBytes << '\n';
                m_profile.flush();
                const CD3DX12_RANGE none(0,0); m_timestampReadback->Unmap(0,&none);
            }
            const XMFLOAT4* src = nullptr;
            const CD3DX12_RANGE range(0, sizeof(XMFLOAT4) * OCEAN_CASCADES);
            if (SUCCEEDED(m_statsReadback[slot]->Map(0, &range, (void**)&src)) && src) {
                for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
                    m_stats.conditioningGain = src[c].w;
                }
                const CD3DX12_RANGE none(0, 0);
                m_statsReadback[slot]->Unmap(0, &none);
            }
        }
    }

    Params selectParams = m_params;
    selectParams.seaLevelY = (float)m_surfaceY;
    m_quadtree.Select(selectParams, cam, OCEAN_MAX_TILES);

    const auto& tiles = m_quadtree.Tiles();
    m_gpuTiles.clear();
    m_gpuTiles.reserve(tiles.size());
    const double invR = m_params.curvature ? 1.0 / kEarthRadius : 0.0;

    for (const TileDesc& t : tiles) {
        OceanTileGPU g{};
        g.anchor = XMFLOAT3((float)(t.minX - m_sceneOrigin.x), (float)(m_surfaceY - m_sceneOrigin.y),
                            (float)(t.minZ - m_sceneOrigin.z));
        g.size = (float)t.size;
        g.vertexBase = m_vertexBase + t.slot * OCEAN_TILE_VERTS;
        g.stitch = t.stitch;
        g.level = t.level;
        g.pad = 0;
        // drop(p) = |a + p|^2 / 2R expanded about the tile anchor, so the shader never squares a
        // hundred-kilometre coordinate in single precision.
        g.curveBase = (float)((t.minX * t.minX + t.minZ * t.minZ) * 0.5 * invR);
        g.curveGrad = XMFLOAT2((float)(t.minX * invR), (float)(t.minZ * invR));
        g.invCurveRadius = (float)invR;
        m_gpuTiles.push_back(g);
    }

    m_stats.whitecapMeasured = 0.0;

    m_stats.tiles = (uint32_t)m_gpuTiles.size();
    m_stats.leaves = m_quadtree.SelectedLeafCount();
    m_stats.dropped = m_quadtree.DroppedCount();
    m_stats.triangles = (uint64_t)m_gpuTiles.size() * OCEAN_TILE_TRIS;

    // Per-frame shader parameters.
    OceanParamsGPU& P = m_gpuParams;
    const double windRad = (double)m_params.windDirectionDeg * 0.017453292519943295;
    P.windDir = XMFLOAT2((float)std::sin(windRad), (float)std::cos(windRad));
    P.surfaceY = (float)(m_surfaceY - m_sceneOrigin.y);
    P.choppiness = m_effectiveChoppiness;
    P.displacementScale = 1.0f;
    P.waveHeightScale = 1.0f;
    P.filterScale = m_params.filterScale;
    // Retain the shared layout, but neither foam nor a roughness floor affects water shading.
    P.foamCoverageScale = 0.0f;
    P.foamRoughness = 0.0f;
    P.minRoughness = 0.0f;

    P.upwelling = WaterUpwellingRGB(m_params.chlorophyll, m_params.turbidity);
    P.bodyStrength = m_params.subsurfaceStrength;
    P.foamAlbedo = XMFLOAT3(m_params.foamAlbedo, m_params.foamAlbedo, m_params.foamAlbedo);
    P.invRadius = m_params.curvature ? float(1.0 / 6371000.0) : 0.0f;
    P.curveOrigin = {(float)m_sceneOrigin.x, (float)m_sceneOrigin.z};
    P.materialBase = m_materialIndex;
    P.originDelta = {(float)(m_sceneOrigin.x - m_previousOrigin.x), (float)(m_sceneOrigin.y - m_previousOrigin.y),
                     (float)(m_sceneOrigin.z - m_previousOrigin.z)};
    P.historyValid = (!m_firstRecord && !m_resetFoam) ? 1.0f : 0.0f;
    P.bodyWeight = std::clamp(m_params.bodyWeight, 0.0f, 1.0f);
    P.halfExtent = std::max(64.0f, m_params.extent);
    P.debugMode = m_params.debugMode;

    // Cascades tile in absolute world space; folding the floating origin in per cascade keeps the
    // waves pinned to the world while the shader's coordinates stay small.
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        const double L = CascadeLengths()[c];
        double wx = std::fmod(m_sceneOrigin.x, L);
        double wz = std::fmod(m_sceneOrigin.z, L);
        if (wx < 0.0)
            wx += L;
        if (wz < 0.0)
            wz += L;
        ((float*)&P.originWrapX)[c] = (float)wx;
        ((float*)&P.originWrapZ)[c] = (float)wz;
    }
}

uint32_t OceanSystem::instance_capacity() const {
    return m_params.enabled ? OCEAN_MAX_TILES : 0u;
}

void OceanSystem::record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) {
    if (!m_initialised || !m_params.enabled || !m_srvHeap)
        return;

    // The streaming lists are reset every frame and the orchestrator has no descriptor heap of its
    // own, so the ocean binds one before any of its dispatches index into it.
    ID3D12DescriptorHeap* heaps[] = {m_srvHeap};
    computeList->SetDescriptorHeaps(1, heaps);

    // Staged uploads travel on the copy queue, which the compute queue waits on.
    {
        void* dst = nullptr;
        const CD3DX12_RANGE none(0, 0);
        ThrowIfFailed(m_paramsUpload[m_frameIndex]->Map(0, &none, &dst));
        std::memcpy(dst, &m_gpuParams, sizeof(OceanParamsGPU));
        m_paramsUpload[m_frameIndex]->Unmap(0, nullptr);
        copyList->CopyBufferRegion(m_paramsBuffer.Get(), 0, m_paramsUpload[m_frameIndex].Get(), 0,
                                   sizeof(OceanParamsGPU));
    }
    if (!m_gpuTiles.empty()) {
        void* dst = nullptr;
        const CD3DX12_RANGE none(0, 0);
        ThrowIfFailed(m_tilesUpload[m_frameIndex]->Map(0, &none, &dst));
        std::memcpy(dst, m_gpuTiles.data(), m_gpuTiles.size() * sizeof(OceanTileGPU));
        m_tilesUpload[m_frameIndex]->Unmap(0, nullptr);
        copyList->CopyBufferRegion(m_tilesBuffer.Get(), 0, m_tilesUpload[m_frameIndex].Get(), 0,
                                   m_gpuTiles.size() * sizeof(OceanTileGPU));
    }
    if (m_h0Dirty)
        UploadBaked(copyList);

    Timestamp(computeList, 0);
    RecordSimulation(computeList);
    RecordTessellation(computeList);
    Timestamp(computeList, 5);
    RecordAccelerationStructures(computeList);
    Timestamp(computeList, 6);
    if (m_timestamps) computeList->ResolveQueryData(m_timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP,
        m_frameIndex*7, 7, m_timestampReadback.Get(), m_frameIndex*7*sizeof(uint64_t));
}

void OceanSystem::UploadBaked(ID3D12GraphicsCommandList* copyList) {
    const uint64_t rowBytes = (uint64_t)N * sizeof(XMFLOAT4);
    const uint64_t rowPitch = planet::align_up(rowBytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const uint64_t sliceBytes = rowPitch * N;
    const uint64_t needed = sliceBytes * OCEAN_CASCADES * 2;

    if (m_bakeUpload->GetDesc().Width < needed) {
        m_bakeUpload = nv_helpers_dx12::CreateBuffer(m_device, needed, D3D12_RESOURCE_FLAG_NONE,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ,
                                                     nv_helpers_dx12::kUploadHeapProps);
        m_bakeUpload->SetName(L"OceanBakeUpload");
    }

    uint8_t* base = nullptr;
    const CD3DX12_RANGE none(0, 0);
    ThrowIfFailed(m_bakeUpload->Map(0, &none, (void**)&base));

    auto stage = [&](const std::vector<XMFLOAT4>& src, uint64_t offset, ID3D12Resource* dstTex) {
        for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
            for (uint32_t row = 0; row < N; ++row) {
                std::memcpy(base + offset + (uint64_t)c * sliceBytes + row * rowPitch,
                            src.data() + ((size_t)c * N + row) * N, rowBytes);
            }
        }
        for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
            D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
            dstLoc.pResource = dstTex;
            dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dstLoc.SubresourceIndex = c; // one mip level, so the slice index is the subresource

            D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
            srcLoc.pResource = m_bakeUpload.Get();
            srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            srcLoc.PlacedFootprint.Offset = offset + (uint64_t)c * sliceBytes;
            srcLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
            srcLoc.PlacedFootprint.Footprint.Width = N;
            srcLoc.PlacedFootprint.Footprint.Height = N;
            srcLoc.PlacedFootprint.Footprint.Depth = 1;
            srcLoc.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
            copyList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        }
    };

    stage(m_h0Data, 0, m_h0.Get());
    stage(m_waveData, sliceBytes * OCEAN_CASCADES, m_wave.Get());
    m_bakeUpload->Unmap(0, nullptr);
    m_h0Dirty = false;
}

void OceanSystem::RecordSimulation(ID3D12GraphicsCommandList4* cl) {
    struct Push {
        uint32_t u0, u1, u2, u3;
        float f0, f1, f2, f3;
    } push{};

    cl->SetComputeRootSignature(m_rootSig.Get());

    auto uavBarrier = [&](ID3D12Resource* r) {
        const D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(r);
        cl->ResourceBarrier(1, &b);
    };
    const uint32_t groups = (N + 7) / 8;

    // The pyramids end each frame as shader resources for the tessellator and the path tracer;
    // bring them back for writing. They are created in the unordered-access state, so the very
    // first frame has nothing to undo.
    if (!m_firstRecord) {
        const D3D12_RESOURCE_BARRIER historyCopy[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_disp.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_previousDisp.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST),
        };
        cl->ResourceBarrier(2, historyCopy);
        cl->CopyResource(m_previousDisp.Get(), m_disp.Get());
        const D3D12_RESOURCE_BARRIER toWrite[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_previousDisp.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_disp.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(m_deriv.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(m_moments.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(m_surface.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        };
        cl->ResourceBarrier((UINT)std::size(toWrite), toWrite);
    }
    const bool firstRecord = m_firstRecord;
    m_firstRecord = false;
    m_resetFoam = false;

    push.f0 = (float)(m_time - m_phaseEpoch);
    push.f1 = m_dt;
    cl->SetPipelineState(m_psoEvolve.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(groups, groups, OCEAN_CASCADES);
    uavBarrier(m_fft.Get());

    cl->SetPipelineState(m_psoFftH.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(N, OCEAN_CASCADES * 2, 1);
    uavBarrier(m_fft.Get());

    cl->SetPipelineState(m_psoFftV.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(N, OCEAN_CASCADES * 2, 1);
    uavBarrier(m_fft.Get());

    cl->SetPipelineState(m_psoAssemble.Get());
    Timestamp(cl, 1);
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(groups, groups, OCEAN_CASCADES);
    uavBarrier(m_disp.Get());
    uavBarrier(m_deriv.Get());
    uavBarrier(m_moments.Get());
    // Reduce raw moments and a maximum strain bound before conditioning the displacement.
    cl->SetPipelineState(m_psoMip.Get());
    push.u0 = 2;
    for (uint32_t level=1;level<OCEAN_MIP_LEVELS;++level) {
        push.u1=level;
        const uint32_t size=N>>level;
        cl->SetComputeRoot32BitConstants(0,8,&push,0);
        cl->Dispatch((size+7)/8,(size+7)/8,OCEAN_CASCADES);
        uavBarrier(m_moments.Get());
    }
    cl->SetPipelineState(m_psoCondition.Get());
    cl->Dispatch(groups,groups,OCEAN_CASCADES);
    uavBarrier(m_disp.Get()); uavBarrier(m_deriv.Get());
    Timestamp(cl, 2);

    // The legacy foam entry point now writes deformation diagnostics with zero foam channels.
    cl->SetPipelineState(m_psoFoam.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(groups, groups, OCEAN_CASCADES);
    uavBarrier(m_surface.Get());
    Timestamp(cl, 3);

    cl->SetPipelineState(m_psoMip.Get());
    ID3D12Resource* pyramids[] = {m_disp.Get(), m_deriv.Get(), m_moments.Get(), m_surface.Get()};
    for (uint32_t pyramid = 0; pyramid < 4; ++pyramid) {
        if (pyramid == 2) continue; // raw moments already reduced for conditioning
        for (uint32_t level = 1; level < OCEAN_MIP_LEVELS; ++level) {
            const uint32_t size = std::max(1u, N >> level);
            push.u0 = pyramid;
            push.u1 = level;
            cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
            cl->Dispatch((size + 7) / 8, (size + 7) / 8, OCEAN_CASCADES);
            uavBarrier(pyramids[pyramid]);
        }
    }

    // Stage the coverage the pyramid just measured; BeginFrame picks it up once the fence for this
    // slot retires, two frames later.
    {
        const D3D12_RESOURCE_BARRIER toCopy =
            CD3DX12_RESOURCE_BARRIER::Transition(m_statsBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COPY_SOURCE);
        cl->ResourceBarrier(1, &toCopy);
        cl->CopyResource(m_statsReadback[m_frameIndex].Get(), m_statsBuffer.Get());
        const D3D12_RESOURCE_BARRIER back =
            CD3DX12_RESOURCE_BARRIER::Transition(m_statsBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cl->ResourceBarrier(1, &back);
    }

    // The tessellator and the path tracer sample these, so they leave the simulation as shader
    // resources. Both states are legal on a compute queue, which is what lets the whole ocean run
    // off the graphics timeline.
    if (firstRecord) {
        const D3D12_RESOURCE_BARRIER copy[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_disp.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
            CD3DX12_RESOURCE_BARRIER::Transition(m_previousDisp.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST),
        };
        cl->ResourceBarrier(2, copy);
        cl->CopyResource(m_previousDisp.Get(), m_disp.Get());
        const D3D12_RESOURCE_BARRIER ready[] = {
            CD3DX12_RESOURCE_BARRIER::Transition(m_disp.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            CD3DX12_RESOURCE_BARRIER::Transition(m_previousDisp.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        };
        cl->ResourceBarrier(2, ready);
    }
    const D3D12_RESOURCE_BARRIER toRead[] = {
        CD3DX12_RESOURCE_BARRIER::Transition(m_disp.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_deriv.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_moments.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(m_surface.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
    };
    cl->ResourceBarrier((UINT)std::size(toRead), toRead);
    Timestamp(cl, 4);
}

void OceanSystem::RecordTessellation(ID3D12GraphicsCommandList4* cl) {
    if (m_gpuTiles.empty())
        return;

    const D3D12_RESOURCE_BARRIER toUav = CD3DX12_RESOURCE_BARRIER::Transition(
        m_globalVertex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->ResourceBarrier(1, &toUav);

    struct Push {
        uint32_t u0, u1, u2, u3;
        float f0, f1, f2, f3;
    } push{};
    push.u0 = (uint32_t)m_gpuTiles.size();
    push.f0 = m_time;
    push.f1 = m_dt;

    cl->SetComputeRootSignature(m_rootSig.Get());
    cl->SetPipelineState(m_psoTiles.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    const uint32_t g = (OCEAN_TILE_EDGE_VERTS + 7) / 8;
    cl->Dispatch(g, g, (uint32_t)m_gpuTiles.size());

    const D3D12_RESOURCE_BARRIER toRead = CD3DX12_RESOURCE_BARRIER::Transition(
        m_globalVertex, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cl->ResourceBarrier(1, &toRead);
}

void OceanSystem::RecordAccelerationStructures(ID3D12GraphicsCommandList4* cl) {
    m_stats.builds = 0;
    m_stats.refits = 0;
    if (m_gpuTiles.empty())
        return;

    const D3D12_GPU_VIRTUAL_ADDRESS vbBase = m_globalVertex->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS ibBase = m_globalIndex->GetGPUVirtualAddress();

    const auto& tiles = m_quadtree.Tiles();
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileDesc& t = tiles[i];

        D3D12_RAYTRACING_GEOMETRY_DESC g = {};
        g.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        g.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        // The stored indices are absolute positions in the global buffer, because that is what the
        // hit evaluator reads, so the vertex range has to start at the buffer's base rather than at
        // this tile's block.
        g.Triangles.VertexCount = VertexSpan();
        g.Triangles.VertexBuffer.StrideInBytes = sizeof(BTriVertex);
        g.Triangles.VertexBuffer.StartAddress = vbBase;
        g.Triangles.IndexFormat = DXGI_FORMAT_R32_UINT;
        g.Triangles.IndexCount = OCEAN_TILE_INDICES;
        g.Triangles.IndexBuffer =
            ibBase + (uint64_t)(m_indexBase + t.slot * OCEAN_TILE_INDICES) * sizeof(uint32_t);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d = {};
        d.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        d.Inputs.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD |
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE);
        d.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        d.Inputs.NumDescs = 1;
        d.Inputs.pGeometryDescs = &g;
        d.DestAccelerationStructureData = m_blasBuffer->GetGPUVirtualAddress() + (uint64_t)t.slot * m_blasSlotSize;

        // A tile that held the same square of ocean last frame keeps its topology: only the
        // vertices moved, so a refit is valid and costs a fraction of a build. Every slot is
        // rebuilt periodically anyway, because refitting indefinitely lets the tree quality decay
        // as the waves travel through it.
        const bool stale = ((m_frameCounter + t.slot) % 8u) == 0u;
        const bool canUpdate = m_blasBuilt[t.slot] != 0 && !m_quadtree.SlotIsNew(t.slot) && !stale;
        if (canUpdate) {
            d.Inputs.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
                d.Inputs.Flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE);
            d.SourceAccelerationStructureData = d.DestAccelerationStructureData;
            ++m_stats.refits;
        } else {
            ++m_stats.builds;
        }

        const uint32_t scratchSlot = (uint32_t)(i % kScratchSlots);
        d.ScratchAccelerationStructureData =
            m_blasScratch->GetGPUVirtualAddress() + (uint64_t)scratchSlot * m_blasScratchSize;

        // Scratch is reused round-robin; a barrier once per lap is enough to keep two builds from
        // sharing a range.
        if (scratchSlot == 0 && i > 0) {
            const D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(m_blasScratch.Get());
            cl->ResourceBarrier(1, &b);
        }

        cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        m_blasBuilt[t.slot] = 1;
    }

    const D3D12_RESOURCE_BARRIER done = CD3DX12_RESOURCE_BARRIER::UAV(m_blasBuffer.Get());
    cl->ResourceBarrier(1, &done);
    ++m_frameCounter;
}

void OceanSystem::append_instances(planet::TlasBuilder& tlas, InstanceProperties* props,
                                   const planet::DVec3& sceneOrigin, uint32_t hitGroup, bool& forceRebuild) {
    if (!m_initialised || !m_params.enabled || m_gpuTiles.empty())
        return;

    const auto& tiles = m_quadtree.Tiles();
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileDesc& t = tiles[i];
        const uint32_t instID = m_propsBase + t.slot;

        const double dx = t.minX - sceneOrigin.x;
        const double dy = m_surfaceY - sceneOrigin.y;
        const double dz = t.minZ - sceneOrigin.z;
        const float xform[12] = {
            1, 0, 0, (float)dx, 0, 1, 0, (float)dy, 0, 0, 1, (float)dz,
        };
        tlas.add_instance(m_blasBuffer->GetGPUVirtualAddress() + (uint64_t)t.slot * m_blasSlotSize, xform, instID,
                          hitGroup, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE);

        if (props) {
            InstanceProperties& p = props[instID];
            const XMMATRIX M = XMMatrixTranslation((float)dx, (float)dy, (float)dz);
            const XMMATRIX Minv = XMMatrixTranslation(-(float)dx, -(float)dy, -(float)dz);
            p.objectToWorld = MakeFloat3x4(M);
            p.objectToWorldInverse = MakeFloat3x4(Minv);
            p.objectToWorldNormal = MakeFloat3x4(XMMatrixIdentity());
            p.prevObjectToWorld = MakeFloat3x4(M);
            p.indexBase = m_indexBase + t.slot * OCEAN_TILE_INDICES;
            p.vertexBase = m_vertexBase + t.slot * OCEAN_TILE_VERTS;
            p.materialBase = m_materialBase;
            p.triToLightBase = 0xFFFFFFFFu; // the ocean emits nothing
            p.opaqueTriCount = OCEAN_TILE_TRIS;
            p._pad[0] = t.stitch;
            p._pad[1] = std::bit_cast<uint32_t>((float)t.size);
            p.lightSlot = 0xFFFFFFFFu;
        }
    }
    // The tiles move with the camera and their vertices move every frame, so the top level is
    // never reusable.
    forceRebuild = true;
}

void OceanSystem::on_submitted(uint64_t copyFence, uint64_t computeFence) {
    (void)copyFence;
    // Records when this frame's coverage readback becomes safe to map.
    m_statsFence[m_frameIndex] = computeFence;
}

} // namespace ocean
