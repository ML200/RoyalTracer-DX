#include "../stdafx.h"
#include "OceanSystem.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include <bit>
#include <execution>
#include <numeric>

namespace ocean {

namespace {

constexpr uint32_t N = OCEAN_FFT_SIZE;
constexpr double kEarthRadius = 6371000.0;

constexpr double kHeightQuantile = 6.0; // sigmas per cascade in the height bound

// Stateless, so the draw at -k is reproducible for exact conjugate pairs.
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

// Must match the evolution kernel's float math (Ocean_Sim_v8.hlsl).
double ModeOmega(int32_t inx, int32_t inz, uint32_t c) {
    const float dk = 6.28318530717958f / (float)CascadeLengths()[c];
    const float kx = (float)inx * dk, kz = (float)inz * dk;
    return (double)std::sqrt(9.80665f * std::sqrt(kx * kx + kz * kz));
}

// L x L periodic value noise with derivative; quintic fade (C2).
void PeriodicValueNoise(double x, double z, uint32_t L, uint32_t seed, double& value, double& ddx, double& ddz) {
    const double sx = x * L, sz = z * L;
    const int32_t ix = (int32_t)std::floor(sx), iz = (int32_t)std::floor(sz);
    const double fx = sx - ix, fz = sz - iz;
    auto wrap = [&](int32_t i) { return (uint32_t)(((i % (int32_t)L) + (int32_t)L) % (int32_t)L); };
    auto lattice = [&](int32_t i, int32_t j) {
        return (double)(HashCoord((int32_t)wrap(i), (int32_t)wrap(j), L, seed) & 0xFFFFFFu) / (double)0xFFFFFFu * 2.0 -
               1.0;
    };
    const double v00 = lattice(ix, iz), v10 = lattice(ix + 1, iz);
    const double v01 = lattice(ix, iz + 1), v11 = lattice(ix + 1, iz + 1);

    auto fade = [](double t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); };
    auto dfade = [](double t) { return 30.0 * t * t * (t * (t - 2.0) + 1.0); };
    const double ux = fade(fx), uz = fade(fz);

    const double a = v00 + (v10 - v00) * ux;
    const double b = v01 + (v11 - v01) * ux;
    value = a + (b - a) * uz;
    ddx = ((v10 - v00) + ((v11 - v01) - (v10 - v00)) * uz) * dfade(fx) * L;
    ddz = (b - a) * dfade(fz) * L;
}

ComPtr<ID3D12Resource> CreateTexArray(ID3D12Device* dev, DXGI_FORMAT fmt, uint32_t mips, const wchar_t* name,
                                      bool unorderedAccess, uint32_t slices = OCEAN_CASCADES, uint32_t size = N) {
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = size;
    d.Height = size;
    d.DepthOrArraySize = (UINT16)slices;
    d.MipLevels = (UINT16)mips;
    d.Format = fmt;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    d.Flags = unorderedAccess ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

    // COMMON: copy-queue writes and compute reads both promote implicitly.
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
                        p.shortWaveAmplitude != m_params.shortWaveAmplitude || p.turbulence != m_params.turbulence ||
                        p.seaLevelY != m_params.seaLevelY || p.seed != m_params.seed ||
                        p.choppiness != m_params.choppiness || p.significantHeight != m_params.significantHeight ||
                        p.peakPeriod != m_params.peakPeriod || p.swellHeight != m_params.swellHeight ||
                        p.swellPeriod != m_params.swellPeriod || p.swellDirectionDeg != m_params.swellDirectionDeg ||
                        p.swellSpreadDeg != m_params.swellSpreadDeg;
    const bool refield = !m_initialised || p.turbulenceVariation != m_params.turbulenceVariation ||
                         p.turbulencePeriod != m_params.turbulencePeriod || p.seed != m_params.seed;
    m_params = p;
    if (respec)
        m_bakePending = true;
    if (refield)
        m_turbulenceBakePending = true;
}

OceanSystem::Reservation OceanSystem::GetReservation() const {
    Reservation r;
    if (!m_params.enabled)
        return r;
    r.vertexElems = TileBudget() * OCEAN_TILE_VERTS;
    r.indexElems = TileBudget() * OCEAN_TILE_INDICES;
    // One shared material, so one tile's IDs serve every tile.
    r.matIDElems = OCEAN_TILE_TRIS;
    r.instanceSlots = TileBudget();
    return r;
}

Material OceanSystem::MakeMaterial(const Params& p) {
    Material m;
    // SSS fields describe the water volume, not the surface.
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
    return m;
}

void OceanSystem::Init(ID3D12Device5* device, DeviceContext* ctx) {
    if (!m_params.enabled)
        return;
    m_device = device;
    m_ctx = ctx;

    m_h0 = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, L"OceanH0", false);
    // FFT scratch fp32; outputs fp16 (mm precision, half the fetch).
    m_fft = CreateTexArray(device, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, L"OceanFFT", true, OCEAN_CASCADES * 2);
    m_disp[0] = CreateTexArray(device, DXGI_FORMAT_R16G16B16A16_FLOAT, OCEAN_MIP_LEVELS, L"OceanDisplacementA", true);
    m_disp[1] = CreateTexArray(device, DXGI_FORMAT_R16G16B16A16_FLOAT, OCEAN_MIP_LEVELS, L"OceanDisplacementB", true);
    m_deriv = CreateTexArray(device, DXGI_FORMAT_R16G16B16A16_FLOAT, OCEAN_MIP_LEVELS, L"OceanDerivatives", true);
    m_foam[0] = CreateTexArray(device, DXGI_FORMAT_R16_FLOAT, OCEAN_FOAM_MIPS, L"OceanFoamA", true, OCEAN_FOAM_LEVELS,
                               OCEAN_FOAM_SIZE);
    m_foam[1] = CreateTexArray(device, DXGI_FORMAT_R16_FLOAT, OCEAN_FOAM_MIPS, L"OceanFoamB", true, OCEAN_FOAM_LEVELS,
                               OCEAN_FOAM_SIZE);
    // Relies on committed memory starting zeroed.
    m_foamStats = planet::create_buffer(device, OCEAN_FOAM_STATS_BYTES, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, planet::HEAP_DEFAULT);
    m_foamStats->SetName(L"OceanFoamStats");
    static_assert(sizeof(OceanFoamState) <= OCEAN_FOAM_STATE_BYTES, "OceanFoamState outgrew its slot");
    m_foamStatsReadback = planet::create_buffer(device, FRAME_COUNT * sizeof(OceanFoamState), D3D12_RESOURCE_FLAG_NONE,
                                                D3D12_RESOURCE_STATE_COPY_DEST, planet::HEAP_READBACK);

    {
        D3D12_RESOURCE_DESC d = {};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = OCEAN_TURBULENCE_SIZE;
        d.Height = OCEAN_TURBULENCE_SIZE;
        d.DepthOrArraySize = 1;
        d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        d.SampleDesc.Count = 1;
        const CD3DX12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE_DEFAULT);
        ThrowIfFailed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, D3D12_RESOURCE_STATE_COMMON,
                                                      nullptr, IID_PPV_ARGS(&m_turbulence)));
        m_turbulence->SetName(L"OceanTurbulenceField");
    }

    m_paramsBuffer = nv_helpers_dx12::CreateBuffer(device, sizeof(OceanParamsGPU), D3D12_RESOURCE_FLAG_NONE,
                                                   D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);
    m_paramsBuffer->SetName(L"OceanParams");
    m_tilesBuffer =
        nv_helpers_dx12::CreateBuffer(device, sizeof(OceanTileGPU) * TileBudget(), D3D12_RESOURCE_FLAG_NONE,
                                      D3D12_RESOURCE_STATE_COMMON, nv_helpers_dx12::kDefaultHeapProps);
    m_tilesBuffer->SetName(L"OceanTiles");

    for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
        m_paramsUpload[i] =
            nv_helpers_dx12::CreateBuffer(device, sizeof(OceanParamsGPU), D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        m_tilesUpload[i] =
            nv_helpers_dx12::CreateBuffer(device, sizeof(OceanTileGPU) * TileBudget(), D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    }

    const uint64_t rowPitch = planet::align_up((uint64_t)N * sizeof(XMFLOAT4), D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    m_bakeUpload = nv_helpers_dx12::CreateBuffer(device, rowPitch * N * OCEAN_CASCADES, D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    m_bakeUpload->SetName(L"OceanBakeUpload");

    // BLAS pool; identical topology, so one size query covers every tile.
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
        // Moves every frame: fast build, refittable.
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
        m_blasScratchSize = planet::align_up(std::max(info.ScratchDataSizeInBytes, info.UpdateScratchDataSizeInBytes),
                                             D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);

        m_blasBuffer = planet::create_buffer(device, m_blasSlotSize * TileBudget(),
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                             planet::HEAP_DEFAULT);
        m_blasBuffer->SetName(L"OceanBlasPool");
        m_blasScratch = planet::create_buffer(device, m_blasScratchSize * TileBudget(),
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS, planet::HEAP_DEFAULT);
        m_blasScratch->SetName(L"OceanBlasScratch");

        m_stats.blasBytes = m_blasSlotSize * TileBudget();
        LOG(L"[ocean] tiles=" << TileBudget() << L" of " << OCEAN_MAX_TILES << L" tris/tile=" << OCEAN_TILE_TRIS
                              << L" blas=" << (m_blasSlotSize >> 10) << L" KiB each, pool "
                              << (m_stats.blasBytes >> 20) << L" MiB; scratch " << (m_blasScratchSize >> 10)
                              << L" KiB x " << TileBudget());
    }

    m_blasBuilt.assign(TileBudget(), 0);
    m_blasRebuiltAt.assign(TileBudget(), -kBlasRebuildSeconds);

    {
        CD3DX12_ROOT_PARAMETER1 rp[1];
        rp[0].InitAsConstants(8, 0, 0, D3D12_SHADER_VISIBILITY_ALL);

        CD3DX12_STATIC_SAMPLER_DESC samp[1];
        samp[0].Init(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR, D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                     D3D12_TEXTURE_ADDRESS_MODE_WRAP, D3D12_TEXTURE_ADDRESS_MODE_WRAP);

        CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
        // Static sampler: no sampler heap on the streaming queue.
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
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanFftH", m_psoFftH);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanFftV", m_psoFftV);
    makePso(L"Ocean_Sim_v8.hlsl", L"OceanMip", m_psoMip);
    makePso(L"Ocean_Tiles_v8.hlsl", L"OceanTiles", m_psoTiles);
    makePso(L"Ocean_Tiles_v8.hlsl", L"OceanFoam", m_psoFoam);
    makePso(L"Ocean_Tiles_v8.hlsl", L"OceanFoamMip", m_psoFoamMip);
    makePso(L"Ocean_Tiles_v8.hlsl", L"OceanFoamStats", m_psoFoamStats);

    // Always timed: absent from per-pass profiles (streaming queue).
    {
        D3D12_QUERY_HEAP_DESC q{};
        q.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        q.Count = FRAME_COUNT * kTimestamps;
        ThrowIfFailed(device->CreateQueryHeap(&q, IID_PPV_ARGS(&m_timestamps)));
        m_timestampReadback = planet::create_buffer(device, FRAME_COUNT * kTimestamps * sizeof(uint64_t),
            D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, planet::HEAP_READBACK);
        ThrowIfFailed(ctx->PlanetComputeQueue()->GetTimestampFrequency(&m_timestampFrequency));
    }
    if (const char* path = std::getenv("RT_OCEAN_PERF_CSV")) {
        m_profile.open(path);
        m_profile << "frame,fft_ms,mips_ms,mesh_ms,blas_ms,total_ms,tiles,triangles,allocated_bytes\n";
    }
    // Committed sizes; the geometry reservation is added below.
    ID3D12Resource* resources[] = {m_h0.Get(), m_fft.Get(), m_disp[0].Get(), m_disp[1].Get(), m_deriv.Get(), m_foam[0].Get(), m_foam[1].Get(),
        m_foamStats.Get(), m_foamStatsReadback.Get(), m_turbulence.Get(), m_bakeUpload.Get(),
        m_paramsBuffer.Get(), m_tilesBuffer.Get(), m_blasBuffer.Get(), m_blasScratch.Get(), m_timestampReadback.Get()};
    m_stats.resourceBytes = 0;
    auto account = [&](ID3D12Resource* r) {
        if (r) {
            const auto d = r->GetDesc();
            m_stats.resourceBytes += device->GetResourceAllocationInfo(0, 1, &d).SizeInBytes;
        }
    };
    for (auto* r : resources) account(r);
    for (uint32_t i = 0; i < FRAME_COUNT; ++i) {
        account(m_paramsUpload[i].Get());
        account(m_tilesUpload[i].Get());
    }
    const auto reservation = GetReservation();
    m_stats.resourceBytes += uint64_t(reservation.vertexElems) * sizeof(BTriVertex) + uint64_t(reservation.indexElems) * 4;
    LOG(L"[ocean] resources including reserved geometry=" << (m_stats.resourceBytes >> 20) << L" MiB");

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

    // Rebound on the streaming queue; heap bindings don't carry across lists.
    m_srvHeap = heap;

    const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto at = [&](uint32_t slot) {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(heap->GetCPUDescriptorHandleForHeapStart(), (INT)slot, inc);
    };

    auto texSrv = [&](ID3D12Resource* res, uint32_t mips, uint32_t slot, uint32_t slices = OCEAN_CASCADES) {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Format = res->GetDesc().Format;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
        s.Texture2DArray.MipLevels = mips;
        s.Texture2DArray.ArraySize = slices;
        device->CreateShaderResourceView(res, &s, at(slot));
    };
    auto texUav = [&](ID3D12Resource* res, uint32_t mip, uint32_t slot, uint32_t slices = OCEAN_CASCADES) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = res->GetDesc().Format;
        u.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
        u.Texture2DArray.MipSlice = mip;
        u.Texture2DArray.ArraySize = slices;
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

    texSrv(m_disp[0].Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_DISP0);
    texSrv(m_disp[1].Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_DISP1);
    texSrv(m_deriv.Get(), OCEAN_MIP_LEVELS, OCEAN_SRV_DERIV);
    texSrv(m_foam[0].Get(), OCEAN_FOAM_MIPS, OCEAN_SRV_FOAM0, OCEAN_FOAM_LEVELS);
    texSrv(m_foam[1].Get(), OCEAN_FOAM_MIPS, OCEAN_SRV_FOAM1, OCEAN_FOAM_LEVELS);
    texSrv(m_h0.Get(), 1, OCEAN_SRV_H0);
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC s = {};
        s.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Format = m_turbulence->GetDesc().Format;
        s.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        s.Texture2D.MipLevels = 1;
        device->CreateShaderResourceView(m_turbulence.Get(), &s, at(OCEAN_SRV_TURBULENCE));
    }
    structuredSrv(m_paramsBuffer.Get(), 1, sizeof(OceanParamsGPU), OCEAN_SRV_PARAMS);
    structuredSrv(m_tilesBuffer.Get(), TileBudget(), sizeof(OceanTileGPU), OCEAN_SRV_TILES);

    texUav(m_fft.Get(), 0, OCEAN_UAV_FFT, OCEAN_CASCADES * 2);
    for (uint32_t m = 0; m < OCEAN_MIP_LEVELS; ++m) {
        texUav(m_disp[0].Get(), m, OCEAN_UAV_DISP0_MIPS + m);
        texUav(m_disp[1].Get(), m, OCEAN_UAV_DISP1_MIPS + m);
        texUav(m_deriv.Get(), m, OCEAN_UAV_DERIV_MIPS + m);
    }
    for (uint32_t m = 0; m < OCEAN_FOAM_MIPS; ++m) {
        texUav(m_foam[0].Get(), m, OCEAN_UAV_FOAM0_MIPS + m, OCEAN_FOAM_LEVELS);
        texUav(m_foam[1].Get(), m, OCEAN_UAV_FOAM1_MIPS + m, OCEAN_FOAM_LEVELS);
    }
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = DXGI_FORMAT_R32_TYPELESS;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = OCEAN_FOAM_STATS_BYTES / 4;
        u.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device->CreateUnorderedAccessView(m_foamStats.Get(), nullptr, &u, at(OCEAN_UAV_FOAM_STATS));
    }

    // Whole global vertex buffer, indexed like the hit evaluator.
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC u = {};
        u.Format = DXGI_FORMAT_UNKNOWN;
        u.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        u.Buffer.NumElements = VertexSpan();
        u.Buffer.StructureByteStride = sizeof(BTriVertex);
        device->CreateUnorderedAccessView(m_globalVertex, nullptr, &u, at(OCEAN_UAV_VERTS));
    }
}

// Tiling fBm of the wave-field gain, with its gradient.
void OceanSystem::BakeTurbulence() {
    constexpr uint32_t S = OCEAN_TURBULENCE_SIZE;
    constexpr int kOctaves = 5;

    m_turbulenceBakePending = false;
    m_turbulenceDirty = true;
    m_turbulenceData.assign((size_t)S * S, XMFLOAT4{1.0f, 0.0f, 0.0f, 0.0f});

    const double variation = std::clamp((double)m_params.turbulenceVariation, 0.0, 0.9);
    if (variation <= 0.0)
        return; // uniform sea

    const double period = std::max(64.0, (double)m_params.turbulencePeriod);

    std::vector<double> raw((size_t)S * S * 3);
    double lo = 1e30, hi = -1e30, mean = 0.0;
    for (uint32_t j = 0; j < S; ++j) {
        for (uint32_t i = 0; i < S; ++i) {
            const double u = ((double)i + 0.5) / S, v = ((double)j + 0.5) / S;
            double sum = 0.0, ddu = 0.0, ddv = 0.0, amplitude = 1.0, total = 0.0;
            for (int o = 0; o < kOctaves; ++o) {
                double n, nu, nv;
                PeriodicValueNoise(u, v, 2u << o, m_params.seed + (uint32_t)o * 7919u, n, nu, nv);
                sum += amplitude * n;
                ddu += amplitude * nu;
                ddv += amplitude * nv;
                total += amplitude;
                amplitude *= 0.5;
            }
            const size_t idx = ((size_t)j * S + i) * 3;
            raw[idx] = sum / total;
            raw[idx + 1] = ddu / total;
            raw[idx + 2] = ddv / total;
            lo = std::min(lo, raw[idx]);
            hi = std::max(hi, raw[idx]);
            mean += raw[idx];
        }
    }
    mean /= (double)S * S;

    // Realised spread maps to the variation; mean-centred keeps the mean gain 1.
    const double spread = std::max(1e-6, std::max(hi - mean, mean - lo));
    const double gradientScale = variation / spread / period; // per metre
    for (size_t t = 0; t < (size_t)S * S; ++t) {
        m_turbulenceData[t] = XMFLOAT4{(float)(1.0 + variation * (raw[t * 3] - mean) / spread),
                                       (float)(raw[t * 3 + 1] * gradientScale),
                                       (float)(raw[t * 3 + 2] * gradientScale), 0.0f};
    }
}

void OceanSystem::UploadTurbulence(ID3D12GraphicsCommandList* copyList) {
    constexpr uint32_t S = OCEAN_TURBULENCE_SIZE;
    const uint64_t rowBytes = (uint64_t)S * sizeof(XMFLOAT4);
    const uint64_t rowPitch = planet::align_up(rowBytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const uint64_t needed = rowPitch * S;

    if (!m_turbulenceUpload) {
        m_turbulenceUpload = nv_helpers_dx12::CreateBuffer(m_device, needed, D3D12_RESOURCE_FLAG_NONE,
                                                           D3D12_RESOURCE_STATE_GENERIC_READ,
                                                           nv_helpers_dx12::kUploadHeapProps);
        m_turbulenceUpload->SetName(L"OceanTurbulenceUpload");
    }

    uint8_t* base = nullptr;
    const CD3DX12_RANGE none(0, 0);
    ThrowIfFailed(m_turbulenceUpload->Map(0, &none, (void**)&base));
    for (uint32_t row = 0; row < S; ++row)
        std::memcpy(base + (uint64_t)row * rowPitch, m_turbulenceData.data() + (size_t)row * S, rowBytes);
    m_turbulenceUpload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
    dstLoc.pResource = m_turbulence.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
    srcLoc.pResource = m_turbulenceUpload.Get();
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Offset = 0;
    srcLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srcLoc.PlacedFootprint.Footprint.Width = S;
    srcLoc.PlacedFootprint.Footprint.Height = S;
    srcLoc.PlacedFootprint.Footprint.Depth = 1;
    srcLoc.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
    copyList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    m_turbulenceDirty = false;
}

void OceanSystem::Bake() {
    if (!m_initialised || !m_params.enabled)
        return;
    const auto t0 = std::chrono::high_resolution_clock::now();

    m_spectrum.Init(m_params);
    m_chopBand = ShortChopBand(m_spectrum.omegaP);
    const ChopBand chopBand = m_chopBand;
    const double shortChop = ShortWaveChop(m_params);

    m_h0Data.assign((size_t)N * N * OCEAN_CASCADES, XMFLOAT4{0, 0, 0, 0});
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        m_cascadeVariance[c] = 0.0;
        m_cascadeMeanK[c] = 0.0;
    }

    const double windRad = (double)m_params.windDirectionDeg * 0.017453292519943295;
    const double wc = std::cos(windRad), ws = std::sin(windRad);
    const double amp = std::max(0.0, (double)m_params.amplitudeScale);

    double sumAlong = 0.0, sumCross = 0.0;
    double elevationVar = 0.0;
    double strainMs = 0.0;
    double kEnergy[OCEAN_CASCADES] = {};

    // Rows baked in parallel, summed in a fixed order for determinism.
    struct RowSums {
        double power = 0.0, kEnergy = 0.0, along = 0.0, cross = 0.0, strain = 0.0;
        double residual[OCEAN_ROUGHNESS_ENTRIES] = {};
    };
    std::vector<RowSums> rows((size_t)OCEAN_CASCADES * N);
    std::vector<uint32_t> rowIndex(rows.size());
    std::iota(rowIndex.begin(), rowIndex.end(), 0u);

    // Gaussian h0 = 1/2 sqrt(PSD dk^2), Hermitian pairs (Tessendorf 2001).
    std::for_each(std::execution::par, rowIndex.begin(), rowIndex.end(), [&](uint32_t row) {
        const uint32_t c = row / N, m = row % N;
        const double L = CascadeLengths()[c];
        const double dk = 6.283185307179586 / L;
        RowSums& sums = rows[row];
        for (uint32_t n = 0; n < N; ++n) {
            const int32_t inx = (int32_t)n - (int32_t)(N / 2);
            const int32_t inz = (int32_t)m - (int32_t)(N / 2);
            const double kx = (double)inx * dk;
            const double kz = (double)inz * dk;
            const double k = std::sqrt(kx * kx + kz * kz);

            // DC and the ambiguous Nyquist row/column stay zero.
            if (k < 1e-6 || n == 0 || m == 0)
                continue;
            // Cascade share of power; amplitude takes its sqrt.
            const double weight = CascadeWeight((int)c, k);
            if (weight <= 1e-6)
                continue;
            const double ampWeight = std::sqrt(weight);

            // Wind frame.
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

            const size_t idx = ((size_t)c * N + m) * N + n;
            m_h0Data[idx] = XMFLOAT4{(float)(A * g0), (float)(A * g1), (float)(An * h0), (float)(-An * h1)};

            const double power = 4.0 * A * A;
            sums.power += power;
            sums.kEnergy += power * k;
            sums.along += kxw * kxw * power;
            sums.cross += kzw * kzw * power;
            // Mode strain k_a k_b / k * h: squared norm k^2 |h|^2.
            const double chop = ChopGainAt(k, chopBand, shortChop);
            sums.strain += power * chop * chop * k * k;
            // Slope share a width-w box removes: 1 - exp(-k^2 w^2 / 12); w doubles per entry.
            const double w0 = std::exp2((double)OCEAN_ROUGHNESS_OFFSET);
            double kept = std::exp(-k * k * w0 * w0 / 12.0);
            for (int i = 0; i < OCEAN_ROUGHNESS_ENTRIES; ++i) {
                sums.residual[i] += power * k * k * (1.0 - kept);
                kept *= kept;
                kept *= kept;
            }
        }
    });
    double residual[OCEAN_ROUGHNESS_ENTRIES] = {};
    for (size_t row = 0; row < rows.size(); ++row) {
        const uint32_t c = (uint32_t)(row / N);
        elevationVar += rows[row].power;
        m_cascadeVariance[c] += rows[row].power;
        kEnergy[c] += rows[row].kEnergy;
        sumAlong += rows[row].along;
        sumCross += rows[row].cross;
        strainMs += rows[row].strain;
        for (int i = 0; i < OCEAN_ROUGHNESS_ENTRIES; ++i)
            residual[i] += rows[row].residual[i];
    }

    // Unresolved tail up to the gravity-capillary crossover, as box-filtered roughness.
    {
        const double k0 = CascadeNyquist(OCEAN_CASCADES - 1), k1 = 370.0;
        const double dlog = std::max(0.0, std::log(k1 / k0)) / 256.0;
        const double w0 = std::exp2((double)OCEAN_ROUGHNESS_OFFSET);
        for (int i = 0; i < 256 && dlog > 0.0; ++i) {
            const double k = k0 * std::exp((i + 0.5) * dlog);
            double slope = 0.0;
            for (int j = 0; j < 64; ++j) {
                const double theta = (j + 0.5) * (6.283185307179586 / 64.0);
                slope += m_spectrum.S2D(k * std::cos(theta), k * std::sin(theta)) * k * k * k * k * dlog *
                         (6.283185307179586 / 64.0) * amp * amp;
            }
            double kept = std::exp(-k * k * w0 * w0 / 12.0);
            for (int e = 0; e < OCEAN_ROUGHNESS_ENTRIES; ++e) {
                residual[e] += slope * (1.0 - kept);
                kept *= kept;
                kept *= kept;
            }
        }
    }
    for (int i = 0; i < OCEAN_ROUGHNESS_ENTRIES; ++i)
        m_residualSlope[i] = (float)residual[i];

    m_strainRms = std::sqrt(strainMs);
    m_phaseEpoch = 0.0;
    m_historyFrames = 0;

    double cmAlong = 0.0, cmCross = 0.0;
    CoxMunkSlopeVariance(m_params.windSpeed, cmAlong, cmCross);
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c)
        m_cascadeMeanK[c] = m_cascadeVariance[c] > 1e-12 ? kEnergy[c] / m_cascadeVariance[c] : 0.0;

    m_stats.slopeVarSpectrum = sumAlong + sumCross;
    m_stats.slopeVarCoxMunk = cmAlong + cmCross;

    // Analytic, to match what scenes query before the bake.
    m_waveDepth = PredictWaveDepth(m_params);
    m_surfaceY = PredictSurfaceLevel(m_params);
    m_stats.surfaceY = m_surfaceY;
    m_stats.significantWaveHeight = 4.0 * std::sqrt(std::max(0.0, elevationVar));
    m_stats.bakeMs =
        std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

    LOG(L"[ocean] sea level " << m_surfaceY << L" m, deepest trough " << (m_surfaceY - m_waveDepth) << L" m");
    LOG(L"[ocean] wind=" << m_params.windSpeed << L" m/s fetch=" << (m_params.fetch / 1000.0f) << L" km Hs="
                         << m_stats.significantWaveHeight << L" m  slope var: spectrum="
                         << m_stats.slopeVarSpectrum << L" Cox-Munk=" << m_stats.slopeVarCoxMunk << L" strain rms="
                         << m_strainRms << L" (bake " << m_stats.bakeMs << L" ms)");
    auto alphaAt = [&](int log2Width) { return std::sqrt(m_residualSlope[log2Width - OCEAN_ROUGHNESS_OFFSET]); };
    LOG(L"[ocean] Tp=" << (6.283185307179586 / m_spectrum.omegaP) << L" s gamma=" << m_spectrum.gamma
                       << L" equilibrium gain " << m_spectrum.equilibriumGain << L" footprint alpha: 1 mm " << alphaAt(-10) << L", 1 cm " << alphaAt(-7) << L", 12 cm "
                       << alphaAt(-3) << L", 1 m "
                       << alphaAt(0) << L", 8 m " << alphaAt(3) << L"; short-wave chop +" << ShortWaveChop(m_params)
                       << L" from " << (6.283185307179586 / std::exp2(m_chopBand.lo)) << L" m waves");

    m_bakePending = false;
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
    if (m_turbulenceBakePending)
        BakeTurbulence();

    // Fold whole epochs into h0 in double; the modes share no common period.
    const double epoch = std::floor(m_time / 128.0) * 128.0;
    if (epoch != m_phaseEpoch) {
        const double elapsed = epoch - m_phaseEpoch;
        for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
            for (uint32_t m = 0; m < N; ++m) {
                for (uint32_t n = 0; n < N; ++n) {
                    auto& h = m_h0Data[((size_t)c * N + m) * N + n];
                    if (h.x == 0.0f && h.y == 0.0f && h.z == 0.0f && h.w == 0.0f)
                        continue;
                    const double omega = ModeOmega((int32_t)n - (int32_t)(N / 2), (int32_t)m - (int32_t)(N / 2), c);
                    const double phase = std::remainder(omega * elapsed, 6.283185307179586);
                    const double cs = std::cos(phase), sn = std::sin(phase);
                    h = {(float)(h.x * cs + h.y * sn), (float)(h.y * cs - h.x * sn),
                         (float)(h.z * cs - h.w * sn), (float)(h.w * cs + h.z * sn)};
                }
            }
        }
        m_phaseEpoch = epoch;
        m_h0Dirty = true;
    }

    // Oldest frame in flight, once retired.
    {
        const uint32_t slot = (m_frameIndex + 1u) % FRAME_COUNT;
        if (m_timestamps && m_timestampFence[slot] != 0 && m_ctx->PlanetComputeCompleted() >= m_timestampFence[slot]) {
            const uint64_t* ticks = nullptr;
            const CD3DX12_RANGE range(slot * kTimestamps * sizeof(uint64_t), (slot + 1) * kTimestamps * sizeof(uint64_t));
            ThrowIfFailed(m_timestampReadback->Map(0, &range, (void**)&ticks));
            ticks += slot * kTimestamps;
            const double toMs = 1000.0 / (double)std::max<uint64_t>(m_timestampFrequency, 1);
            for (uint32_t i = 1; i < kTimestamps; ++i)
                m_stats.gpuStageMs[i - 1] = (float)(double(ticks[i] - ticks[i - 1]) * toMs);
            m_stats.gpuTotalMs = (float)(double(ticks[kTimestamps - 1] - ticks[0]) * toMs);
            const CD3DX12_RANGE none(0, 0);
            m_timestampReadback->Unmap(0, &none);
            const OceanFoamState* foam = nullptr;
            const CD3DX12_RANGE foamRange(slot * sizeof(OceanFoamState), (slot + 1) * sizeof(OceanFoamState));
            ThrowIfFailed(m_foamStatsReadback->Map(0, &foamRange, (void**)&foam));
            foam += slot;
            if (foam->valid != 0u) {
                for (uint32_t l = 0; l < OCEAN_FOAM_LEVELS; ++l) {
                    m_stats.foamCoverage[l] = foam->coverage[l];
                    m_stats.foamThreshold[l] = foam->threshold[l];
                    m_stats.foamMatch[l] = foam->match[l];
                }
                m_stats.foamWhite = foam->white;
                m_stats.foamBreaking = foam->breaking;
            }
            m_foamStatsReadback->Unmap(0, &none);
            if (m_profile.is_open()) {
                m_profile << m_profileFrame++;
                for (uint32_t i = 0; i < kGpuStages; ++i) m_profile << ',' << m_stats.gpuStageMs[i];
                m_profile << ',' << m_stats.gpuTotalMs << ',' << m_stats.tiles << ',' << m_stats.triangles << ','
                          << m_stats.resourceBytes << '\n';
                m_profile.flush();
            }
            m_timestampFence[slot] = 0;
        }
    }

    // Distance where Hs spans one pixel of a 1080-line image.
    const double pixelAngle = std::max(1e-5, (double)cam.fov_y / 1080.0);
    const double silhouette = std::max(0.02, m_stats.significantWaveHeight) / pixelAngle;

    Params selectParams = m_params;
    selectParams.seaLevelY = (float)m_surfaceY;
    m_quadtree.Select(selectParams, cam, TileBudget(), m_coverage, silhouette);

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
        // Curvature drop |a + p|^2 / 2R, expanded about the anchor for float precision.
        g.curveBase = (float)((t.minX * t.minX + t.minZ * t.minZ) * 0.5 * invR);
        g.curveGrad = XMFLOAT2((float)(t.minX * invR), (float)(t.minZ * invR));
        g.invCurveRadius = (float)invR;
        m_gpuTiles.push_back(g);
    }

    m_stats.tiles = (uint32_t)m_gpuTiles.size();
    m_stats.leaves = m_quadtree.SelectedLeafCount();
    m_stats.dropped = m_quadtree.DroppedCount();
    m_stats.triangles = (uint64_t)m_gpuTiles.size() * OCEAN_TILE_TRIS;

    OceanParamsGPU& P = m_gpuParams;
    P.surfaceY = (float)(m_surfaceY - m_sceneOrigin.y);
    P.filterScale = m_params.filterScale;
    P.sunLobeRoughness = std::clamp(m_params.sunLobeRoughness, 0.0f, 1.0f);
    P.invRadius = (float)invR;
    P.curveOrigin = {(float)m_sceneOrigin.x, (float)m_sceneOrigin.z};
    P.materialBase = m_materialIndex;
    P.originDelta = {(float)(m_sceneOrigin.x - m_previousOrigin.x), (float)(m_sceneOrigin.y - m_previousOrigin.y),
                     (float)(m_sceneOrigin.z - m_previousOrigin.z)};
    P.historyValid = m_historyFrames > 0 ? 1.0f : 0.0f;
    P.halfExtent = std::max(64.0f, m_params.extent);
    P.debugMode = m_params.debugMode;
    P.dispParity = m_parity;
    P.pad = 0;

    // Selection's LOD rule, for the tessellator's filter width.
    {
        Params ruleParams = m_params;
        ruleParams.lodFactor = (float)m_quadtree.EffectiveLodFactor();
        const LodRule rule(ruleParams, cam, silhouette);
        auto toF3 = [](const double* v) { return XMFLOAT3((float)v[0], (float)v[1], (float)v[2]); };
        P.lodCamera = {(float)(cam.position_world.x - m_sceneOrigin.x), (float)(cam.position_world.y - m_sceneOrigin.y),
                       (float)(cam.position_world.z - m_sceneOrigin.z)};
        P.lodRatio = (float)(rule.ratio * 0.75 / OCEAN_TILE_GRID); // quads span 0.5..1x the ratio
        P.lodForward = toF3(rule.view.F);
        P.lodRight = toF3(rule.view.R);
        P.lodUp = toF3(rule.view.U);
        P.lodTanH = (float)rule.view.tanH;
        P.lodTanV = (float)rule.view.tanV;
        P.lodOffscreen = (float)rule.offscreen;
        P.lodNearKeep = (float)rule.nearKeep;
        P.lodSilhouette = (float)rule.silhouette;
        P.lodMinWidth = std::max(1.0f, m_params.minTileSize) / OCEAN_TILE_GRID;
        P.lodPad = 0.0f;
    }

    for (int i = 0; i < OCEAN_ROUGHNESS_ENTRIES; ++i)
        ((float*)&P.residualSlope[i / 4])[i % 4] = m_residualSlope[i];

    // Whitecaps, after Crest: foam where the surface breaks, steered to WhitecapCover.
    {
        const double decay = std::clamp((double)m_params.foamDecay, 0.01, 0.99);
        P.foamDecayRate = (float)(-std::log(decay));
        P.foamStrength = m_params.foamCoverage > 0.0f ? 1.0f : 0.0f;
        P.foamCover = (float)WhitecapCover(m_params);
        m_stats.foamTarget = P.foamCover;
        P.foamBreakMax = (float)BreakingThreshold(m_params);
        // Steer at the fresh-foam fade rate, so the loop stays damped.
        P.foamSteer = P.foamDecayRate * OCEAN_FOAM_FRESH_DECAY;
        P.foamRate = 5.0f; // rafts per second of full breaking
        P.frameIndex = m_frameCounter++;
        P.foamParity = m_foamParity;
        const double camWorld[2] = {cam.position_world.x, cam.position_world.z};
        const double sceneOrigin[2] = {m_sceneOrigin.x, m_sceneOrigin.z};
        bool history = m_foamHistory;
        // Levels snap to whole world texels, so history shifts exactly.
        for (uint32_t l = 0; l < OCEAN_FOAM_LEVELS; ++l) {
            const double texel = (double)OCEAN_FOAM_TEXEL0 * (double)(1u << l);
            float origin[2], shift[2];
            int32_t index[2];
            for (int a = 0; a < 2; ++a) {
                const int64_t cell = (int64_t)std::floor(camWorld[a] / texel) - OCEAN_FOAM_SIZE / 2;
                const int64_t moved = cell - m_foamCell[l][a];
                history = history && std::llabs(moved) < OCEAN_FOAM_SIZE;
                shift[a] = (float)moved;
                origin[a] = (float)((double)cell * texel - sceneOrigin[a]);
                index[a] = (int32_t)cell; // hashed; int32 reaches 1e5 km on the finest level
                m_foamCell[l][a] = cell;
            }
            P.foamLevel[l] = {origin[0], origin[1], shift[0], shift[1]};
            P.foamCell[l] = {index[0], index[1], 0, 0};
        }
        P.foamHistory = history ? 1u : 0u;
        const double bearing = (double)m_params.windDirectionDeg * 0.017453292519943295;
        P.foamWind = {(float)std::sin(bearing), (float)std::cos(bearing)};
        const double shortChop = ShortWaveChop(m_params);
        P.chopBand = {(float)m_chopBand.lo, (float)m_chopBand.hi, (float)m_chopBand.cut, (float)shortChop};
    }

    P.turbulencePeriod = std::max(64.0f, m_params.turbulencePeriod);
    P.turbulenceStrength = std::clamp(m_params.turbulenceVariation, 0.0f, 0.9f) > 0.0f ? 1.0f : 0.0f;
    const double maxGain = TurbulenceMaxGain(m_params);

    // m_strainRms was baked with the same chop profile.
    P.choppiness = std::max(0.0f, m_params.choppiness);
    m_stats.horizontalGain = (float)(P.choppiness * (1.0 + ShortWaveChop(m_params)));
    m_stats.crestStrain = (float)(P.choppiness * m_strainRms);

    // Skew warp is monotonic past its turn, so the bounds sit at eta's extremes.
    m_stats.crestSteepness = 0.0;
    double crest = 0.0, trough = 0.0;
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        ((float*)&P.cascadeLength)[c] = (float)CascadeLengths()[c];
        const double sigma = std::sqrt(std::max(0.0, m_cascadeVariance[c]));
        const double skew = CrestSkew(m_params, m_cascadeMeanK[c], sigma,
                                      ChopGainAt(m_cascadeMeanK[c], m_chopBand, ShortWaveChop(m_params)));
        ((float*)&P.crestSkew)[c] = (float)skew;
        ((float*)&P.cascadeVariance)[c] = (float)m_cascadeVariance[c];
        m_stats.crestSteepness = std::max(m_stats.crestSteepness, skew * sigma);
        const double e = kHeightQuantile * sigma * maxGain;
        crest += e + skew * e * e;
        trough += e + skew * m_cascadeVariance[c] * maxGain * maxGain;
    }
    // Margin covers fp16 rounding only.
    P.crestHeight = (float)(crest * 1.01 + 0.01);
    P.troughDepth = (float)(trough * 1.01 + 0.01);
    m_stats.crestHeight = P.crestHeight;
    m_stats.troughDepth = P.troughDepth;

    // Origin wrapped per cascade: world-pinned waves, small shader coordinates.
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
    return m_params.enabled ? TileBudget() : 0u;
}

void OceanSystem::record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) {
    if (!m_initialised || !m_params.enabled || !m_srvHeap)
        return;

    // The orchestrator binds no heap of its own.
    ID3D12DescriptorHeap* heaps[] = {m_srvHeap};
    computeList->SetDescriptorHeaps(1, heaps);

    // Copy queue; compute waits on it.
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
    if (m_turbulenceDirty)
        UploadTurbulence(copyList);

    Timestamp(computeList, 0);
    RecordSimulation(computeList);
    RecordFoam(computeList);
    RecordTessellation(computeList);
    Timestamp(computeList, 3);
    RecordAccelerationStructures(computeList);
    Timestamp(computeList, 4);
    if (m_timestamps)
        computeList->ResolveQueryData(m_timestamps.Get(), D3D12_QUERY_TYPE_TIMESTAMP, m_frameIndex * kTimestamps,
                                      kTimestamps, m_timestampReadback.Get(),
                                      m_frameIndex * kTimestamps * sizeof(uint64_t));

    m_parity ^= 1u;
    ++m_historyFrames;
    m_foamParity ^= 1u;
    m_foamHistory = true;
}

void OceanSystem::UploadBaked(ID3D12GraphicsCommandList* copyList) {
    const uint64_t rowBytes = (uint64_t)N * sizeof(XMFLOAT4);
    const uint64_t rowPitch = planet::align_up(rowBytes, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    const uint64_t sliceBytes = rowPitch * N;

    uint8_t* base = nullptr;
    const CD3DX12_RANGE none(0, 0);
    ThrowIfFailed(m_bakeUpload->Map(0, &none, (void**)&base));
    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c)
        for (uint32_t row = 0; row < N; ++row)
            std::memcpy(base + (uint64_t)c * sliceBytes + row * rowPitch,
                        m_h0Data.data() + ((size_t)c * N + row) * N, rowBytes);
    m_bakeUpload->Unmap(0, nullptr);

    for (uint32_t c = 0; c < OCEAN_CASCADES; ++c) {
        D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
        dstLoc.pResource = m_h0.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = c; // single mip: slice == subresource

        D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
        srcLoc.pResource = m_bakeUpload.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint.Offset = (uint64_t)c * sliceBytes;
        srcLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        srcLoc.PlacedFootprint.Footprint.Width = N;
        srcLoc.PlacedFootprint.Footprint.Height = N;
        srcLoc.PlacedFootprint.Footprint.Depth = 1;
        srcLoc.PlacedFootprint.Footprint.RowPitch = (UINT)rowPitch;
        copyList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    }
    m_h0Dirty = false;
}

void OceanSystem::Transition(ID3D12GraphicsCommandList* cl, ID3D12Resource* r, D3D12_RESOURCE_STATES& state,
                             D3D12_RESOURCE_STATES next) {
    if (state == next)
        return;
    const D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(r, state, next);
    cl->ResourceBarrier(1, &b);
    state = next;
}

void OceanSystem::RecordSimulation(ID3D12GraphicsCommandList4* cl) {
    struct Push {
        uint32_t u0, u1, u2, u3;
        float f0, f1, f2, f3;
    } push{};

    cl->SetComputeRootSignature(m_rootSig.Get());

    auto uavBarrier = [&](std::initializer_list<ID3D12Resource*> resources) {
        D3D12_RESOURCE_BARRIER bs[3];
        UINT n = 0;
        for (ID3D12Resource* r : resources) bs[n++] = CD3DX12_RESOURCE_BARRIER::UAV(r);
        cl->ResourceBarrier(n, bs);
    };

    const uint32_t cur = m_parity;
    ID3D12Resource* disp = m_disp[cur].Get();
    Transition(cl, disp, m_dispState[cur], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Transition(cl, m_deriv.Get(), m_derivState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    push.f0 = (float)(m_time - m_phaseEpoch);
    push.f1 = m_dt;
    cl->SetPipelineState(m_psoFftH.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(N, OCEAN_CASCADES, 1);
    uavBarrier({m_fft.Get()});

    cl->SetPipelineState(m_psoFftV.Get());
    cl->Dispatch(N, OCEAN_CASCADES, 1);
    uavBarrier({disp, m_deriv.Get()});
    Timestamp(cl, 1);

    cl->SetPipelineState(m_psoMip.Get());
    push.u0 = cur;
    for (uint32_t level = 1; level < OCEAN_MIP_LEVELS; ++level) {
        const uint32_t size = std::max(1u, N >> level);
        push.u1 = level;
        cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
        cl->Dispatch((size + 7) / 8, (size + 7) / 8, OCEAN_CASCADES);
        uavBarrier({disp, m_deriv.Get()});
    }
    Timestamp(cl, 2);

    // Compute-legal read state, history array included.
    Transition(cl, disp, m_dispState[cur], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(cl, m_deriv.Get(), m_derivState, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(cl, m_disp[cur ^ 1u].Get(), m_dispState[cur ^ 1u], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}

// After RecordSimulation; reads last frame's foam, writes this frame's.
void OceanSystem::RecordFoam(ID3D12GraphicsCommandList4* cl) {
    struct Push {
        uint32_t u0, u1, u2, u3;
        float f0, f1, f2, f3;
    } push{};
    push.f0 = (float)m_time;
    push.f1 = m_dt;
    const uint32_t cur = m_foamParity, prev = m_foamParity ^ 1u;
    Transition(cl, m_foam[prev].Get(), m_foamState[prev], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(cl, m_foam[cur].Get(), m_foamState[cur], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    cl->SetComputeRootSignature(m_rootSig.Get());
    cl->SetPipelineState(m_psoFoam.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    constexpr uint32_t groups = (OCEAN_FOAM_SIZE + 7) / 8;
    cl->Dispatch(groups, groups, OCEAN_FOAM_LEVELS);
    const D3D12_RESOURCE_BARRIER uav = CD3DX12_RESOURCE_BARRIER::UAV(m_foam[cur].Get());
    cl->ResourceBarrier(1, &uav);

    cl->SetPipelineState(m_psoFoamMip.Get());
    for (uint32_t level = 1; level < OCEAN_FOAM_MIPS; ++level) {
        const uint32_t size = std::max(1u, (uint32_t)OCEAN_FOAM_SIZE >> level);
        push.u1 = level;
        cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
        cl->Dispatch((size + 7) / 8, (size + 7) / 8, OCEAN_FOAM_LEVELS);
        cl->ResourceBarrier(1, &uav);
    }

    // Next frame's breaking points, from this frame's histograms and mips.
    Transition(cl, m_foamStats.Get(), m_foamStatsState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const D3D12_RESOURCE_BARRIER statsUav = CD3DX12_RESOURCE_BARRIER::UAV(m_foamStats.Get());
    cl->ResourceBarrier(1, &statsUav);
    cl->SetPipelineState(m_psoFoamStats.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch(1, 1, 1);
    cl->ResourceBarrier(1, &statsUav);
    Transition(cl, m_foamStats.Get(), m_foamStatsState, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->CopyBufferRegion(m_foamStatsReadback.Get(), m_frameIndex * sizeof(OceanFoamState), m_foamStats.Get(),
                         OCEAN_FOAM_STATE_OFFSET, sizeof(OceanFoamState));
    Transition(cl, m_foamStats.Get(), m_foamStatsState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    Transition(cl, m_foam[cur].Get(), m_foamState[cur], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
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
    push.f0 = (float)m_time;
    push.f1 = m_dt;

    cl->SetComputeRootSignature(m_rootSig.Get());
    cl->SetPipelineState(m_psoTiles.Get());
    cl->SetComputeRoot32BitConstants(0, 8, &push, 0);
    cl->Dispatch((OCEAN_TILE_VERTS + 63) / 64, (uint32_t)m_gpuTiles.size(), 1);

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
        // Absolute indices: the range starts at the buffer base.
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

        // Refit unchanged tiles; timed rebuilds, staggered per slot, limit BVH decay.
        const double age = m_time - m_blasRebuiltAt[t.slot];
        const double due = kBlasRebuildSeconds * (0.75 + 0.5 * (double)(t.slot % 64u) / 64.0);
        const bool stale = age >= due;
        const bool canUpdate = m_blasBuilt[t.slot] != 0 && !m_quadtree.SlotIsNew(t.slot) && !stale;
        if (canUpdate) {
            d.Inputs.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(
                d.Inputs.Flags | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE);
            d.SourceAccelerationStructureData = d.DestAccelerationStructureData;
            ++m_stats.refits;
        } else {
            m_blasRebuiltAt[t.slot] = m_time;
            ++m_stats.builds;
        }

        // Scratch range per build, so no barriers between builds.
        d.ScratchAccelerationStructureData =
            m_blasScratch->GetGPUVirtualAddress() + (uint64_t)i * m_blasScratchSize;

        cl->BuildRaytracingAccelerationStructure(&d, 0, nullptr);
        m_blasBuilt[t.slot] = 1;
    }

    const D3D12_RESOURCE_BARRIER done = CD3DX12_RESOURCE_BARRIER::UAV(m_blasBuffer.Get());
    cl->ResourceBarrier(1, &done);
}

void OceanSystem::append_instances(planet::TlasBuilder& tlas, InstanceProperties* props,
                                   const planet::DVec3& sceneOrigin, uint32_t hitGroup, bool& forceRebuild, bool& forceRefit) {
    (void)forceRebuild;
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
    // Vertices moved: refit, since a rebuild would redo the whole scene.
    forceRefit = true;
}

void OceanSystem::on_submitted(uint64_t copyFence, uint64_t computeFence) {
    (void)copyFence;
    m_timestampFence[m_frameIndex] = computeFence;
}

} // namespace ocean
