#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "SharcLayout.h"
#include "PathStateLayout.h"
#ifndef SHARC_RESOLVE_GROUPS
#define SHARC_RESOLVE_GROUPS (SHARC_CAPACITY / SHARC_GROUP_SIZE)
#endif
using Microsoft::WRL::ComPtr;

void Check(HRESULT hr) {
    if (FAILED(hr)) { char msg[64]; sprintf_s(msg, "D3D12 failed: 0x%08lx", hr); throw std::runtime_error(msg); }
}
void Require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
uint32_t Bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

struct Runner {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12Resource> cache, output, readback, pathState;
    ComPtr<ID3D12Fence> fence;
    std::array<ComPtr<ID3D12PipelineState>, 13> psos;
    std::array<uint32_t, 21> constants = {1, 0, 32, 64, 120, 0};
    uint64_t serial = 0;
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ~Runner() { CloseHandle(event); }

    ComPtr<ID3D12Resource> Buffer(uint64_t bytes, D3D12_HEAP_TYPE type) {
        D3D12_HEAP_PROPERTIES heap{}; heap.Type = type;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width = bytes;
        desc.Height = 1; desc.DepthOrArraySize = 1; desc.MipLevels = 1;
        desc.SampleDesc.Count = 1; desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
        ComPtr<ID3D12Resource> resource;
        Check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            type == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&resource)));
        return resource;
    }
    Runner(const char* directory) {
        Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC q{};
        Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
        Check(commands->Close());
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        D3D12_ROOT_PARAMETER params[4]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 21};
        params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor.ShaderRegister = 27;
        params[2].Descriptor.ShaderRegister = 0;
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[3].Descriptor.ShaderRegister = 10;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters = 4; desc.pParameters = params;
        ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        const char* names[] = {"prepare", "fill", "resolve", "query", "eraseTop", "benchmark", "guideFill", "guideQuery", "materialCheck", "materialBenchmark", "liteCheck", "materialSamplingCheck", "liteAccumCheck"};
        for (int i = 0; i < 13; ++i) {
            std::ifstream file(std::string(directory) + "/" + names[i] + ".dxil", std::ios::binary);
            Require(bool(file), "Missing compiled test shader");
            std::vector<char> code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = root.Get(); pso.CS = {code.data(), code.size()};
            Check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&psos[i])));
        }
        cache = Buffer(uint64_t(SHARC_BUFFER_BYTES), D3D12_HEAP_TYPE_DEFAULT);
        output = Buffer(4u * 1024u * 1024u, D3D12_HEAP_TYPE_DEFAULT);
        // 53x45 test image (2688 padded pixels) plus one training spill lane per thread.
        pathState = Buffer(uint64_t(PS_PATH_STATE_BYTES) * 2688u + uint64_t(SHARC_CAPACITY) * 48u, D3D12_HEAP_TYPE_DEFAULT);
        readback = Buffer(2048, D3D12_HEAP_TYPE_READBACK);
        constants[6] = Bits(0.125f); constants[7] = Bits(0.01f); constants[8] = Bits(3.0f);
        constants[20] = Bits(0.5f);
        constants[16] = GUIDE_PARAM_ENABLED | (255u << GUIDE_PARAM_QMAX_SHIFT) | (3u << GUIDE_PARAM_LEVEL_SHIFT) |
            (31u << GUIDE_PARAM_FRESHNESS_SHIFT) | GUIDE_PARAM_TRAIN | (2u << GUIDE_PARAM_DEPTH_SHIFT);
    }
    void Begin() {
        Check(allocator->Reset()); Check(commands->Reset(allocator.Get(), nullptr));
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootUnorderedAccessView(1, cache->GetGPUVirtualAddress());
        commands->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
        commands->SetComputeRootUnorderedAccessView(3, pathState->GetGPUVirtualAddress());
    }
    void Dispatch(int pso, uint32_t groups) {
        commands->SetPipelineState(psos[pso].Get());
        commands->SetComputeRoot32BitConstants(0, 21, constants.data(), 0);
        commands->Dispatch(groups, 1, 1);
        D3D12_RESOURCE_BARRIER barrier{}; barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        commands->ResourceBarrier(1, &barrier);
    }
    void End() {
        Check(commands->Close()); ID3D12CommandList* list[] = {commands.Get()};
        queue->ExecuteCommandLists(1, list); Check(queue->Signal(fence.Get(), ++serial));
        Check(fence->SetEventOnCompletion(serial, event));
        Require(WaitForSingleObject(event, 30000) == WAIT_OBJECT_0, "GPU test timed out");
        Check(device->GetDeviceRemovedReason());
    }
    double Benchmark(int pso = 5, uint32_t groups = 16384, int setupPso = -1, uint32_t setupGroups = 0,
                     int finishPso = -1, uint32_t finishGroups = 0) {
        D3D12_QUERY_HEAP_DESC desc{}; desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; desc.Count = 2;
        ComPtr<ID3D12QueryHeap> heap;
        Check(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&heap)));
        auto timestamps = Buffer(16, D3D12_HEAP_TYPE_READBACK);
        uint64_t frequency = 0; Check(queue->GetTimestampFrequency(&frequency));
        double bestMs = 1e30;
        for (int repeat = 0; repeat < 8; ++repeat) {
            Begin();
            if (setupPso >= 0) { ++constants[1]; Dispatch(setupPso, setupGroups); }
            commands->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            Dispatch(pso, groups);
            if (finishPso >= 0) Dispatch(finishPso, finishGroups);
            commands->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
            commands->ResolveQueryData(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, timestamps.Get(), 0);
            End();
            uint64_t* ticks; D3D12_RANGE range{0, 16};
            Check(timestamps->Map(0, &range, reinterpret_cast<void**>(&ticks)));
            double ms = double(ticks[1] - ticks[0]) * 1000.0 / double(frequency);
            timestamps->Unmap(0, nullptr);
            if (repeat > 0 && ms < bestMs) bestMs = ms;
        }
        return bestMs;
    }
    void Reset() {
        constants[0] = 1; constants[1] = 0; constants[15] = 0;
        Begin(); Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); End(); constants[0] = 0;
    }
    void Train(std::initializer_list<uint32_t> modes, int frames = 12) {
        for (int f = 0; f < frames; ++f) {
            ++constants[1]; Begin();
            for (uint32_t mode : modes) { constants[5] = mode; Dispatch(1, 64); }
            Dispatch(2, SHARC_RESOLVE_GROUPS); End();
        }
    }
    void GuideFill(uint32_t mode, int frames = 1) {
        for (int f = 0; f < frames; ++f) {
            ++constants[1]; constants[5] = mode; Begin(); Dispatch(6, 1);
            Dispatch(2, SHARC_RESOLVE_GROUPS); End();
            Begin(); Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); End();
        }
    }
    std::array<float, 512> Query(uint32_t mode, int pso = 3) {
        constants[5] = mode; Begin(); Dispatch(pso, 1);
        D3D12_RESOURCE_BARRIER b{}; b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = output.Get(); b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        commands->ResourceBarrier(1, &b);
        commands->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, 2048);
        std::swap(b.Transition.StateBefore, b.Transition.StateAfter); commands->ResourceBarrier(1, &b); End();
        void* ptr; D3D12_RANGE range{0, 2048}; Check(readback->Map(0, &range, &ptr));
        std::array<float, 512> result; std::memcpy(result.data(), ptr, 2048); readback->Unmap(0, nullptr);
        return result;
    }
};

int main(int argc, char** argv) try {
    std::cout << std::unitbuf;
    Require(argc == 2, "Expected compiled shader directory"); Runner r(argv[1]);
    auto materialErrors = r.Query(0, 8);
    float maxMaterialError = 0.0f;
    for (int i = 0; i < 64; ++i) maxMaterialError = std::max(maxMaterialError, materialErrors[i]);
    Require(maxMaterialError < 2e-5f, "Fused material/lobe evaluation changed the BSDF or PDF");
    std::cout << "PASS: 16,384 layered material cases, max relative error " << maxMaterialError << '\n';
    auto materialSampling = r.Query(0, 11);
    float accepted = 0.0f, furnace = 0.0f, coatError = 0.0f;
    for (int i = 0; i < 64; ++i) {
        accepted += materialSampling[i * 4] / 64.0f;
        furnace += materialSampling[i * 4 + 1] / 64.0f;
    }
    const float expectedFurnace = (1.0f - std::log(2.0f)) / 0.725f;
    for (uint32_t mode : {1u, 2u}) {
        const auto precision = r.Query(mode, 11);
        float error = 0.0f;
        for (int i = 0; i < 64; ++i) error = std::max(error, precision[i * 4 + 2]);
        std::cout << "Coat precision mode " << mode << ": max relative PDF error " << error << '\n';
        coatError = std::max(coatError, error);
    }
    std::cout << "GGX furnace: acceptance " << accepted << " vs 0.5, energy " << furnace
              << " vs " << expectedFurnace << '\n';
    float floorError = 0.0f;
    for (uint32_t mode : {3u, 4u, 5u}) {
        const auto floor = r.Query(mode, 11);
        float mixture = 0.0f, lobe = 0.0f, reference = 0.0f;
        for (int i = 0; i < 64; ++i) {
            mixture += floor[i * 4] / 64.0f;
            lobe += floor[i * 4 + 1] / 64.0f;
            reference += floor[i * 4 + 2] / 64.0f;
        }
        std::cout << "City floor mode " << mode << ": mixture " << mixture << ", lobe " << lobe
                  << ", uniform integral " << reference << '\n';
        floorError = std::max(floorError, std::max(std::abs(mixture - reference), std::abs(lobe - reference)) / reference);
    }
    Require(std::abs(accepted - 0.5f) < 0.005f, "GGX sampler folds null events into valid directions");
    Require(std::abs(furnace - expectedFurnace) < 0.01f, "GGX sampler adds energy in a white furnace");
    Require(coatError < 0.01f, "Smooth coat PDF loses precision near the highlight");
    Require(floorError < 0.015f, "City floor BSDF sampling disagrees with independently integrated lighting");
    std::cout << "PASS: GGX sampling distribution, furnace energy and smooth-coat precision\n";
    {
        float worstDiffuse = 0.0f, worstLayered = 0.0f;
        for (float nv : {0.15f, 0.8f, 1.0f}) for (float rough : {0.5f, 1.0f}) {
            r.constants[15] = Bits(nv);
            r.constants[12] = Bits(rough);
            const auto moments = r.Query(100u, 11);
            for (int component = 0; component < 4; ++component) {
                float mean = 0.0f;
                for (int lane = 0; lane < 64; ++lane) mean += moments[lane * 4 + component] / 64.0f;
                r.constants[component == 3 ? 14 : 17 + component] = Bits(mean);
            }
            for (float dr : {0.0f, 0.5f, 1.0f}) {
                r.constants[13] = Bits(dr);
                for (uint32_t mode : {101u, 102u, 103u, 104u}) {
                    const auto energy = r.Query(mode, 11);
                    float mean = 0.0f;
                    for (int lane = 0; lane < 64; ++lane) mean += energy[lane * 4] / 64.0f;
                    Require(std::isfinite(mean), "Material furnace returned non-finite energy");
                    if (mode == 104u) worstDiffuse = std::max(worstDiffuse, std::abs(mean - 1.0f));
                    else worstLayered = std::max(worstLayered, std::abs(mean - 1.0f));
                }
            }
        }
        Require(worstDiffuse < 0.005f, "Oren-Nayar diffuse loses or adds white-furnace energy");
        Require(worstLayered < 0.015f, "Layered material compensation loses or adds white-furnace energy");
        std::cout << "PASS: Oren-Nayar furnace error " << worstDiffuse
                  << ", layered GGX/coat/sheen furnace error " << worstLayered << '\n';
        r.constants[12] = r.constants[13] = r.constants[14] = r.constants[15] = 0;
        r.constants[17] = r.constants[18] = r.constants[19] = 0;
    }
    r.Reset(); Require(r.Query(0)[3] == 0, "Fresh cache must miss");
    r.Train({0}); auto top = r.Query(0);
    Require(top[3] > 0.95f && std::abs(top[0] - 100.0f) < 0.01f, "Concurrent HDR accumulation/resolve failed");
    Require(r.Query(1)[3] == 0, "Thin underside leaked from the bright top");
    r.Train({0, 1, 2});
    for (uint32_t mode : {1u, 2u}) {
        auto dark = r.Query(mode);
        Require(dark[3] > 0.8f && std::abs(dark[0] - 0.001f) < 0.00001f, "Dark sheet failed to retain separate radiance");
    }
    r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(2)[3] > 0.8f, "Eviction hole hid a later colliding surface");
    std::cout << "PASS: concurrent HDR accumulation, 1 mm opposite faces, parallel sheets, eviction holes\n";
    r.constants[5] = 9; r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(2)[3] == 0, "Fingerprint collision was accepted without exact-key verification");
    r.Reset(); r.constants[5] = 10; r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    r.Train({0}, 1); Require(r.Query(0)[3] == 0, "Saturated table returned a false hit");
    std::cout << "PASS: exact-key collision rejection and saturated-table fallback\n";
    r.Reset(); r.Train({2});
    r.constants[1] += 121; r.Begin(); r.Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(2)[3] == 0, "Stale entries were not evicted");
    r.Reset(); r.Train({3}); Require(r.Query(3)[3] == 0, "Zero-only history incorrectly certified darkness");
    std::cout << "PASS: stale eviction, fresh reset, zero-only fallback\n";
    r.Reset(); r.Train({4}, 20); auto sweep = r.Query(4);
    for (int i = 0; i < 64; ++i) {
        Require(sweep[i * 8 + 3] > 0.8f, "Insufficient interpolation coverage");
        for (int c = 0; c < 3; ++c)
            Require(std::abs(sweep[i * 8 + c] - (2.0f + c)) < 0.002f, "Grid boundary changed constant radiance");
    }
    auto contact = r.Query(7);
    for (int i = 0; i < 64; ++i) Require(contact[i * 8 + 7] == 0, "Tiny path footprint used cache");
    std::cout << "PASS: filtered cell-boundary sweep, contact-path fallback\n";
    r.constants[12] = Bits(-12.0f); r.Train({4}, 12);
    r.constants[12] = Bits(-24.0f); r.Train({4}, 12);
    for (int j = 0; j < 32; ++j) {
        r.constants[12] = Bits(-12.0f - 0.5f * j);
        auto lod = r.Query(4); int hits = 0;
        for (int i = 0; i < 64; ++i) if (lod[i * 8 + 7] > 0) {
            ++hits;
            for (int c = 0; c < 3; ++c)
                Require(std::abs(lod[i * 8 + 4 + c] - (2.0f + c)) < 0.002f, "LOD transition changed constant radiance");
        }
        Require(hits > 16, "LOD transition unexpectedly lost trained coverage");
    }
    r.constants[12] = 0;
    r.Reset(); r.Train({8}, 16); auto normals = r.Query(8);
    for (int i = 0; i < 64; ++i) {
        Require(normals[i * 8 + 3] > 0.9f, "Normal-axis transition lost coverage");
        Require(std::abs(normals[i * 8] - 2.0f) < 0.002f, "Normal hash boundary changed radiance");
    }
    std::cout << "PASS: adjacent-LOD blend and continuous normal-axis blend\n";
    for (uint32_t level : {0u, 3u, 12u, 24u}) {
        r.constants[15] = level;
        r.constants[9] = Bits(8000000); r.constants[10] = Bits(-8000000); r.constants[11] = Bits(8000000);
        auto a = r.Query(6);
        r.constants[9] = Bits(8001000); r.constants[10] = Bits(-8001000); r.constants[11] = Bits(8001000);
        auto b = r.Query(6);
        Require(std::memcmp(a.data(), b.data(), 32) == 0, "Floating-origin rebase changed key/fraction");
    }
    r.constants[9] = r.constants[10] = r.constants[11] = 0; r.constants[15] = 0;
    std::cout << "PASS: signed centimetre precision across km rebases at 8,000 km\n";
    r.Reset(); r.Train({5}, 24); auto deep = r.Query(5);
    const float expected = 10.0f * std::pow(0.8f, 20.0f);
    Require(deep[3] > 0.8f && std::abs(deep[0] / expected - 1.0f) < 0.04f, "20-bounce RR suffix estimator failed");
    std::cout << "PASS: 20-bounce rare-survival suffix (observed " << deep[0] << ", expected " << expected << ")\n";
    r.Reset();
    auto missing = r.Query(20);
    Require(missing[3] == 0 && missing[0] > 0.4f && missing[1] < 0.03f,
        "Debug missing history must be magenta, not black radiance");
    r.Train({0}, 1);
    auto cold = r.Query(20);
    Require(cold[3] == 1 && std::abs(cold[0] - 100.0f) < 0.01f && r.Query(0)[3] == 0,
        "Debug must expose resolved radiance before rendering confidence accepts it");
    Require(r.Query(21)[3] == 0 && r.Query(22)[3] == 0,
        "Debug lookup leaked across an opposite face or parallel sheet");
    auto otherView = r.Query(24);
    Require(otherView[3] == 1 && std::abs(otherView[0] - 100.0f) < 0.01f,
        "Debug occupancy must remain visible from another camera direction");
    r.constants[12] = Bits(-24.0f); r.Train({0}); r.constants[12] = 0;
    r.constants[15] = 1;
    auto coarse = r.Query(20);
    Require(coarse[3] == 1 && std::abs(coarse[0] - 100.0f) < 0.01f,
        "Debug coarse level lookup failed");
    r.constants[15] = 0;
    r.Train({0, 1, 2});
    for (uint32_t mode : {21u, 22u}) {
        auto dark = r.Query(mode);
        Require(dark[3] == 1 && std::abs(dark[0] - 0.001f) < 0.00001f,
            "Debug lost independent thin-sheet lighting");
    }
    r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(22)[3] == 1 && r.Query(20)[3] == 0,
        "Debug collision scan failed across an eviction hole");
    r.constants[5] = 9; r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(22)[3] == 0, "Debug accepted a fingerprint collision");
    r.Reset(); r.Train({3}, 1);
    auto zero = r.Query(23);
    Require(zero[3] == 1 && zero[0] == 0 && zero[1] == 0 && zero[2] == 0,
        "Debug stored zero must be distinguishable from missing history");
    std::cout << "PASS: debug cold/black/missing history, camera direction, hierarchy, thin sheets and collisions\n";
    r.Reset(); r.Train({16});
    const float suffixes[2][3] = {{1.5f, 3.6f, 1.7f}, {1.0f, 1.8f, 2.8f}};
    const float reflectance[3] = {0.25f, 0.5f, 0.75f};
    for (int vertex = 0; vertex < 2; ++vertex) {
        auto suffix = r.Query(16u + vertex);
        Require(suffix[3] > 0.8f, "Compact training state lost a vertex's deposits");
        for (int channel = 0; channel < 3; ++channel)
            Require(std::abs(suffix[channel] * reflectance[channel] - suffixes[vertex][channel]) < 0.002f,
                "Compact suffix propagation or per-vertex demodulation is incorrect");
    }
    std::cout << "PASS: compact multi-vertex suffix propagation and colored demodulation\n";
    r.Reset(); r.Train({42});
    auto mirrored = r.Query(42);
    Require(mirrored[3] > 0.8f && std::abs(mirrored[0] - 2.0f) < 0.002f,
        "The cache learned light found past a specular reflection");
    std::cout << "PASS: light past a specular reflection stays out of the registered vertex's label\n";
    r.Reset();
    r.constants[12] = Bits(0.04f - 12.5f * std::sqrt(2.0f));
    r.constants[14] = Bits(0.04f);
    r.Train({14});
    for (int holes = 0; holes < 2; ++holes) {
        if (holes != 0) {
            r.constants[5] = 15; r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
        }
        auto sampled = r.Query(30);
        for (int component = 0; component < 4; ++component) {
            double estimate = 0;
            for (int lane = 0; lane < 64; ++lane) estimate += sampled[lane * 8 + 4 + component] / 64.0;
            Require(std::abs(estimate - sampled[component]) < 0.01 * std::max(1.0f, sampled[component]),
                "Stochastic reconstruction differs from the deterministic kernel's radiance or acceptance probability");
        }
    }
    r.constants[12] = r.constants[14] = 0;
    std::cout << "PASS: stochastic spatial/normal/LOD reconstruction matches varying radiance and partial coverage\n";
    r.Reset();
    r.constants[4] = 512;
    for (int observation = 0; observation < 256; ++observation) {
        r.constants[1] += 59;
        r.Begin(); r.Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
        r.Train({11}, 1);
    }
    auto sparse = r.Query(11);
    std::cout << "Sparse history: radiance " << sparse[0] << ", support " << sparse[3] << "\n";
    auto sparseRecord = r.Query(31);
    std::cout << "Sparse record W/W2/frames/conf/mean/age: " << sparseRecord[0] << " / " << sparseRecord[1] << " / "
        << sparseRecord[2] << " / " << sparseRecord[3] << " / " << sparseRecord[4] << " / " << sparseRecord[7] << "\n";
    Require(sparse[3] > 0.3f && std::abs(sparse[0] - 2.0f) < 0.002f,
        "Infrequently visited cells lost their history before reaching confidence");
    r.constants[4] = 120;
    r.Reset(); r.Train({0});
    r.constants[5] = 12; r.Begin(); r.Dispatch(4, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(0)[3] == 0, "Corrupt history must fail closed before resolve repairs it");
    r.Train({0}, 1);
    auto healed = r.Query(20);
    Require(healed[3] == 1 && std::abs(healed[0] - 100.0f) < 0.01f,
        "Valid observations did not repair NaN history");
    r.Reset(); r.Train({13}, 1);
    Require(r.Query(20)[3] == 0, "Invalid training position populated the cache");
    std::cout << "PASS: sparse 60-frame revisits accumulate evidence, poisoned history repairs, invalid input rejected\n";
    {
        // Light goes out: no path ends in the old light, history follows within a few updates.
        auto light = [&](float radiance, int frames, uint32_t mode = 40u) {
            r.constants[17] = Bits(radiance); r.Train({mode}, frames);
        };
        r.Reset(); light(100.0f, 40);
        auto lit = r.Query(40);
        // Half-float mean: one step at 100 is 0.0625.
        for (int i = 0; i < 64; ++i)
            Require(lit[i * 8 + 3] == 1 && std::abs(lit[i * 8] - 100.0f) < 0.5f, "A steady light was not cached");
        Require(lit[4] > 0.999f && lit[5] == 0, "A steady light was not converged");
        light(1.0f, 1);
        auto dark = r.Query(40);
        int gated = 0;
        for (int i = 0; i < 64; ++i) {
            gated += dark[i * 8 + 3] == 0 ? 1 : 0;
            Require(dark[i * 8 + 3] == 0 || dark[i * 8] < 10.0f, "A path ended in a light that went out");
            Require(dark[i * 8 + 7] == 0, "A confidence-drawn query ended in a light that went out");
        }
        Require(dark[4] < 0.5f && dark[5] == 1, "The gate did not flag a history behind the lighting");
        light(1.0f, 3);
        auto caught = r.Query(40);
        for (int i = 0; i < 64; ++i)
            Require(caught[i * 8 + 3] == 1 && std::abs(caught[i * 8] - 1.0f) < 0.05f && caught[i * 8 + 5] == 0,
                "The history did not follow a light that went out");
        r.constants[20] = Bits(0.0f);
        r.Reset(); light(100.0f, 40); light(1.0f, 1);
        auto ungated = r.Query(40);
        for (int i = 0; i < 64; ++i)
            Require(ungated[i * 8 + 3] == 1 && ungated[i * 8] < 10.0f && ungated[i * 8 + 5] == 0,
                "Threshold 0 still gated, or the history kept the old light");
        r.constants[20] = Bits(0.5f);
        r.Reset();
        int noisyGated = 0;
        float lowest = 1.0f;
        for (int frame = 0; frame < 48; ++frame) {
            light(10.0f, 1, 41u);
            if (frame < 4) continue;
            auto noisy = r.Query(40);
            for (int i = 0; i < 64; ++i) noisyGated += noisy[i * 8 + 3] == 0 ? 1 : 0;
            lowest = std::min(lowest, noisy[4]);
        }
        Require(noisyGated == 0 && lowest > 0.9f, "A steady noisy light was taken for a light change");
        r.constants[17] = 0;
        std::cout << "PASS: light going out gated on " << gated << "/64 lanes, history followed within 4 updates; "
                     "steady noisy light never gated (lowest convergence " << lowest << ")\n";
    }

    {
        auto cap = r.Query(4, 7);
        for (int lane = 0; lane < 64; ++lane) {
            Require(std::abs(cap[lane * 4 + 0] - 1.0f) < 0.03f, "Soft cap density does not integrate to one");
            Require(std::abs(cap[lane * 4 + 1] - cap[lane * 4 + 2]) < 0.01f, "Soft cap sampler moment differs from 1 - a/3");
        }
        std::cout << "PASS: soft cap normalization and sampling moments for apertures 0.001 .. 2\n";
    }
    r.Reset(); r.constants[15] = 64u | 256u; r.GuideFill(0, 16); r.constants[15] = 0;
    auto guide = r.Query(0, 7);
    auto dumpGuide = [](const std::array<float, 512>& g) {
        std::cout << "  receiver: found " << g[0] << " opportunities " << g[1] << " mean " << g[2] << " q " << g[3]
            << " weightSum " << g[4] << " active " << g[5] << " measured " << g[6] << " bsdf " << g[7]
            << " unexplained " << g[12] << " qTrain " << g[13] << " measuredBranch " << g[14] << " worth " << g[15] << "\n";
        for (int c = 0; c < 8; ++c) {
            const float* s = &g[16 + c * 12];
            if (s[11] != 1) continue;
            std::cout << "  lobe " << c << ": axis (" << s[0] << ", " << s[1] << ", " << s[2] << ") aperture " << s[3]
                << " branch " << s[4] << " invDist " << s[5] << " mass " << s[6] << " support " << s[7] << " phi " << s[8]
                << " measured " << s[9] << " age " << s[10] << "\n";
        }
    };
    dumpGuide(guide);
    Require(guide[0] == 1, "Guide receiver entry was not created");
    Require(std::abs(guide[1] - 1024.0f) < 2.0f, "Guide opportunity history was not capped at 1024");
    Require(guide[2] > 2.5f && guide[2] < 4.5f, "Guide mean return per opportunity is off");
    Require(guide[6] >= 1 && guide[15] == 1 && guide[3] > 0.6f,
        "The bright blob was not learned as a measured lobe with a strong branch probability");
    int bright = -1;
    for (int c = 0; c < 8; ++c)
        if (guide[16 + c * 12 + 11] == 1 && (bright < 0 || guide[16 + c * 12 + 4] > guide[16 + bright * 12 + 4])) bright = c;
    Require(bright >= 0, "No lobe carries branch probability");
    {
        const float* s = &guide[16 + bright * 12];
        Require(s[1] > 0.999f && std::abs(s[0]) < 0.03f && std::abs(s[2]) < 0.03f, "Bright lobe axis is off");
        Require(s[3] > 0.005f && s[3] < 0.08f, "Bright lobe aperture does not match a 10 degree blob");
        Require(std::abs(s[5] * 3.0f - 1.0f) < 0.1f, "Bright lobe anchor distance is off");
        Require(s[9] == 1 && s[7] >= 16.0f, "Bright lobe was not confirmed");
        Require(s[6] > 0.7f * (guide[12] + s[6]), "Bright lobe does not hold most of the mass");
        std::cout << "PASS: guide learned the blob: axis (" << s[0] << ", " << s[1] << ", " << s[2] << "), aperture " << s[3]
            << ", branch " << s[4] << ", anchor " << 1.0f / s[5] << " m, q = " << guide[3] << "\n";
    }
    auto mc = r.Query(1, 7);
    double pdfIntegral = 0, mix = 0, mix2 = 0, cosine = 0, cosine2 = 0;
    for (int lane = 0; lane < 64; ++lane) {
        pdfIntegral += mc[lane * 8 + 0] / 64.0; mix += mc[lane * 8 + 1] / 64.0; mix2 += mc[lane * 8 + 2] / 64.0;
        cosine += mc[lane * 8 + 3] / 64.0; cosine2 += mc[lane * 8 + 4] / 64.0;
    }
    Require(mc[7] == 1 && mc[5] > 0.5f, "Guide Monte Carlo ran without an active lobe set");
    Require(std::abs(pdfIntegral - 1.0) < 0.03, "Guide mixture pdf does not integrate to one over the sphere");
    Require(std::abs(mix - cosine) < 0.03 * cosine, "Guided mixture estimate disagrees with cosine sampling");
    const double varMix = mix2 - mix * mix, varCos = cosine2 - cosine * cosine;
    Require(varMix < 0.5 * varCos, "Guided mixture did not reduce variance on the bright blob");
    std::cout << "PASS: guide pdf integral " << pdfIntegral << ", mixture " << mix << " vs cosine " << cosine
        << ", variance ratio " << varMix / varCos << "\n";
    r.constants[1] += 768;
    auto stale = r.Query(0, 7);
    r.constants[1] -= 768;
    Require(stale[3] < 0.2f * guide[3] && stale[3] > 0.05f * guide[3], "Guide freshness attenuation is off");
    r.Reset(); r.constants[15] = 64u | 256u; r.GuideFill(8, 16); r.constants[15] = 0;
    auto inherited = r.Query(0, 7);
    Require(inherited[0] == 0 && inherited[112] > 0.6f, "Parent-level fallback did not guide a cold child receiver");
    std::cout << "PASS: cold child receiver inherits the parent model, q = " << inherited[112] << "\n";
    r.Reset(); r.constants[15] = 64u | 256u; r.GuideFill(5, 24); r.constants[15] = 0;
    auto two = r.Query(0, 7);
    bool sideFound = false;
    for (int c = 0; c < 8; ++c) {
        const float* s = &two[16 + c * 12];
        if (s[11] != 1 || s[9] != 1) continue;
        if (s[0] * 0.866f + s[1] * 0.5f > 0.95f && s[5] == 0.0f) sideFound = true;
    }
    Require(two[6] >= 2 && sideFound, "Challenger discovery did not seed and confirm the second blob");
    std::cout << "PASS: two blobs learned as " << two[6] << " measured lobes, q = " << two[3] << "\n";
    r.Reset(); r.constants[15] = 64u | 256u; r.GuideFill(6, 24); r.constants[15] = 0;
    auto uniform = r.Query(0, 7);
    Require(uniform[0] == 1 && uniform[14] < 0.1f && uniform[3] == 0.0f,
        "A uniformly lit receiver kept a measured guide probability");
    std::cout << "PASS: uniform lighting converges to no guiding (measured branch sum " << uniform[14] << ")\n";
    auto stochastic = r.Query(2, 7);
    Require(stochastic[0] + stochastic[1] == 4096 && stochastic[0] > 520 && stochastic[0] < 680,
        "Stochastic receiver keys do not follow the tent reconstruction weights");
    std::cout << "PASS: stochastic key blend " << stochastic[0] << " own / " << stochastic[1] << " neighbour draws\n";
    r.Reset(); r.constants[15] = 8; r.GuideFill(0, 2); r.constants[15] = 0;
    r.constants[1] += 121; r.Begin(); r.Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(0, 7)[0] == 0, "Stale guide entries were not evicted");
    std::cout << "PASS: guide entry eviction\n";
    std::array<float, 512> rebased[2];
    for (int run = 0; run < 2; ++run) {
        r.Reset();
        const float origin = run == 0 ? 8000000.0f : 8001000.0f;
        r.constants[9] = Bits(origin); r.constants[10] = Bits(-origin); r.constants[11] = Bits(origin);
        const float camera = 8000000.0f - origin;
        r.constants[12] = Bits(camera); r.constants[13] = Bits(-camera); r.constants[14] = Bits(camera);
        r.constants[15] = 64u | 256u;
        r.GuideFill(3, 16);
        r.constants[15] = 1;
        rebased[run] = r.Query(0, 7);
    }
    r.constants[9] = r.constants[10] = r.constants[11] = 0;
    r.constants[12] = r.constants[13] = r.constants[14] = 0; r.constants[15] = 0;
    Require(rebased[0][0] == 1 && rebased[1][0] == 1 && std::memcmp(&rebased[0][8], &rebased[1][8], 16) == 0,
        "Floating-origin rebase changed guide keys");
    {
        int a = -1, b = -1;
        for (int c = 0; c < 8; ++c) {
            if (rebased[0][16 + c * 12 + 11] == 1 && (a < 0 || rebased[0][16 + c * 12 + 4] > rebased[0][16 + a * 12 + 4])) a = c;
            if (rebased[1][16 + c * 12 + 11] == 1 && (b < 0 || rebased[1][16 + c * 12 + 4] > rebased[1][16 + b * 12 + 4])) b = c;
        }
        Require(a >= 0 && b >= 0, "Rebased guides did not learn the blob");
        const float* s0 = &rebased[0][16 + a * 12];
        const float* s1 = &rebased[1][16 + b * 12];
        Require(s0[1] > 0.999f && s1[1] > 0.999f && std::abs(s0[3] - s1[3]) < 0.01f &&
            std::abs(s0[5] - s1[5]) < 0.02f && std::abs(rebased[0][3] - rebased[1][3]) < 0.05f,
            "Floating-origin rebase changed the learned lobe");
    }
    std::cout << "PASS: guide keys and learned lobes agree across a km rebase at 8,000 km\n";

    r.Reset();
    std::cout << "BENCH: 1M cold scattered queries " << r.Benchmark() << " ms\n";
    r.Train({4}, 20);
    r.constants[5] = 4;
    std::cout << "BENCH: 1M warm surface queries " << r.Benchmark() << " ms\n";

    r.Reset(); r.constants[5] = 30;
    r.Begin(); r.Dispatch(1, SHARC_CAPACITY / 64u); r.End();
    r.constants[5] = 31; r.constants[15] = 7;
    ++r.constants[1]; r.Begin(); r.Dispatch(1, SHARC_CAPACITY / 64u);
    r.Dispatch(2, SHARC_RESOLVE_GROUPS); r.End();
    auto sparseResolved = r.Query(31);
    for (uint32_t offset : {0u, 255u, 8191u, SHARC_CAPACITY - 64u}) {
        r.constants[17] = offset;
        auto boundary = r.Query(31);
        for (uint32_t i = 0; i < 64u; ++i) {
            const bool updated = (offset + i) % 7u == 0u;
            Require(boundary[i * 8 + 3] == (updated ? 1.0f : 0.0f), "Resolve changed an untouched slot or missed a dirty bit");
            Require(boundary[i * 8] == (updated ? 2.0f : 0.0f), "Sparse resolve lost radiance");
        }
    }
    r.constants[17] = 0;
    r.Begin(); r.Dispatch(2, SHARC_RESOLVE_GROUPS); r.End();
    auto idle = r.Query(31);
    Require(sparseResolved == idle, "Resolve replayed a consumed dirty mask");
    std::cout << "PASS: sparse resolve visits every dirty bit once and preserves untouched history\n";
    std::cout << "BENCH: occupied idle resolve " << r.Benchmark(2, SHARC_RESOLVE_GROUPS) << " ms\n";
    for (uint32_t stride : {128u, 16u, 1u}) {
        r.constants[5] = 31; r.constants[15] = stride;
        std::cout << "BENCH: resolve 1/" << stride << " occupied slots "
            << r.Benchmark(2, SHARC_RESOLVE_GROUPS, 1, SHARC_CAPACITY / 64u) << " ms\n";
        std::cout << "BENCH: accumulate + resolve 1/" << stride << " occupied slots "
            << r.Benchmark(1, SHARC_CAPACITY / 64u, -1, 0, 2, SHARC_RESOLVE_GROUPS) << " ms\n";
    }
    r.constants[0] = 1;
    std::cout << "BENCH: cache + guide reset " << r.Benchmark(0, SHARC_CAPACITY / SHARC_GROUP_SIZE) << " ms\n";
    r.constants[0] = 0;
    for (uint32_t mode : {33u, 34u, 35u, 36u}) {
        r.constants[5] = mode;
        std::cout << "BENCH: 1M material evaluations mode " << mode << " " << r.Benchmark(9) << " ms\n";
    }
    {
        auto pack = r.Query(0, 10);
        Require(pack[0] < 1e-6f && pack[1] == 7.0f && pack[2] < 0.005f && pack[3] > 0.9999f &&
            pack[4] == 3.75f && pack[5] == 200.0f && pack[6] == 1.0f && pack[7] < 0.01f,
            "Lite reservoir packing changed a field");
        auto mis = r.Query(1, 10);
        float worstMis = 0.0f;
        for (int i = 0; i < 64; ++i) worstMis = std::max(worstMis, mis[i]);
        Require(worstMis < 1e-5f, "Lite resampling MIS weights do not sum to one");
        {
            auto merged = r.Query(2, 10);
            double mean = 0.0;
            for (int i = 0; i < 64; ++i) mean += merged[i];
            mean /= 64.0;
            std::cout << "Lite spatial merge: estimate " << mean << " vs exact " << merged[64] << "\n";
            Require(std::abs(mean / merged[64] - 1.0) < 0.01, "Lite resampling is biased");
        }
        auto scale = r.Query(3, 10);
        for (int i = 0; i < 64; ++i)
            Require(std::abs(scale[i] - 1.0f) < 1e-5f,
                "Lite area-measure weight lost energy when the scene scale changed");
        std::cout << "PASS: lite packing, spatial MIS and area-weight scale invariance\n";
    {
        const auto accum = r.Query(0, 12);
        for (int i = 0; i < 64; ++i)
            Require(accum[i] == 0.0f, "Reservoir-side candidate accumulation differs from the register generator");
        std::cout << "PASS: reservoir-side reuse accumulation matches the register generator\n";
    }
    }
    return 0;
} catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
