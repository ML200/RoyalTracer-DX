// Standalone, headless D3D12 regression runner. Run tests/run_sharc_tests.ps1
// from a VS Developer PowerShell. Uses the actual production HLSL algorithms.
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
// Allow the same fixture to benchmark a pre-optimization shader snapshot.
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
    ComPtr<ID3D12Resource> cache, output, readback;
    ComPtr<ID3D12Fence> fence;
    std::array<ComPtr<ID3D12PipelineState>, 11> psos;
    std::array<uint32_t, 20> constants = {1, 0, 32, 64, 120, 0};
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
        D3D12_ROOT_PARAMETER params[3]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 20};
        params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor.ShaderRegister = 27;
        params[2].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters = 3; desc.pParameters = params;
        ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        const char* names[] = {"prepare", "fill", "resolve", "query", "eraseTop", "benchmark", "guideFill", "guideQuery", "materialCheck", "materialBenchmark", "liteCheck"};
        for (int i = 0; i < 11; ++i) {
            std::ifstream file(std::string(directory) + "/" + names[i] + ".dxil", std::ios::binary);
            Require(bool(file), "Missing compiled test shader");
            std::vector<char> code((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = root.Get(); pso.CS = {code.data(), code.size()};
            Check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&psos[i])));
        }
        cache = Buffer(uint64_t(SHARC_BUFFER_BYTES), D3D12_HEAP_TYPE_DEFAULT);
        output = Buffer(4u * 1024u * 1024u, D3D12_HEAP_TYPE_DEFAULT);
        readback = Buffer(2048, D3D12_HEAP_TYPE_READBACK);
        constants[6] = Bits(0.125f); constants[7] = Bits(0.01f); constants[8] = Bits(3.0f);
        // Guiding: enabled, cap 1.0, receiver level +3, lifetime 256 frames,
        // training guided, patch radius 0.75 cell widths (production packing).
        constants[16] = GUIDE_PARAM_ENABLED | (255u << GUIDE_PARAM_QMAX_SHIFT) | (3u << GUIDE_PARAM_LEVEL_SHIFT) |
            (31u << GUIDE_PARAM_LIFETIME_SHIFT) | GUIDE_PARAM_TRAIN | (24u << GUIDE_PARAM_RADIUS_SHIFT) |
            (2u << GUIDE_PARAM_DEPTH_SHIFT);
    }
    void Begin() {
        Check(allocator->Reset()); Check(commands->Reset(allocator.Get(), nullptr));
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootUnorderedAccessView(1, cache->GetGPUVirtualAddress());
        commands->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
    }
    void Dispatch(int pso, uint32_t groups) {
        commands->SetPipelineState(psos[pso].Get());
        commands->SetComputeRoot32BitConstants(0, 20, constants.data(), 0);
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
    // Production order per frame is prepare -> update; the prepare of the NEXT
    // frame flags receivers worth guiding, so each fill frame is followed by one.
    void GuideFill(uint32_t mode, int frames = 1) {
        for (int f = 0; f < frames; ++f) {
            ++constants[1]; constants[5] = mode; Begin(); Dispatch(6, 1); End();
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
    // Deposits go to the level the rendering query samples at the current
    // camera distance, so the coarser level fills from a camera where frac(lod)
    // is large (24 m: lod 0.94), not from the close-up training above.
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
    r.constants[4] = 512; // production sparse-cell lifetime
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

    // ---- Path guiding (SharcGuide_v8.hlsli) ----
    // Result layout of guideQuery mode 0: [0] found, [1] observations, [2] mean
    // irradiance, [3] q, [4] weight sum, [5] stored patches, [8..11] raw key,
    // then per slot at 16 + 8c: offset xyz, level, luminance, hits, lastSeen,
    // valid; raw slot words from 80.
    r.Reset(); r.GuideFill(0, 4);
    auto guide = r.Query(0, 7);
    Require(guide[0] == 1, "Guide receiver entry was not created");
    Require(guide[1] >= 192 && guide[1] <= 256 && std::abs(guide[2] - 20.0f) < 0.001f,
        "Guide irradiance observations were lost or mis-scaled");
    // The dim patch is below the mean incident radiance (20 / pi) and must be
    // rejected by the contrast ranking; only the bright one is stored.
    Require(guide[5] == 1, "Guide discovery stored the wrong number of patches");
    // Fixture geometry: the bright patch explains 13.48 of the mean 20, so
    // q = 0.674 of the (unit) cap.
    Require(std::abs(guide[3] - 0.674f) < 0.02f, "Guide explained-fraction mixture weight is off");
    bool brightFound = false, dimFound = false;
    for (int c = 0; c < 8; ++c) {
        const float* s = &guide[16 + c * 8];
        if (s[7] != 1) continue;
        if (std::abs(s[0]) < 1e-4f && std::abs(s[1] - 3.0f) < 1e-4f && std::abs(s[2]) < 1e-4f)
            brightFound = s[3] == 3 && std::abs(s[4] - 100.0f) < 0.01f && s[5] == 3;
        if (std::abs(s[0] - 4.0f) < 1e-4f && std::abs(s[1] - 1.0f) < 1e-4f && std::abs(s[2]) < 1e-4f)
            dimFound = true;
    }
    Require(brightFound && !dimFound, "Guide patch offset/level/luminance/hits round trip failed");
    std::cout << "PASS: guide receiver entry, patch discovery round trip, irradiance mean, q = " << guide[3] << "\n";
    auto mc = r.Query(1, 7);
    double pdfIntegral = 0, mix = 0, mix2 = 0, cosine = 0, cosine2 = 0;
    for (int lane = 0; lane < 64; ++lane) {
        pdfIntegral += mc[lane * 8 + 0] / 64.0; mix += mc[lane * 8 + 1] / 64.0; mix2 += mc[lane * 8 + 2] / 64.0;
        cosine += mc[lane * 8 + 3] / 64.0; cosine2 += mc[lane * 8 + 4] / 64.0;
    }
    Require(mc[7] == 1 && mc[5] > 0.5f, "Guide Monte Carlo ran without an active cone set");
    Require(std::abs(pdfIntegral - 1.0) < 0.03, "Guide pdf does not integrate to one over the sphere");
    Require(std::abs(mix - cosine) < 0.03 * cosine, "Guided mixture estimate disagrees with cosine sampling");
    const double varMix = mix2 - mix * mix, varCos = cosine2 - cosine * cosine;
    Require(varMix < 0.5 * varCos, "Guided mixture did not reduce variance on the bright patch");
    std::cout << "PASS: guide pdf integral " << pdfIntegral << ", mixture " << mix << " vs cosine " << cosine
        << ", variance ratio " << varMix / varCos << "\n";
    r.Reset(); r.GuideFill(1, 1);
    auto ranked = r.Query(0, 7);
    Require(ranked[0] == 1 && ranked[5] == 8, "Guide slots did not fill under replacement");
    for (int c = 0; c < 8; ++c) {
        const float* s = &ranked[16 + c * 8];
        Require(s[7] == 1 && s[1] >= 14.99f && s[1] <= 22.01f, "Guide replacement kept a weaker patch over a stronger one");
    }
    r.Reset(); r.GuideFill(2, 1);
    auto merged = r.Query(0, 7);
    Require(merged[0] == 1 && merged[5] == 1 && merged[16 + 5] == 3 && std::abs(merged[16 + 4] - 66.667f) < 0.2f,
        "Guide hits inside one patch were not merged with averaged luminance");
    std::cout << "PASS: guide top-8 replacement by contribution, same-patch merging\n";
    // Visibility: six guided rays that miss the patch leave it at 255 * 0.75^6 = 45,
    // six that reach it bring it back above 200; the cone weight follows.
    r.Reset(); r.constants[15] = 6; r.GuideFill(4, 1); r.constants[15] = 0;
    auto seen = r.Query(0, 7);
    Require(seen[0] == 1 && seen[5] == 1 && seen[16 + 6] > 200 && seen[16 + 6] < 255,
        "Guide visibility feedback did not lower and recover the patch estimate");
    // Stochastic keys: the receiver sits 0.04 m from a corner of its 1 m cell,
    // so the tent weights give its own cell 0.54 * 0.5 * 0.54 = 14.6% of the
    // draws (598 of 4096, binomial sd 23) and spread the rest over neighbours.
    auto stochastic = r.Query(2, 7);
    Require(stochastic[0] + stochastic[1] == 4096 && stochastic[0] > 520 && stochastic[0] < 680,
        "Stochastic receiver keys do not follow the tent reconstruction weights");
    std::cout << "PASS: guide visibility feedback " << seen[16 + 6] << "/255, stochastic key blend "
        << stochastic[0] << " own / " << stochastic[1] << " neighbour draws\n";
    r.Reset(); r.GuideFill(0, 2);
    r.constants[1] += 121; r.Begin(); r.Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End();
    Require(r.Query(0, 7)[0] == 0, "Stale guide entries were not evicted");
    r.Reset(); r.GuideFill(0, 2);
    r.constants[4] = 4096; r.constants[1] += 1100;
    for (int f = 0; f < 16; ++f) { ++r.constants[1]; r.Begin(); r.Dispatch(0, SHARC_CAPACITY / SHARC_GROUP_SIZE); r.End(); }
    auto retired = r.Query(0, 7);
    Require(retired[0] == 1 && retired[5] == 0, "Unseen guide patches were not retired");
    r.constants[4] = 120;
    std::cout << "PASS: guide entry eviction and patch retirement\n";
    std::array<float, 512> rebased[2];
    for (int run = 0; run < 2; ++run) {
        r.Reset();
        const float origin = run == 0 ? 8000000.0f : 8001000.0f;
        r.constants[9] = Bits(origin); r.constants[10] = Bits(-origin); r.constants[11] = Bits(origin);
        // Same absolute camera, expressed in each local frame.
        const float camera = 8000000.0f - origin;
        r.constants[12] = Bits(camera); r.constants[13] = Bits(-camera); r.constants[14] = Bits(camera);
        r.constants[15] = 1;
        r.GuideFill(3, 1);
        rebased[run] = r.Query(0, 7);
    }
    r.constants[9] = r.constants[10] = r.constants[11] = 0;
    r.constants[12] = r.constants[13] = r.constants[14] = 0; r.constants[15] = 0;
    Require(rebased[0][0] == 1 && rebased[1][0] == 1 && rebased[0][5] == 2 &&
        std::memcmp(&rebased[0][8], &rebased[1][8], 16) == 0 &&
        std::memcmp(&rebased[0][80], &rebased[1][80], 128) == 0,
        "Floating-origin rebase changed guide keys or patch encodings");
    std::cout << "PASS: guide keys and patches identical across a km rebase at 8,000 km\n";

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
    // Seven does not divide a word or group: exercises multiple bits per word,
    // untouched neighbours and updates crossing every mask/group boundary.
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
    // ---- ReSTIR lite (RestirLite_v8.hlsli) ----
    {
        auto pack = r.Query(0, 10);
        Require(pack[0] < 1e-6f && pack[1] == 7.0f && pack[2] < 0.005f && pack[3] > 0.9999f &&
            pack[4] == 3.75f && pack[5] == 200.0f && pack[6] == 1.0f && pack[7] < 0.01f,
            "Lite reservoir packing changed a field");
        auto mis = r.Query(1, 10);
        float worstMis = 0.0f;
        for (int i = 0; i < 64; ++i) worstMis = std::max(worstMis, mis[i]);
        Require(worstMis < 1e-5f, "Lite resampling MIS weights do not sum to one");
        for (uint32_t mode : {2u, 3u}) {
            auto merged = r.Query(mode, 10);
            double mean = 0.0;
            for (int i = 0; i < 64; ++i) mean += merged[i];
            mean /= 64.0;
            std::cout << "Lite " << (mode == 2u ? "paired spatial" : "temporal") << " merge: estimate "
                << mean << " vs exact " << merged[64] << "\n";
            Require(std::abs(mean / merged[64] - 1.0) < 0.01, "Lite resampling is biased");
        }
        std::cout << "PASS: lite reservoir packing, pairwise/temporal MIS normalization, unbiased merges\n";
    }
    return 0;
} catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
