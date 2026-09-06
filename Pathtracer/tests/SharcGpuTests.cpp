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
#include "../shaders/SharcLayout.h"
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
    std::array<ComPtr<ID3D12PipelineState>, 6> psos;
    std::array<uint32_t, 16> constants = {1, 0, 32, 64, 120, 0};
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
        params[0].Constants = {0, 0, 16};
        params[1].ParameterType = params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor.ShaderRegister = 27;
        params[2].Descriptor.ShaderRegister = 0;
        D3D12_ROOT_SIGNATURE_DESC desc{}; desc.NumParameters = 3; desc.pParameters = params;
        ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        const char* names[] = {"prepare", "fill", "resolve", "query", "eraseTop", "benchmark"};
        for (int i = 0; i < 6; ++i) {
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
    }
    void Begin() {
        Check(allocator->Reset()); Check(commands->Reset(allocator.Get(), nullptr));
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootUnorderedAccessView(1, cache->GetGPUVirtualAddress());
        commands->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
    }
    void Dispatch(int pso, uint32_t groups) {
        commands->SetPipelineState(psos[pso].Get());
        commands->SetComputeRoot32BitConstants(0, 16, constants.data(), 0);
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
    double Benchmark() {
        D3D12_QUERY_HEAP_DESC desc{}; desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP; desc.Count = 2;
        ComPtr<ID3D12QueryHeap> heap;
        Check(device->CreateQueryHeap(&desc, IID_PPV_ARGS(&heap)));
        auto timestamps = Buffer(16, D3D12_HEAP_TYPE_READBACK);
        uint64_t frequency = 0; Check(queue->GetTimestampFrequency(&frequency));
        double bestMs = 1e30;
        for (int repeat = 0; repeat < 4; ++repeat) {
            Begin();
            commands->EndQuery(heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
            Dispatch(5, 16384); // 1,048,576 diffuse cache queries
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
            Dispatch(2, SHARC_CAPACITY / SHARC_GROUP_SIZE); End();
        }
    }
    std::array<float, 512> Query(uint32_t mode) {
        constants[5] = mode; Begin(); Dispatch(3, 1);
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
    Require(argc == 2, "Expected compiled shader directory"); Runner r(argv[1]);
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
    r.Reset();
    std::cout << "BENCH: 1M cold scattered queries " << r.Benchmark() << " ms\n";
    r.Train({4}, 20);
    r.constants[5] = 4;
    std::cout << "BENCH: 1M warm surface queries " << r.Benchmark() << " ms\n";
    return 0;
} catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
