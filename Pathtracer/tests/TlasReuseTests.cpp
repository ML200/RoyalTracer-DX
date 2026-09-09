#define NOMINMAX
#include <windows.h>
#include <d3dx12.h>
#include <DirectXMath.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include "../rdn/planet/tlas_builder.h"

using namespace planet;
static void Check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D12 operation failed"); }
static void Require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

struct Runner {
    ComPtr<ID3D12Device5> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList4> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12Resource> output, readback, vertices, blas, scratch;
    uint64_t serial = 0;
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    TlasBuilder tlas;

    explicit Runner(const char* shader) {
        Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
        Check(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options)));
        Require(options.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1, "DXR 1.1 is required");
        D3D12_COMMAND_QUEUE_DESC q{};
        Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        Require(event != nullptr, "Event creation failed");
        CD3DX12_ROOT_PARAMETER params[3];
        params[0].InitAsShaderResourceView(0);
        params[1].InitAsUnorderedAccessView(0);
        params[2].InitAsConstants(4, 0);
        CD3DX12_ROOT_SIGNATURE_DESC desc(3, params);
        ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        std::ifstream file(shader, std::ios::binary);
        Require(bool(file), "Missing ray query shader");
        std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root.Get(); pd.CS = {code.data(), code.size()};
        Check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
        output = create_buffer(device.Get(), 4 * 8, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, HEAP_DEFAULT);
        readback = create_buffer(device.Get(), 4 * 8, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, HEAP_READBACK);
        const float v[] = {0,0,0, 1,0,0, 0,1,0};
        vertices = create_buffer(device.Get(), sizeof(v), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
        void* mapped; Check(vertices->Map(0, nullptr, &mapped));
        memcpy(mapped, v, sizeof(v)); vertices->Unmap(0, nullptr);
        D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
        geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geometry.Triangles.VertexBuffer = {vertices->GetGPUVirtualAddress(), 12};
        geometry.Triangles.VertexCount = 3;
        geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
        build.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        build.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        build.Inputs.NumDescs = 1;
        build.Inputs.pGeometryDescs = &geometry;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs, &info);
        blas = create_buffer(device.Get(), info.ResultDataMaxSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, HEAP_DEFAULT);
        scratch = create_buffer(device.Get(), info.ScratchDataSizeInBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, HEAP_DEFAULT);
        build.DestAccelerationStructureData = blas->GetGPUVirtualAddress();
        build.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
        commands->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(blas.Get());
        commands->ResourceBarrier(1, &barrier);
        Flush();
        tlas.init(device.Get(), 1);
    }
    ~Runner() { if (event) CloseHandle(event); }
    void Flush() {
        Check(commands->Close());
        ID3D12CommandList* lists[] = {commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        Check(queue->Signal(fence.Get(), ++serial));
        Check(fence->SetEventOnCompletion(serial, event));
        Require(WaitForSingleObject(event, 30000) == WAIT_OBJECT_0, "GPU timed out");
        Check(allocator->Reset());
        Check(commands->Reset(allocator.Get(), nullptr));
    }
    void Add(float x, UINT id, UINT hitGroup = 0,
             D3D12_RAYTRACING_INSTANCE_FLAGS flags = D3D12_RAYTRACING_INSTANCE_FLAG_NONE) {
        const float transform[] = {1,0,0,x, 0,1,0,0, 0,0,1,0};
        tlas.add_instance(blas->GetGPUVirtualAddress(), transform, id, hitGroup, flags);
    }
    void Probe(float x, UINT expectedIdPlusOne) {
        const float constants[] = {x, 0.25f, 2, 0}; // output slot 0 (same bits in uint)
        commands->SetPipelineState(pso.Get());
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootShaderResourceView(0, tlas.tlas_address());
        commands->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        commands->SetComputeRoot32BitConstants(2, 4, constants, 0);
        commands->Dispatch(1,1,1);
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
        commands->ResourceBarrier(1, &barrier);
        commands->CopyResource(readback.Get(), output.Get());
        barrier = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        commands->ResourceBarrier(1, &barrier);
        Flush();
        void* mapped; Check(readback->Map(0, nullptr, &mapped));
        UINT result[2]; memcpy(result, mapped, sizeof(result)); readback->Unmap(0, nullptr);
        Require(result[0] == expectedIdPlusOne, "Ray hit the wrong instance after TLAS change/reuse");
        if (result[0]) { float t; memcpy(&t, &result[1], 4); Require(std::abs(t - 2) < 1e-5f, "Wrong ray distance"); }
    }
    void Test() {
        const auto original = tlas.tlas_address();
        tlas.begin(2); Add(0, 0); Add(5, 1);
        Require(tlas.max_instances() >= 2 && tlas.tlas_address() != original, "TLAS capacity did not grow");
        Require(tlas.build(commands.Get()), "Initial build skipped");
        Probe(0.25f, 1); Probe(5.25f, 2);
        const auto stable = tlas.tlas_address();
        for (int i = 0; i < 100; ++i) {
            tlas.begin(); Add(0, 0); Add(5, 1);
            Require(!tlas.build(commands.Get()), "Unchanged scene rebuilt its TLAS");
        }
        Require(tlas.tlas_address() == stable, "Unchanged TLAS reallocated");
        Probe(5.25f, 2);
        tlas.begin(); Add(2, 0); Add(5, 1);
        Require(tlas.build(commands.Get()), "Transform change skipped");
        Probe(0.25f, 0); Probe(2.25f, 1);
        tlas.begin(); Add(2, 17, 4, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE); Add(5, 1);
        Require(tlas.build(commands.Get()), "Instance metadata change skipped");
        Probe(2.25f, 18);
        tlas.begin(); Add(2, 17, 4, D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
        Require(tlas.build(commands.Get()), "Removed instance did not rebuild");
        Probe(5.25f, 0);
        tlas.begin(3); Add(0, 0); Add(5, 1); Add(10, 2);
        Require(tlas.build(commands.Get()), "Growing instance set did not rebuild");
        Probe(10.25f, 3);
        tlas.begin(); Add(0, 0); Add(5, 1); Add(10, 2);
        Require(tlas.build(commands.Get(), true), "Explicit BLAS-change invalidation skipped");
        Probe(0.25f, 1);
        tlas.begin();
        Require(tlas.build(commands.Get()), "Empty scene did not rebuild");
        Probe(0.25f, 0);
        tlas.begin(); Require(!tlas.build(commands.Get()), "Empty scene rebuilt again");
        bool rejected = false;
        try { Add(0, 0x1000000u); } catch (const std::runtime_error&) { rejected = true; }
        Require(rejected, "Out-of-range instance ID silently truncated");
        std::cout << "PASS: GPU intersections after TLAS reuse, movement, metadata edits, removal, growth and empty scenes.\n"
                     "PASS: 100 unchanged frames recorded 0 TLAS builds and kept the same result allocation.\n";
    }
};

int main(int argc, char** argv) {
    try { Require(argc == 2, "Pass the compiled shader path"); Runner r(argv[1]); r.Test(); return 0; }
    catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
