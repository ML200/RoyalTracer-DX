// GPU regression for ray-t reconstruction errors on the city floor triangles.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3dx12.h>
#include <DirectXMath.h>
#include <fstream>
#include <string>
#include <iostream>
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
        params[1].InitAsUnorderedAccessView(0, 1);
        params[2].InitAsConstants(4, 0, 1);
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
        output = create_buffer(device.Get(), 512 * 512 * 16, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, HEAP_DEFAULT);
        readback = create_buffer(device.Get(), 512 * 512 * 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, HEAP_READBACK);
        const float v[] = {-250,1,250, 250,1,250, 250,1,-250, -250,1,250, 250,1,-250, -250,1,-250};
        vertices = create_buffer(device.Get(), sizeof(v), D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, HEAP_UPLOAD);
        void* mapped; Check(vertices->Map(0, nullptr, &mapped));
        memcpy(mapped, v, sizeof(v)); vertices->Unmap(0, nullptr);
        D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
        geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
        geometry.Triangles.VertexBuffer = {vertices->GetGPUVirtualAddress(), 12};
        geometry.Triangles.VertexCount = 6;
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
    void Test() {
        const float transform[] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
        tlas.begin();
        tlas.add_instance(blas->GetGPUVirtualAddress(), transform, 0, 0,
            D3D12_RAYTRACING_INSTANCE_FLAG_NONE);
        tlas.build(commands.Get());
        for (float height : {1.0001f,1.1f,1.5f,10.0f,100.0f,1000.0f,-100.0f,-1000.0f}) {
            const float constants[] = {-1.5f,height,3.5f,0};
            commands->SetPipelineState(pso.Get());
            commands->SetComputeRootSignature(root.Get());
            commands->SetComputeRootShaderResourceView(0,tlas.tlas_address());
            commands->SetComputeRootUnorderedAccessView(1,output->GetGPUVirtualAddress());
            commands->SetComputeRoot32BitConstants(2,4,constants,0);
            commands->Dispatch(64,64,1);
            D3D12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->ResourceBarrier(1, &barrier);
            commands->CopyResource(readback.Get(), output.Get());
            std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
            commands->ResourceBarrier(1, &barrier);
            Flush();

            float* data;
            Check(readback->Map(0, nullptr, (void**)&data));
            unsigned badRay = 0, badBary = 0;
            float lo = 1, hi = -1;
            for (int i = 0; i < 512 * 512; i++) {
                Require(data[i * 4 + 3] >= 0, "Primary ray missed the floor");
                Require(std::abs(data[i * 4 + 1]) < 2e-7f, "Barycentric floor position left its surface");
                lo = std::min(lo, data[i * 4]);
                hi = std::max(hi, data[i * 4]);
                badRay += unsigned(data[i * 4 + 2]);
                badBary += unsigned(data[i * 4 + 3]);
            }
            readback->Unmap(0, nullptr);
            std::cout << "height " << height << " ray-position error [" << lo << "," << hi
                << "], self-hits ray " << badRay << ", bary " << badBary << std::endl;
            Require(badBary == 0, "Reconstructed surface point self-intersects the floor");
        }
        std::cout << "PASS: floor surface precision (8 views, 262144 rays per view)" << std::endl;
    }
};

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "Pass the compiled surface-precision shader path");
        Runner r(argv[1]);
        r.Test();
        return 0;
    }
    catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
