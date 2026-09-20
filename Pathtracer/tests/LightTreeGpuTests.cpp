#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <array>
#include <fstream>
#include <functional>
#include "Lighting/LightTreeRefit.h"
#include "Lighting/LightTreeIncremental.h"

void Check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D12 operation failed"); }
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

std::vector<lt::LightTLASNodeGpu> DecodeTLAS(const std::vector<lt::LightTLASNodePacked>& packed) {
    std::vector<lt::LightTLASNodeGpu> result;
    for (const auto& p : packed) result.push_back(lt::UnpackTLASNode<lt::LightTLASNodeGpu>(p));
    return result;
}

struct Runner {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12Resource> instanceBuffer;
    ComPtr<ID3D12PipelineState> pso, learningPso;
    ComPtr<ID3D12Resource> learningBuffer;
    ComPtr<ID3D12Resource> cameraBuffer;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    UINT increment = 0;
    uint64_t serial = 0;
    UINT currentFrame = 1;
    UINT clockMs = 0;
    float lodScale = .05f;
    float rewardScale = 1.0f;
    bool compactNodes = true;
    XMFLOAT3 testCamera{};
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    ~Runner() { CloseHandle(event); }

    explicit Runner(const char* directory) {
        Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        D3D12_COMMAND_QUEUE_DESC q{};
        Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors = 21;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        increment = device->GetDescriptorHandleIncrementSize(hd.Type);

        CD3DX12_DESCRIPTOR_RANGE range;
        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 21, 0);
        CD3DX12_ROOT_PARAMETER params[6];
        params[0].InitAsDescriptorTable(1, &range);
        params[1].InitAsUnorderedAccessView(0, 1);
        params[2].InitAsConstants(8, 0, 1);
        params[3].InitAsConstants(SHARC_ROOT_CONSTANTS, 1);
        params[4].InitAsUnorderedAccessView(27);
        params[5].InitAsConstantBufferView(0);
        CD3DX12_ROOT_SIGNATURE_DESC desc(6, params);
        ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
        Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        std::ifstream file(std::string(directory) + "/light-tree.dxil", std::ios::binary);
        Require(bool(file), "Missing test shader");
        std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = root.Get(); pd.CS = {code.data(), code.size()};
        Check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
        std::ifstream lf(std::string(directory)+"/light-learning.dxil",std::ios::binary);
        std::vector<char> lc((std::istreambuf_iterator<char>(lf)),{});
        pd.CS={lc.data(),lc.size()};Check(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(&learningPso)));
        learningBuffer=Buffer(LT_LEARNING_BYTES,D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        cameraBuffer=Buffer(4096,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
        std::vector<InstanceProperties> props(4096);
        for(UINT i=0;i<props.size();++i) {
            auto& p=props[i];
            p.objectToWorld=p.objectToWorldInverse=p.objectToWorldNormal=p.prevObjectToWorld=MakeFloat3x4(XMMatrixIdentity());
            p.indexBase=p.vertexBase=p.materialBase=0;p.triToLightBase=0xFFFFFFFFu;p.opaqueTriCount=0;
            p._pad[0]=p._pad[1]=0;p.lightSlot=i;
        }
        instanceBuffer=Upload(props);Srv(instanceBuffer.Get(),3,sizeof(InstanceProperties));Flush();
    }

    D3D12_CPU_DESCRIPTOR_HANDLE Handle(UINT slot) {
        return CD3DX12_CPU_DESCRIPTOR_HANDLE(heap->GetCPUDescriptorHandleForHeapStart(), slot, increment);
    }

    ComPtr<ID3D12Resource> Buffer(UINT64 bytes, D3D12_HEAP_TYPE type,
        D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        ComPtr<ID3D12Resource> result;
        CD3DX12_HEAP_PROPERTIES hp(type);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);
        Check(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&result)));
        return result;
    }

    void Flush() {
        Check(commands->Close());
        ID3D12CommandList* lists[] = {commands.Get()};
        queue->ExecuteCommandLists(1, lists);
        Check(queue->Signal(fence.Get(), ++serial));
        Check(fence->SetEventOnCompletion(serial, event));
        Require(WaitForSingleObject(event, 30000) == WAIT_OBJECT_0, "GPU test timed out");
        Check(allocator->Reset());
        Check(commands->Reset(allocator.Get(), nullptr));
        uploads.clear();
    }

    template<class T> ComPtr<ID3D12Resource> Upload(const std::vector<T>& values) {
        const auto bytes = UINT64(values.size()) * sizeof(T);
        auto staging = Buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* data = nullptr;
        Check(staging->Map(0, nullptr, &data));
        memcpy(data, values.data(), size_t(bytes));
        staging->Unmap(0, nullptr);
        auto result = Buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyBufferRegion(result.Get(), 0, staging.Get(), 0, bytes);
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(result.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
        commands->ResourceBarrier(1, &barrier);
        uploads.push_back(staging);
        return result;
    }

    template<class T> std::vector<T> Read(ID3D12Resource* resource) {
        const auto bytes = resource->GetDesc().Width;
        auto staging = Buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyBufferRegion(staging.Get(), 0, resource, 0, bytes);
        Flush();
        void* data = nullptr;
        Check(staging->Map(0, nullptr, &data));
        std::vector<T> result(size_t(bytes / sizeof(T)));
        memcpy(result.data(), data, size_t(bytes));
        staging->Unmap(0, nullptr);
        return result;
    }

    void Srv(ID3D12Resource* resource, UINT slot, UINT stride, DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN) {
        if (slot == 9u || slot == 10u) stride = 16u;
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Format = format;
        d.Buffer.StructureByteStride = stride;
        const UINT bytes = stride ? stride : (format == DXGI_FORMAT_R32G32_UINT ? 8u : 4u);
        d.Buffer.NumElements = UINT(resource->GetDesc().Width / bytes);
        device->CreateShaderResourceView(resource, &d, Handle(slot));
    }

    void BindLearning(UINT flags, UINT frame=0u, bool reset=false) {
        if (compactNodes) flags |= RS_FLAG_COMPACT_LIGHT_TREE;
        if(frame!=0u) currentFrame=frame;
        {void* cam=nullptr;Check(cameraBuffer->Map(0,nullptr,&cam));memset(cam,0,4096);memcpy(static_cast<float*>(cam)+44,&testCamera,sizeof(testCamera));cameraBuffer->Unmap(0,nullptr);}
        ID3D12DescriptorHeap* heaps[]={heap.Get()};commands->SetDescriptorHeaps(1,heaps);
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootConstantBufferView(5,cameraBuffer->GetGPUVirtualAddress());
        commands->SetComputeRootDescriptorTable(0,heap->GetGPUDescriptorHandleForHeapStart());
        commands->SetComputeRootUnorderedAccessView(4,learningBuffer->GetGPUVirtualAddress());
        UINT push[SHARC_ROOT_CONSTANTS]{};push[0]=8u;push[1]=4u;push[2]=flags;float cellSize=1.0f;memcpy(push+15,&cellSize,4);
        memcpy(push+16,&lodScale,4);
        push[17]=clockMs;
        push[23]=reset?LT_RESET_BIT:0u;push[24]=currentFrame;push[34]=0u;
        commands->SetComputeRoot32BitConstants(3,SHARC_ROOT_CONSTANTS,push,0);
        UINT test[8]{};memcpy(test+4,&testCamera,sizeof(testCamera));memcpy(test+7,&rewardScale,4);
        commands->SetComputeRoot32BitConstants(2,8,test,0);
    }
    void PrepareLearning(UINT flags, UINT frame, bool reset=false, double* gpuMs=nullptr) {
        BindLearning(flags,frame,reset);commands->SetPipelineState(learningPso.Get());
        ComPtr<ID3D12QueryHeap> timestamps;ComPtr<ID3D12Resource> readback;
        if(gpuMs) {
            D3D12_QUERY_HEAP_DESC q{};q.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;q.Count=2;
            Check(device->CreateQueryHeap(&q,IID_PPV_ARGS(&timestamps)));
            readback=Buffer(16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            commands->EndQuery(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        }
        auto barrier=CD3DX12_RESOURCE_BARRIER::UAV(learningBuffer.Get());commands->ResourceBarrier(1,&barrier);
        commands->Dispatch(LT_LEARNING_GROUPS,1,1);commands->ResourceBarrier(1,&barrier);
        if(!reset) {
            commands->SetComputeRoot32BitConstant(3,LT_INITIALIZE_BIT,23);
            commands->Dispatch(LT_LEARNING_GROUPS,1,1);commands->ResourceBarrier(1,&barrier);
        }
        if(gpuMs) {commands->EndQuery(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);commands->ResolveQueryData(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,readback.Get(),0);}
        Flush();
        if(gpuMs) {
            UINT64 frequency;Check(queue->GetTimestampFrequency(&frequency));void* data=nullptr;Check(readback->Map(0,nullptr,&data));
            auto ticks=static_cast<UINT64*>(data);*gpuMs=double(ticks[1]-ticks[0])*1000.0/double(frequency);readback->Unmap(0,nullptr);
        }
    }
    std::vector<XMFLOAT4> LearningSamples(UINT lights,UINT work,UINT mode,UINT seed,UINT flags,double* gpuMs=nullptr) {
        auto output=Buffer(work*sizeof(XMFLOAT4),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        BindLearning(flags);commands->SetPipelineState(pso.Get());
        auto learningBarrier=CD3DX12_RESOURCE_BARRIER::UAV(learningBuffer.Get());commands->ResourceBarrier(1,&learningBarrier);
        commands->SetComputeRootUnorderedAccessView(1,output->GetGPUVirtualAddress());
        UINT test[8]={work,mode,lights,seed};memcpy(test+4,&testCamera,sizeof(testCamera));memcpy(test+7,&rewardScale,4);commands->SetComputeRoot32BitConstants(2,8,test,0);
        ComPtr<ID3D12QueryHeap> timestamps;ComPtr<ID3D12Resource> timestampReadback;
        if(gpuMs) {
            D3D12_QUERY_HEAP_DESC q{};q.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;q.Count=2;
            Check(device->CreateQueryHeap(&q,IID_PPV_ARGS(&timestamps)));
            timestampReadback=Buffer(16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
            commands->EndQuery(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        }
        commands->Dispatch((work+63u)/64u,1,1);
        if(gpuMs) {commands->EndQuery(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);commands->ResolveQueryData(timestamps.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,timestampReadback.Get(),0);}
        auto barrier=CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_GENERIC_READ);
        commands->ResourceBarrier(1,&barrier);auto values=Read<XMFLOAT4>(output.Get());
        if(gpuMs) {UINT64 frequency;Check(queue->GetTimestampFrequency(&frequency));void* p=nullptr;Check(timestampReadback->Map(0,nullptr,&p));
            auto* ticks=static_cast<UINT64*>(p);*gpuMs=double(ticks[1]-ticks[0])*1000.0/double(frequency);timestampReadback->Unmap(0,nullptr);}
        return values;
    }

    void VerifyPdf(const std::vector<float>& expected, bool invalidTree = false, bool noLights = false) {
        auto output = Buffer(expected.size() * sizeof(XMFLOAT4), D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        commands->SetDescriptorHeaps(1, heaps);
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootConstantBufferView(5,cameraBuffer->GetGPUVirtualAddress());
        commands->SetPipelineState(pso.Get());
        commands->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        commands->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        UINT test[4]={UINT(expected.size()),0u,0u,0u};
        commands->SetComputeRoot32BitConstants(2,4,test,0);
        commands->SetComputeRootUnorderedAccessView(4,learningBuffer->GetGPUVirtualAddress());
        UINT push[SHARC_ROOT_CONSTANTS] = {};
        if (noLights) push[2] = RS_FLAG_NO_MESH_LIGHTS;
        if (compactNodes) push[2] |= RS_FLAG_COMPACT_LIGHT_TREE;
        commands->SetComputeRoot32BitConstants(3, SHARC_ROOT_CONSTANTS, push, 0);
        commands->Dispatch((UINT(expected.size()) + 63u) / 64u, 1, 1);
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_GENERIC_READ);
        commands->ResourceBarrier(1, &barrier);
        const auto results = Read<XMFLOAT4>(output.Get());
        for (size_t i = 0; i < expected.size(); ++i) {
            Require(std::isfinite(results[i].x) && std::abs(results[i].x - expected[i]) <= expected[i] * 2e-5f,
                "PDF does not match independently traversed tree");
            Require((invalidTree || noLights) ? (results[i].y == 0 && results[i].z == 0) :
                (results[i].y > 0 && std::abs(results[i].y - results[i].z) <= results[i].y * 2e-5f),
                "Sampling and PDF evaluation disagree");
            Require((invalidTree || noLights) ? results[i].w == -1.0f :
                results[i].w >= 0 && results[i].w < float(expected.size()), "Sampled triangle out of range");
        }
    }
};

template<class Node, class Leaf>
uint32_t Walk(const std::vector<Node>& nodes, uint32_t node, uint32_t depth,
    lt::LightTreeTrail path, float pdf, Leaf&& leaf) {
    Require(node < nodes.size() && depth <= LT_TRAIL_MAX_DEPTH, "Tree exceeds trail capacity");
    const auto& n = nodes[node];
    if (n.childCount == 0) { leaf(n, path, pdf); return depth; }
    Require(n.childCount <= 4 && depth < LT_TRAIL_MAX_DEPTH, "Invalid inner node/depth");
    uint32_t maxDepth = depth;
    for (uint32_t c = 0; c < n.childCount; ++c)
        maxDepth = std::max(maxDepth, Walk(nodes, n.firstChild + c, depth + 1,
            path | (uint64_t(c) << (2u * depth)), pdf / n.childCount, leaf));
    return maxDepth;
}

template<class Node>
void VerifyNoUnreachableNodes(const std::vector<Node>& nodes) {
    if(nodes.empty()) return;
    std::vector<bool> reached(nodes.size());
    std::vector<uint32_t> stack{0};
    while(!stack.empty()) {
        const uint32_t i=stack.back();stack.pop_back();
        Require(i<nodes.size() && !reached[i],"Invalid or duplicate child in compact tree");
        reached[i]=true;
        for(uint32_t c=0;c<nodes[i].childCount;++c) stack.push_back(nodes[i].firstChild+c);
    }
    for(bool live:reached) Require(live,"Builder retained an unreachable placeholder");
}

#include "LightTreePackingTests.h"

void VerifyBuiltTree(Runner& runner, lt::LightTreeBuilder& builder, const std::vector<LightTriangle>& tris,
    bool refit, bool requireDeep) {
    runner.compactNodes = builder.CompactGpuNodes();
    builder.UploadAll(runner.device.Get(), runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(), runner.Handle(9));
    builder.WriteLookupSrvs(runner.device.Get(), runner.Handle(16));
    builder.WriteSlotSrv(runner.device.Get(), runner.Handle(7));

    auto gpu = builder.GetGpu();
    if (refit) {
        lt::TLASRebuilder rebuilder;
        auto result = rebuilder.Build(lt::ComputeBLASLocalRoots(tris), {}, 4);
        gpu.TLASNodes = runner.compactNodes ? runner.Upload(lt::PackTLAS(result.nodes)) : runner.Upload(result.nodes);
        gpu.BLASBitTrail = runner.Upload(result.blasBitTrails);
        runner.Srv(gpu.TLASNodes.Get(), 9, sizeof(lt::LightTLASNodePacked));
        runner.Srv(gpu.BLASBitTrail.Get(), 18, 0, DXGI_FORMAT_R32G32_UINT);
    }
    auto emissive = runner.Upload(tris);
    runner.Srv(emissive.Get(), 6, sizeof(LightTriangle));
    const auto tlas = runner.compactNodes ? DecodeTLAS(runner.Read<lt::LightTLASNodePacked>(gpu.TLASNodes.Get()))
                                          : runner.Read<lt::LightTLASNodeGpu>(gpu.TLASNodes.Get());
    VerifyNoUnreachableNodes(tlas);
    const auto blas = runner.compactNodes ? runner.Read<lt::LightBLASNodePacked>(gpu.BLASNodes.Get())
                                          : std::vector<lt::LightBLASNodePacked>{};
    const auto fullBlas = runner.compactNodes ? std::vector<lt::LightBLASNodeGpu>{}
                                              : runner.Read<lt::LightBLASNodeGpu>(gpu.BLASNodes.Get());
    const auto ranges = runner.Read<lt::BlasRangeGpu>(gpu.BLASRanges.Get());
    const auto indices = runner.Read<uint32_t>(gpu.LeafTriIndex.Get());
    const auto triTrails = runner.Read<lt::LightTreeTrail>(gpu.TriBitTrail.Get());
    const auto blasTrails = runner.Read<lt::LightTreeTrail>(gpu.BLASBitTrail.Get());
    Require(triTrails.size() == tris.size() && blasTrails.size() == ranges.size(), "Trail buffer has wrong stride/count");
    std::vector<float> expected(tris.size());
    uint32_t maxBlasDepth = 0;
    const auto maxTlasDepth = Walk(tlas, 0, 0, 0, 1.0f, [&](const auto& t, uint64_t path, float pdf) {
        Require(path == blasTrails.at(t.slot), "TLAS trail does not reach its slot");
        const auto& r = ranges.at(t.slot);
        std::vector<lt::LightBLASNodeGpu> local;
        if (runner.compactNodes) {
            Require(r.nodeOffset + r.nodeCount + 1u <= blas.size(), "Packed BLAS range exceeds allocation");
            for (uint32_t i=0; i<r.nodeCount; ++i)
                local.push_back(lt::UnpackBLASNode<lt::LightBLASNodeGpu>(blas.data()+r.nodeOffset,i));
        } else {
            Require(r.nodeOffset + r.nodeCount <= fullBlas.size(), "Full BLAS range exceeds allocation");
            local.assign(fullBlas.begin()+r.nodeOffset, fullBlas.begin()+r.nodeOffset+r.nodeCount);
        }
        VerifyNoUnreachableNodes(local);
        maxBlasDepth = std::max(maxBlasDepth, Walk(local, 0, 0, 0, pdf, [&](const auto& b, uint64_t bp, float p) {
            Require(b.triCount == 1, "One-triangle leaf invariant broken");
            const auto tri = indices.at(b.triFirst);
            Require(bp == triTrails.at(tri), "BLAS trail does not reach its triangle");
            Require(expected.at(tri) == 0, "Duplicate triangle leaf");
            expected[tri] = p;
        }));
    });
    for (float p : expected) Require(p > 0, "Missing triangle leaf");
    if (requireDeep) Require(std::max(maxTlasDepth, maxBlasDepth) > 16, "Fixture did not exceed old trail capacity");
    runner.VerifyPdf(expected);
    std::cout << "Built tree" << (refit ? " with refit" : "") << ": TLAS depth " << maxTlasDepth
              << ", BLAS depth " << maxBlasDepth << ", " << tris.size() << " PDF queries passed\n";
}

void VerifyBoundaryTree(Runner& runner, uint32_t depth, bool deepTlas) {
    const bool invalidTree = depth > LT_TRAIL_MAX_DEPTH;
    std::vector<lt::LightBLASNodeGpu> nodes(1);
    std::vector<lt::LightTreeTrail> trails;
    std::vector<uint32_t> indices;
    std::vector<float> expected;
    std::function<void(uint32_t, uint32_t, uint64_t, float)> build = [&](uint32_t ni, uint32_t d, uint64_t path, float p) {
        nodes[ni].bmin = {-1,-1,0}; nodes[ni].bmax = {1,1,1}; nodes[ni].axis = {0,0,1};
        if (d == depth) {
            nodes[ni].triFirst = UINT(indices.size()); nodes[ni].triCount = 1;
            indices.push_back(UINT(indices.size())); trails.push_back(path); expected.push_back(invalidTree ? 0.0f : p);
            return;
        }
        const uint32_t first = UINT(nodes.size());
        const uint32_t children = invalidTree ? 1u : 4u;
        nodes[ni].firstChild = first; nodes[ni].childCount = children; nodes.resize(first + children);
        const uint32_t continuation = invalidTree ? 0u : (d * 3u + 1u) & 3u;
        for (uint32_t c = 0; c < children; ++c) {
            const uint64_t childPath = d < LT_TRAIL_MAX_DEPTH ? path | (uint64_t(c) << (2u * d)) : path;
            build(first + c, c == continuation ? d + 1 : depth, childPath, p / children);
        }
    };
    build(0, 0, 0, 1);
    std::vector<lt::LightTLASNodeGpu> tlas(1);
    std::vector<lt::BlasRangeGpu> ranges(1);
    std::vector<lt::LightTreeTrail> blasTrails(1);
    std::vector<uint32_t> triToBlas(indices.size(), 0);
    if (deepTlas) {
        tlas.resize(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            memcpy(&tlas[i], &nodes[i], 48);
            tlas[i].firstChild = nodes[i].firstChild; tlas[i].childCount = nodes[i].childCount;
            tlas[i].slot = nodes[i].triFirst;
        }
        nodes.assign(indices.size(), {});
        ranges.resize(indices.size());
        blasTrails = trails; trails.assign(indices.size(), 0);
        for (UINT i = 0; i < indices.size(); ++i) {
            nodes[i].triCount = 1; nodes[i].triFirst = i; ranges[i].nodeOffset = i; ranges[i].triIndexOffset = i; triToBlas[i] = i;
        }
    }
    std::vector<lt::LightBLASNodePacked> packed;
    if (deepTlas) {
        for (UINT i=0; i<nodes.size(); ++i) {
            ranges[i].nodeOffset=UINT(packed.size()); ranges[i].nodeCount=1u;
            auto mesh=lt::PackBLAS(std::vector<lt::LightBLASNodeGpu>{nodes[i]});
            packed.insert(packed.end(),mesh.begin(),mesh.end());
        }
    } else {
        ranges[0].nodeCount=UINT(nodes.size()); packed=lt::PackBLAS(nodes);
    }
    std::vector<lt::LightSlotGpu> slots(ranges.size());
    for (UINT i = 0; i < slots.size(); ++i) slots[i] = lt::makeSlotRecord(i, ranges[i].nodeOffset, lt::LT_IDENTITY_4X4);
    auto sg = runner.Upload(slots); runner.Srv(sg.Get(), 7, sizeof(slots[0]));
    auto tg = runner.Upload(lt::PackTLAS(tlas)); runner.Srv(tg.Get(), 9, sizeof(lt::LightTLASNodePacked));
    auto bg = runner.Upload(packed); runner.Srv(bg.Get(), 10, sizeof(lt::LightBLASNodePacked));
    auto rg = runner.Upload(ranges); runner.Srv(rg.Get(), 11, sizeof(ranges[0]));
    auto ig = runner.Upload(indices); runner.Srv(ig.Get(), 12, 0, DXGI_FORMAT_R32_UINT);
    auto mg = runner.Upload(triToBlas); runner.Srv(mg.Get(), 16, 0, DXGI_FORMAT_R32_UINT);
    auto bt = runner.Upload(blasTrails); runner.Srv(bt.Get(), 18, 0, DXGI_FORMAT_R32G32_UINT);
    auto tt = runner.Upload(trails); runner.Srv(tt.Get(), 17, 0, DXGI_FORMAT_R32G32_UINT);
    std::vector<LightTriangle> records(indices.size());
    for (UINT i = 0; i < records.size(); ++i) records[i].meshID = deepTlas ? i : 0u;
    auto eg = runner.Upload(records); runner.Srv(eg.Get(), 6, sizeof(LightTriangle));
    runner.VerifyPdf(expected, invalidTree);
    std::cout << (deepTlas ? "TLAS" : "BLAS") << " boundary depth " << depth << " passed\n";
}

void VerifyConeGeometry(Runner& runner) {
    auto metrics=runner.LearningSamples(2,2,5,0,0);
    Require(std::abs(metrics[0].x-.5f)<1e-6 && metrics[0].y==0,"Back-facing cone importance is incorrect");
    Require(std::abs(metrics[0].z-4*metrics[0].w)<1e-4,"Near-field distance regularization is not scale consistent");
    const auto normal=lt::normalize3({2,3,4});
    Require(lt::length3(lt::sub3({metrics[1].x,metrics[1].y,metrics[1].z},normal))<1e-6f && metrics[1].w==0,"Receiver-normal transform incorrect");
    lt::Cone ca,cb;ca.axis={1,0,0};ca.theta_o=.7f;cb.axis={-1,0,0};cb.theta_o=.1f;
    auto cu=lt::coneUnion(ca,cb);
    Require(lt::safe_acosf(lt::dot3(ca.axis,cu.axis))+ca.theta_o<=cu.theta_o+1e-5f &&
        lt::safe_acosf(lt::dot3(cb.axis,cu.axis))+cb.theta_o<=cu.theta_o+1e-5f,"Antiparallel cone union lost support");
    LightTriangle tri{};tri.x={0,0,0};tri.y={0,2,0};tri.z={3,0,0};tri.weight=2;tri.meshID=0;
    std::vector<LightTriangle> tris{tri};auto roots=lt::ComputeBLASLocalRoots(tris);
    Require(roots.size()==1 && roots[0].localCone.theta_o==0,"Refit cone lost the emitter orientation");
    std::vector<InstanceXformCPU> transforms(1);
    XMStoreFloat4x4(&transforms[0].objectToWorld,XMMatrixScaling(-2,3,4)*XMMatrixRotationY(.4f)*XMMatrixTranslation(100,200,-300));
    lt::LightTreeBuilder builder;builder.Build(tris,transforms);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));
    auto initial=DecodeTLAS(runner.Read<lt::LightTLASNodePacked>(builder.GetGpu().TLASNodes.Get()));
    lt::TLASRebuilder rebuilder;auto refit=rebuilder.Build(roots,transforms);
    const auto& a=initial[0];const auto& b=refit.nodes[0];
    auto rebased=initial;rebased[0].bmin.x-=1000;rebased[0].bmax.x-=1000;
    Require(lt::SameLightTreeTopology(initial,rebased),"Origin rebase invalidates unchanged light topology");
    rebased[0].slot+=1;
    Require(!lt::SameLightTreeTopology(initial,rebased),"Changed leaf identity retained learned node references");
    std::vector<lt::LightTLASNodeGpu> topology(3);topology[0].childCount=2;topology[0].firstChild=1;
    auto changedTopology=topology;changedTopology[0].firstChild=2;
    Require(!lt::SameLightTreeTopology(topology,changedTopology),"Changed edge retained learned node references");
    Require(std::memcmp(&a.bmin,&b.bmin,sizeof(XMFLOAT3))==0 && std::memcmp(&a.bmax,&b.bmax,sizeof(XMFLOAT3))==0,"Initial/refit world bounds disagree");
    lt::LightTreeRefitManager manager;
    manager.RequestRefit(roots,transforms);manager.DiscardPending();
    lt::TLASRefitResult discarded;Require(!manager.IsPending() && !manager.PollResult(discarded),"Obsolete refit survived an emission rebuild");
    std::cout<<"Cone support and initial/refit affine transforms passed\n";
}
uint32_t LearningHash(uint32_t v) {v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);}
void VerifyAdaptiveCells(Runner& runner) {
    std::vector<XMFLOAT4> positions={{.25f,.25f,-9.75f,0},{100000.25f,.25f,-9.75f,0}};
    auto input=runner.Upload(positions);runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.testCamera={0,0,0};runner.PrepareLearning(LT_FLAG_LEARNING,1,true);
    auto cold=runner.LearningSamples(2,2,19,0,LT_FLAG_LEARNING);
    Require(cold[0].x==0 && cold[1].x>=10 && cold[0].y==LT_MAX_LEVEL+1u && cold[1].w>=0,"Adaptive distance LOD or immediate parent coverage failed");
    for(UINT frame=2;frame<9;++frame) {
        runner.LearningSamples(2,2048,11,0,LT_FLAG_LEARNING);runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    auto warm=runner.LearningSamples(2,2,19,0,LT_FLAG_LEARNING);
    Require(warm[0].x==warm[0].y && warm[1].x==warm[1].y,"Cells did not refine to the camera's requested detail");
    runner.testCamera={100000,0,0};
    auto moved=runner.LearningSamples(2,2,19,0,LT_FLAG_LEARNING);
    Require(moved[0].x>=10 && moved[1].x==0 && moved[0].w>=0 && moved[1].w>=0,"Moving the camera lost learned fallback coverage");
    for(UINT frame=9;frame<15;++frame) {
        runner.LearningSamples(2,2048,11,0,LT_FLAG_LEARNING);runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    auto refined=runner.LearningSamples(2,2,19,0,LT_FLAG_LEARNING);
    Require(refined[1].y==0,"New camera region waited for unrelated cells to expire");
    runner.LearningSamples(2,2,17,0,LT_FLAG_LEARNING);
    runner.PrepareLearning(LT_FLAG_LEARNING,15);
    runner.clockMs=LT_CELL_HISTORY_MS+1u;runner.PrepareLearning(LT_FLAG_LEARNING,16);
    auto idle=runner.LearningSamples(2,2,12,1,LT_FLAG_LEARNING);
    for(const auto& c:idle) Require(c.x<=14 && c.y==1,"Unfinished feedback batch kept an idle cell alive forever");
    runner.clockMs=0u;runner.PrepareLearning(LT_FLAG_LEARNING,1,true);
    for(UINT frame=2;frame<9;++frame) {
        runner.LearningSamples(2,2048,11,0,LT_FLAG_LEARNING);runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    runner.LearningSamples(2,4099,17,0,LT_FLAG_LEARNING);
    auto moments=runner.LearningSamples(2,2,18,0,LT_FLAG_LEARNING);
    for(UINT receiver=0;receiver<2;++receiver) {
        double sum=0,squared=0;UINT count=0;
        for(UINT i=receiver;i<4099;i+=2) {double value=float(i%7u)*moments[receiver].w;sum+=value;squared+=value*value;++count;}
        Require(moments[receiver].z==count && std::abs(moments[receiver].x-sum)<std::max(1e-5,sum*2e-5) &&
            std::abs(moments[receiver].y-squared)<std::max(1e-5,squared*2e-5),"Direct feedback changed observation moments");
    }
    runner.testCamera={0,0,0};runner.lodScale=.001f;
#if LT_GRID_CAPACITY >= 131072u
    positions.clear();
    for(UINT i=0;i<65536u;++i) positions.push_back({float(i%256)+.25f,float(i/256)+.25f,-9.75f,0});
    auto fixedInput=runner.Upload(positions);runner.Srv(fixedInput.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.PrepareLearning(LT_FLAG_LEARNING,1,true);
    for(UINT frame=2;frame<10;++frame) {
        runner.LearningSamples(65536,65536,11,0,LT_FLAG_LEARNING);runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    auto fixedCoverage=runner.LearningSamples(65536,65536,19,0,LT_FLAG_LEARNING);UINT fixedFine=0;
    for(const auto& c:fixedCoverage) fixedFine+=c.x==c.y;
    std::cout<<"Expanded-grid fixed workload: "<<fixedFine<<"/65536 receivers at requested detail\n";
    Require(fixedFine>=60000u,"Expanded grid still fails to supply local detail for the old capacity workload");
#endif
    positions.clear();
    for(UINT i=0;i<LT_GRID_CAPACITY*2u;++i) positions.push_back({float(i%256)+.25f,float(i/256)+.25f,-9.75f,0});
    auto crowded=runner.Upload(positions);runner.Srv(crowded.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.PrepareLearning(LT_FLAG_LEARNING,1,true);
    for(UINT frame=2;frame<10;++frame) {
        runner.LearningSamples(UINT(positions.size()),UINT(positions.size()),11,0,LT_FLAG_LEARNING);
        runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    auto coverage=runner.LearningSamples(UINT(positions.size()),UINT(positions.size()),19,0,LT_FLAG_LEARNING);
    UINT fine=0,parent=0;for(const auto& c:coverage) {Require(c.w>=0,"Overloaded grid left a region without learning");fine+=c.x==c.y;parent+=c.y>c.x;}
    Require(fine>0 && parent>0,"Pressure fixture did not exercise fine and fallback cells");
    runner.currentFrame=1024;
    runner.LearningSamples(UINT(positions.size()),UINT(positions.size()),11,0,LT_FLAG_LEARNING);
    runner.PrepareLearning(LT_FLAG_LEARNING,1025);
    auto persistent=runner.LearningSamples(UINT(positions.size()),UINT(positions.size()),19,0,LT_FLAG_LEARNING);
    for(UINT i=0;i<coverage.size();++i) if(coverage[i].x==coverage[i].y)
        Require(persistent[i].w==coverage[i].w,"Active fine cell was recycled merely because it was old");
    std::cout<<"Adaptive coverage: "<<positions.size()<<" receivers, "<<fine<<" requested LOD, "<<parent<<" learned parents; no holes after camera move or overload\n";
    runner.testCamera={100000,0,0};runner.clockMs=LT_CELL_DISTANT_MS+1u;
    positions.clear();for(UINT i=0;i<512;++i) positions.push_back({100000.25f+float(i%32),.25f+float(i/32),-9.75f,0});
    auto arrived=runner.Upload(positions);runner.Srv(arrived.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.PrepareLearning(LT_FLAG_LEARNING,1026);
    for(UINT frame=1027;frame<1051;++frame) {
        runner.LearningSamples(512,4096,11,0,LT_FLAG_LEARNING);runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    auto local=runner.LearningSamples(512,512,19,0,LT_FLAG_LEARNING);UINT detailed=0;
    for(const auto& c:local) detailed+=c.x==c.y;
    Require(detailed>=500,"Camera arrival into a full table waited for long-lived distant cells");
    std::cout<<"Full-table camera arrival: "<<detailed<<"/512 receivers regained local detail\n";
    runner.clockMs=0u;
    runner.lodScale=.05f;runner.testCamera={0,0,0};
}

void VerifyMovement(Runner& runner) {
    constexpr UINT flags=LT_FLAG_LEARNING,lights=128;
    std::vector<std::vector<XMFLOAT4>> buckets(LT_GRID_CAPACITY/LT_BUCKET_SIZE);
    std::vector<XMFLOAT4> collision;
    for(UINT i=0;i<LT_GRID_CAPACITY*8u && collision.empty();++i) {
        UINT x=i%512,y=i/512;
        UINT h=LearningHash(x^LearningHash(y)^LearningHash(UINT(-10))^LearningHash(318u+512u));
        auto& group=buckets[(h&(LT_GRID_CAPACITY-1u))/LT_BUCKET_SIZE];
        group.push_back({float(x)+.25f,float(y)+.25f,-9.75f,0});
        if(group.size()==16) collision=group;
    }
    Require(collision.size()==16,"Could not construct the bucket collision fixture");
    auto bindReceivers=[&](const std::vector<XMFLOAT4>& receivers) {
        auto buffer=runner.Upload(receivers);runner.Srv(buffer.Get(),19,sizeof(XMFLOAT4));runner.Flush();return buffer;
    };
    std::vector<XMFLOAT4> oldView(collision.begin(),collision.begin()+8),newView(collision.begin()+8,collision.end());
    runner.testCamera={0,0,0};runner.lodScale=.001f;runner.clockMs=0xfffffe00u;
    auto oldInput=bindReceivers(oldView);runner.PrepareLearning(flags,1,true);
    for(UINT frame=2;frame<34;++frame) {
        runner.LearningSamples(8,2048,11,frame,flags);runner.PrepareLearning(flags,frame);
    }
    auto oldCells=runner.LearningSamples(8,8,19,0,flags);
    for(const auto& c:oldCells) Require(c.y==0,"Second hash bucket did not resolve primary collisions");
    auto newInput=bindReceivers(newView);runner.clockMs+=1000u;
    for(UINT frame=34;frame<66;++frame) {
        runner.LearningSamples(8,2048,11,frame,flags);runner.PrepareLearning(flags,frame);
    }
    auto newCells=runner.LearningSamples(8,8,19,0,flags);
    for(const auto& c:newCells) Require(c.y==0,"Newly visible colliding cells could not obtain local detail");
    runner.Srv(oldInput.Get(),19,sizeof(XMFLOAT4));
    runner.PrepareLearning(flags,50000);
    auto returned=runner.LearningSamples(8,8,19,0,flags);
    for(UINT i=0;i<8;++i) Require(returned[i].w==oldCells[i].w && returned[i].z==oldCells[i].z,
        "Turning back lost the previous cell or its update history");
    auto protectedCells=runner.LearningSamples(8,8,12,1,flags);
    for(const auto& c:protectedCells) Require(c.y==0,"Nearby history was prematurely eligible for replacement");
    std::cout<<"Turn-away/return, competing hash buckets, frame-rate independence and timer wrap passed\n";

    auto input=bindReceivers({{.25f,.25f,-9.75f,0}});
    runner.testCamera={0,0,0};runner.lodScale=.05f;runner.clockMs=0;
    runner.PrepareLearning(flags,1,true);
    UINT warmFrame=1;
    for(float distance:{180.0f,90.0f,45.0f}) {
        runner.testCamera={.25f,.25f,-9.75f-distance};
        runner.LearningSamples(lights,64,25,0,flags);runner.PrepareLearning(flags,++warmFrame);
    }
    runner.testCamera={0,0,0};
    auto train=[&](UINT first,UINT end,UINT visible) {
        for(UINT frame=first;frame<end;++frame) {
            auto values=runner.LearningSamples(lights,4096,24,(visible<<16u)|((frame*7919u)&65535u),flags);
            for(const auto& v:values) Require(v.x>0 && std::abs(v.x-v.y)<v.x*3e-5f,"Moving-view training changed the declared PDF");
            runner.clockMs=frame*16u;runner.PrepareLearning(flags,frame);
        }
    };
    train(5,133,0);
    auto initialParent=runner.LearningSamples(lights,1,23,1u,flags)[0];
    Require(initialParent.x>.3f && initialParent.y>5,"Parent stopped learning once its fine child became available");
    train(133,261,127);
    auto oldFine=runner.LearningSamples(lights,1,23,0u,flags)[0];
    auto newFine=runner.LearningSamples(lights,1,23,127u<<8u,flags)[0];
    auto oldParent=runner.LearningSamples(lights,1,23,1u,flags)[0];
    auto newParent=runner.LearningSamples(lights,1,23,1u|(127u<<8u),flags)[0];
    Require(newFine.x>.3f && newFine.x>oldFine.x*2,"Mature fine cell could not adapt to changed view contributions");
    Require(newParent.x>.2f && newParent.x>oldParent.x*2 && newParent.y>initialParent.y+20,
        "Coarse parent retained stale lighting while its fine child adapted");
    std::cout<<"Changed-view adaptation: fine probability "<<newFine.x<<", refreshed parent "<<newParent.x<<"\n";

    runner.LearningSamples(lights,1,26,0u,flags);
    runner.LearningSamples(lights,1,26,1u|(127u<<8u),flags);
    std::vector<XMFLOAT4> below,above;
    for(float distance:{33.0f,35.0f,38.0f,39.999f,40.001f,41.0f,40.001f,39.999f,38.0f,35.0f,33.0f}) {
        runner.testCamera={.25f,.25f,-9.75f-distance};
        auto pdfs=runner.LearningSamples(lights,lights,22,0,flags);double sum=0;
        for(const auto& v:pdfs) {
            sum+=v.x;float expected=v.y+(v.z-v.y)*v.w;
            Require(v.x>0 && std::abs(v.x-expected)<2e-6,"LOD PDF is not the full fine/coarse mixture");
        }
        Require(std::abs(sum-1)<2e-5,"LOD transition lost normalization or support");
        auto samples=runner.LearningSamples(lights,16384,25,45343,flags);
        for(const auto& v:samples) Require(v.x>0 && std::abs(v.x-v.y)<v.x*3e-5f,"LOD sample and emitter MIS disagree");
        if(distance==39.999f) below=pdfs;
        if(distance==40.001f) above=pdfs;
    }
    for(UINT i=0;i<lights;++i) Require(std::abs(below[i].x-above[i].x)<1e-5,"Crossing the LOD boundary abruptly changed the proposal");
    std::cout<<"Bidirectional LOD sweep: continuous normalized mixtures and matching sample/MIS PDFs passed\n";
    runner.testCamera={.25f,.25f,-9.75f-36.0f};
    for(UINT level:{0u,1u,2u,3u,25u}) runner.LearningSamples(lights,1,28,level,flags);
    auto mixed=runner.LearningSamples(lights,32768,24,49173u,flags);
    for(UINT level:{0u,1u,2u,3u,25u}) {
        auto batch=runner.LearningSamples(lights,1,27,level,flags)[0];double expected=0;UINT count=0;
        for(const auto& sample:mixed) if(level==0u || sample.w==batch.z) {expected+=sample.z;++count;}
        Require(batch.y==count && std::abs(batch.x-expected)<std::max(.001,expected*1e-4),
            "Mixed-proposal feedback lost observations or used a component PDF instead of the mixture");
#ifdef LT_BLEND_FEEDBACK
        Require(count==(level<=1u?mixed.size():0u),"A participating LOD did not receive every mixture observation");
#else
        Require(count>0,"No ancestor observations in the mixture fixture");
#endif
    }
    std::cout<<"Mixed-proposal fine and ancestor feedback matched independent observation sums\n";
    runner.testCamera={0,0,0};runner.clockMs=0;
}

void VerifyLearning(Runner& runner, bool twoLevel) {
    constexpr UINT count=128;
    std::vector<LightTriangle> tris(count);
    for(UINT i=0;i<count;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;
        t.meshID=twoLevel?i/16:0;
    }
    lt::LightTreeBuilder::Settings settings;settings.compactGpuNodes=runner.compactNodes;
    lt::LightTreeBuilder builder;builder.Build(tris,settings);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));runner.Flush();
    auto roots=lt::ComputeBLASLocalRoots(tris);
    auto mapping=runner.Read<UINT>(builder.GetGpu().TriToBLAS.Get());
    for(UINT i=0;i<count;++i) Require(roots.at(mapping[i]).meshID==tris[i].meshID,"Initial and refit BLAS identities disagree");
    UINT flags=LT_FLAG_LEARNING;
    runner.PrepareLearning(flags,1,true);
    auto before=runner.LearningSamples(count,32768,1,123,flags);
    double baselineSecond=0;
    for(const auto& v:before) baselineSecond+=double(v.w)*v.w/before.size();
    runner.PrepareLearning(flags,2);
    auto initial=runner.LearningSamples(count,count,2,0,flags);
    Require(initial[0].z>=LT_CUT_INITIAL,"Initial light cut was not allocated");
    runner.LearningSamples(count,2048,4,234,flags);
    auto frozen=runner.LearningSamples(count,count,2,0,flags);
    for(UINT i=0;i<count;++i) Require(initial[i].x==frozen[i].x,"Training changed the current frame's PDF");
    runner.PrepareLearning(flags,3);
    auto black=runner.LearningSamples(count,count,2,0,flags);
    Require(black[0].w>0 && black[0].w<initial[0].w*.5f,"Occluded samples did not lower the learned weights");
    double mean=0;uint64_t samples=0;
    for(UINT frame=4;frame<100;++frame) {
        auto values=runner.LearningSamples(count,2048,1,frame*7919,flags);
        for(const auto& v:values) {
            Require(v.y>0 && std::isfinite(v.z) && std::abs(v.y-v.z)<v.y*3e-5f,"Learned sampling and MIS PDF disagree");
            mean+=v.w;++samples;
        }
        runner.PrepareLearning(flags,frame);
    }
    mean/=double(samples)*runner.rewardScale;Require(std::abs(mean-1.0)<.05,"Learned estimator changed the known unit integral");
    auto pdfs=runner.LearningSamples(count,count,2,0,flags);double total=0;
    for(const auto& v:pdfs) { total+=v.x;Require(v.x>0 && v.y==1 && v.z<=LT_CUT_MAX,"Learned cut lost support, overlaps, or exceeded capacity"); }
    Require(std::abs(total-1)<2e-5,"Learned leaf probabilities do not sum to one");
    Require(pdfs[0].z>initial[0].z,"Variance-based cut refinement never occurred");
    auto after=runner.LearningSamples(count,32768,3,45343,flags);double second=0,finalMean=0;
    for(const auto& v:after) { second+=double(v.w)*v.w/after.size();finalMean+=double(v.w)/after.size(); }
    finalMean/=runner.rewardScale;
    Require(std::abs(finalMean-1)<.03 && second<baselineSecond*.3,"Learning did not reduce variance at equal sample count");
    std::cout<<"Learning "<<"cone"<<(twoLevel?" TLAS+BLAS":" BLAS")<<": mean "<<mean<<", cut "<<initial[0].z<<" -> "<<pdfs[0].z<<", second moment "<<baselineSecond<<" -> "<<second<<"\n";
    runner.PrepareLearning(flags,101,true);
    auto reset=runner.LearningSamples(count,count,2,0,flags);Require(reset[0].z==LT_CUT_INITIAL,"Reset did not restore the global prior cut");
    if(runner.rewardScale<1.0f) {
        auto receiver=runner.Upload(std::vector<XMFLOAT4>{{.25f,.25f,-9.75f,0}});
        runner.Srv(receiver.Get(),19,sizeof(XMFLOAT4));runner.Flush();
        for(UINT frame=102;frame<150;++frame) {
            runner.LearningSamples(1,8,15,frame*7919u,flags);runner.PrepareLearning(flags,frame);
            auto sparse=runner.LearningSamples(count,count,2,0,flags);
            Require(sparse[0].w>0 && sparse[0].w<10000.0f*runner.rewardScale,
                "Sparse dim feedback retained or reintroduced an unscaled prior");
        }
        std::cout<<"Sparse dim-feedback startup and inherited priors passed\n";
    }
    if(twoLevel) {VerifyAdaptiveCells(runner);VerifyMovement(runner);}
}

void VerifyCutRecovery(Runner& runner,bool twoLevel=false) {
    constexpr UINT lights=4096,flags=LT_FLAG_LEARNING;
    std::vector<LightTriangle> tris(lights);
    for(UINT i=0;i<lights;++i) {
        auto& t=tris[i];float x=float(i%64)*.0125f,y=float(i/64)*.0125f;
        t.x={x,y,0};t.y={x,y+.005f,0};t.z={x+.005f,y,0};t.weight=1;t.meshID=twoLevel?i/32u:0;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));
    auto input=runner.Upload(std::vector<XMFLOAT4>{{.25f,.25f,-9.75f,0}});runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.testCamera=twoLevel?XMFLOAT3{.25f,.25f,-53.75f}:XMFLOAT3{};
    runner.lodScale=twoLevel?.05f:.001f;runner.clockMs=0;runner.rewardScale=1;
    UINT frame=1u;runner.PrepareLearning(flags,frame,true);
    auto partition=[&]() {
        auto pdfs=runner.LearningSamples(lights,lights,36,0,flags);double sum=0;
        for(const auto& v:pdfs) {
            Require(v.x>0 && v.y==1 && v.z<=LT_CUT_MAX && v.w==1,"Maintained cut lost support, overlaps, capacity or finite bounded history");sum+=v.x;
        }
        Require(std::abs(sum-1)<2e-5,"Maintained cut PDF is not normalized");
    };
    auto train=[&](UINT batches,bool changed) {
        for(UINT i=0;i<batches;++i) {
            ++frame;
            auto values=runner.LearningSamples(lights,4096,34,((frame*7919u)&0x7fffffffu)|(changed?0x80000000u:0u),flags);
            for(const auto& v:values) Require(v.x>0 && std::abs(v.x-v.y)<v.x*3e-5f,"Adaptive cut sample and MIS disagree");
            runner.clockMs=frame*16;runner.PrepareLearning(flags,frame);
            if((i&31u)==31u) partition();
        }
    };
    train(384,false);
    auto mature=runner.LearningSamples(1,1,12,0,flags)[0];
    if(twoLevel) {
        runner.testCamera={.25f,.25f,-19.75f};
        train(1,true);
        auto inherited=runner.LearningSamples(1,1,12,0,flags)[0];
        Require(inherited.z>=29 && inherited.y==0,"Recovery fixture did not inherit a full cut into a new fine cell");
    }
    auto before=runner.LearningSamples(lights,1,23,(lights-1u)<<8u,flags)[0];
    train(512,true);
    auto recovered=runner.LearningSamples(lights,1,23,(lights-1u)<<8u,flags)[0];
    runner.PrepareLearning(flags,++frame,true);train(128,true);
    auto reset=runner.LearningSamples(lights,1,23,(lights-1u)<<8u,flags)[0];
    std::cout<<"Full-cut recovery "<<(twoLevel?"TLAS+BLAS, inherited":"BLAS")<<": initial cut "<<mature.z<<", target PDF "<<before.x<<" -> "<<recovered.x
        <<", reset reference "<<reset.x<<std::endl;
    Require(mature.z>=29,"Recovery fixture did not exhaust the cut budget");
    Require(reset.x>.3f && recovered.x>reset.x*.5f,"Mature cut stays stuck until learned lighting is reset");
    train(256,false);train(512,true);partition();
    auto repeated=runner.LearningSamples(lights,1,23,(lights-1u)<<8u,flags)[0];
    Require(repeated.x>.5f,"Repeated movement left the full cut unable to adapt");
    auto independent=runner.LearningSamples(lights,65536,35,0x80012345u,flags);double mean=0;
    for(const auto& v:independent) mean+=v.z/independent.size();
    Require(std::abs(mean-1)<.04,"Merge/split adaptation changed the known unit integral");
    runner.PrepareLearning(flags,++frame,true);train(4,false);
    auto idle=runner.LearningSamples(lights,1,37,0,flags)[0];
    train(256,true);partition();
    auto resumed=runner.LearningSamples(lights,1,23,(lights-1u)<<8u,flags)[0];
    Require(resumed.y>1000000 && resumed.x>.3f,"Inactivity permanently disabled refinement or old moments froze adaptation");
    std::cout<<"Repeated recovery PDF "<<repeated.x<<", integral "<<mean<<", inactive-cut recovery "<<resumed.x<<" passed\n";
    runner.testCamera={0,0,0};runner.lodScale=.05f;runner.clockMs=0;
}

void VerifyColdLodTransition(Runner& runner) {
    constexpr UINT lights=128,flags=LT_FLAG_LEARNING,target=127;
    std::vector<LightTriangle> tris(lights);
    for(UINT i=0;i<lights;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;t.meshID=i/16u;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));
    auto input=runner.Upload(std::vector<XMFLOAT4>{{.25f,.25f,-9.75f,0}});runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.lodScale=.05f;runner.clockMs=0;runner.rewardScale=1;
    UINT frame=1;
    for(UINT level:{0u,2u,6u}) {
        auto move=[&](float fraction) {runner.testCamera={.25f,.25f,-9.75f-20.0f*std::exp2(float(level)+fraction)};};
        move(.25f);runner.PrepareLearning(flags,++frame,true);
        for(UINT i=0;i<96u;++i) {
            runner.LearningSamples(lights,4096,24,(target<<16u)|((frame*7919u)&65535u),flags);
            runner.PrepareLearning(flags,++frame);
        }
        const auto fine=runner.LearningSamples(lights,1,23,level|(target<<8u),flags)[0];
        Require(fine.x>.8f,"Cold LOD fixture did not learn the local emitter");
        runner.testCamera={.25f,.25f,-1000000.0f};
        runner.LearningSamples(lights,1,41,0,flags);
        const auto root=runner.LearningSamples(lights,1,23,25u|(target<<8u),flags)[0];
        Require(root.x<.1f,"Cold LOD fixture root is too similar to the fine proposal");
        move(.625f);
        runner.LearningSamples(lights,4,25,12345,flags);runner.PrepareLearning(flags,++frame);
        const auto coarse=runner.LearningSamples(lights,1,23,(level+1u)|(target<<8u),flags)[0];
        move(.999f);
        const auto entering=runner.LearningSamples(lights,lights,22,0,flags)[target];
        std::cout<<"Cold LOD "<<level<<" -> "<<level+1u<<": fine PDF "<<fine.x
            <<", root "<<root.x<<", prepared coarse "<<coarse.x<<", boundary "<<entering.x<<std::endl;
        Require(coarse.x>.8f,"Next LOD was not prepared from the available learned fine cell before blending");
        Require(entering.x>.8f,"LOD transition discarded useful local learning for a shared root");
        move(1.001f);auto outside=runner.LearningSamples(lights,lights,22,0,flags);
        move(.999f);auto inside=runner.LearningSamples(lights,lights,22,0,flags);
        double sum=0;
        for(UINT i=0;i<lights;++i) {
            Require(inside[i].x>0 && outside[i].x>0,"Cold LOD transition lost light support");
            Require(std::abs(inside[i].x-outside[i].x)<1e-4,"Cold LOD crossing changed the frozen proposal discontinuously");
            sum+=outside[i].x;
        }
        Require(std::abs(sum-1)<2e-5,"Cold LOD proposal is not normalized");
        move(.625f);
        for(UINT i=0;i<32u;++i) {
            auto values=runner.LearningSamples(lights,4,24,(target<<16u)|((frame*7919u)&65535u),flags);
            for(const auto& v:values) Require(v.x>0 && std::abs(v.x-v.y)<v.x*3e-5f,"Warmup sampling and MIS disagree");
            runner.PrepareLearning(flags,++frame);
        }
        const auto warmed=runner.LearningSamples(lights,1,23,(level+1u)|(target<<8u),flags)[0];
        Require(warmed.y>=16,"Sparse future LOD feedback stalled before the blend");
        Require(runner.LearningSamples(lights,1,23,level,flags)[0].z==fine.z,"Coarse preparation replaced its fine source");
    }
    runner.testCamera={0,0,0};runner.clockMs=0;
}

void VerifyIntermediateDistance(Runner& runner) {
    constexpr UINT lights=128,flags=LT_FLAG_LEARNING;
    std::vector<LightTriangle> tris(lights);
    for(UINT i=0;i<lights;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;t.meshID=i/16u;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));
    auto input=runner.Upload(std::vector<XMFLOAT4>{{.25f,.25f,-9.75f,0}});runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.lodScale=.05f;runner.testCamera={.25f,.25f,-53.75f};runner.clockMs=0;runner.rewardScale=1;
    UINT frame=1;runner.PrepareLearning(flags,frame,true);
    for(UINT i=0;i<6u;++i) {
        if(i==3u) runner.testCamera={0,0,0};
        runner.LearningSamples(lights,64,24,frame*7919u,flags);runner.PrepareLearning(flags,++frame);
    }
    runner.testCamera={.25f,.25f,-49.65f};
    runner.LearningSamples(lights,1,26,1u|(127u<<8u),flags);
    auto before=runner.LearningSamples(lights,1,23,1u|(127u<<8u),flags)[0];
    auto blend=runner.LearningSamples(lights,1,22,0,flags)[0];
    Require(blend.w>.99f,"Sparse LOD fixture is outside the coarse-dominated transition");
    for(UINT i=0;i<512u;++i) {
        auto values=runner.LearningSamples(lights,4,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        for(const auto& v:values) Require(v.x>0 && std::abs(v.x-v.y)<v.x*3e-5f,"Sparse blended sampling and MIS disagree");
        runner.clockMs=frame*16u;runner.PrepareLearning(flags,++frame);
        if(i==255u) {
            auto halfway=runner.LearningSamples(lights,1,23,1u|(127u<<8u),flags)[0];
            std::cout<<"Sparse LOD after 256 frames: "<<halfway.y-before.y<<" updates, target PDF "<<halfway.x<<std::endl;
            Require(halfway.y-before.y>=64,"A participating coarse cell did not receive enough sparse feedback to update");
        }
    }
    auto after=runner.LearningSamples(lights,1,23,1u|(127u<<8u),flags)[0];
    std::cout<<"Sparse intermediate LOD: coarse updates "<<after.y-before.y<<", target PDF "<<after.x<<std::endl;
    Require(after.y-before.y>=64 && after.x>.5f,"The actively sampled coarse cell is starved of feedback at intermediate distances");

    const auto fine=runner.LearningSamples(lights,1,23,0,flags)[0];
    runner.LearningSamples(lights,1,38,0,flags);
    Require(runner.LearningSamples(lights,1,23,1,flags)[0].x<0,"Intermediate eviction fixture did not remove the coarse cell");
    for(UINT i=0;i<16u;++i) {
        runner.LearningSamples(lights,4,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        runner.clockMs=frame*16u;runner.PrepareLearning(flags,++frame);
    }
    const auto restored=runner.LearningSamples(lights,1,23,1,flags)[0];
    const auto retained=runner.LearningSamples(lights,1,23,0,flags)[0];
    Require(restored.x>0 && restored.y>0,"A fine cell prevented rebuilding its missing intermediate parents");
    Require(retained.z==fine.z && retained.y>fine.y,"Restoring intermediate parents replaced the active fine cell");
    for(float distance:{39.999f,40.001f,39.999f}) {
        runner.testCamera={.25f,.25f,-9.75f-distance};
        auto pdfs=runner.LearningSamples(lights,lights,22,0,flags);double sum=0;
        for(const auto& v:pdfs) {Require(v.x>0,"Restored LOD lost light support");sum+=v.x;}
        Require(std::abs(sum-1)<2e-5,"Restored LOD mixture is not normalized");
    }
    std::cout<<"Missing intermediate parents restored while retaining the fine cell\n";
    runner.testCamera={0,0,0};runner.clockMs=0;
}

void VerifyGridReview(Runner& runner) {
    constexpr UINT lights=128,flags=LT_FLAG_LEARNING;
    std::vector<LightTriangle> tris(lights);
    for(UINT i=0;i<lights;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;t.meshID=i/16u;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));
    const auto primary=[](UINT x,UINT y,UINT z,UINT w) {
        return LearningHash(x^LearningHash(y)^LearningHash(z)^LearningHash(w))&(LT_GRID_CAPACITY-LT_BUCKET_SIZE);
    };
    const auto secondary=[](UINT x,UINT y,UINT z,UINT w) {
        return LearningHash(y^LearningHash(z+0x9e3779b9u)^LearningHash(x+w))&(LT_GRID_CAPACITY-LT_BUCKET_SIZE);
    };
    auto collisions=[&](UINT blockedLevel,UINT freeLevel) {
        const float width=std::ldexp(1.0f,int(blockedLevel));
        UINT z=UINT(int(std::floor(-9.75f/width))),w=318u+((blockedLevel+1u)<<9u);
        std::array<UINT,2> blocked={primary(0,0,z,w),secondary(0,0,z,w)};
        Require(blocked[0]!=blocked[1],"Blocking fixture needs two distinct buckets");
        UINT freeZ=UINT(int(std::floor(-9.75f/std::ldexp(1.0f,int(freeLevel))))),freeW=318u+((freeLevel+1u)<<9u);
        for(UINT b:blocked) Require(b!=primary(0,0,freeZ,freeW) && b!=secondary(0,0,freeZ,freeW),
            "Expected free slots overlap the blocked slots");
        std::vector<XMFLOAT4> receivers={{.25f,.25f,-9.75f,0}};
        for(UINT b:blocked) {
            UINT found=0;
            for(int x=-40;x<40 && found<LT_BUCKET_SIZE;++x)
            for(int y=-40;y<40 && found<LT_BUCKET_SIZE;++y)
            for(int iz=-40;iz<40 && found<LT_BUCKET_SIZE;++iz) {
                if(x==0 && y==0 && UINT(iz)==z) continue;
                float px=float(x)*width+.25f,py=float(y)*width+.25f,pz=float(iz)*width+.25f;
                float level=std::log2(std::max(1.0f,std::sqrt(px*px+py*py+pz*pz)*.05f));
                if(level<float(blockedLevel)+.05f || level>float(blockedLevel)+.95f) continue;
                if(primary(UINT(x),UINT(y),UINT(iz),w)!=b) continue;
                receivers.push_back({px,py,pz,float(b+found)});++found;
            }
            Require(found==LT_BUCKET_SIZE,"Could not find intermediate-key collisions");
        }
        return receivers;
    };
    auto input=runner.Upload(collisions(3u,0u));runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.lodScale=.05f;runner.testCamera={0,0,0};runner.clockMs=0;runner.rewardScale=1;
    UINT frame=1;runner.PrepareLearning(flags,frame,true);
    runner.LearningSamples(lights,1,39,3,flags);
    for(UINT i=0;i<32u;++i) {
        Require(runner.LearningSamples(lights,1,40,3,flags)[0].x==1,"Refinement overwrote active colliding cells");
        runner.LearningSamples(lights,64,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        runner.clockMs=frame*16u;runner.PrepareLearning(flags,++frame);
    }
    auto allocated=runner.LearningSamples(lights,1,42,0,flags)[0];
    const bool allocatedFine=allocated.y==0 && allocated.z>0;
    std::cout<<"Blocked intermediate buckets: requested LOD 0, obtained "<<allocated.y
        <<", updates "<<allocated.z<<"; all requested fine slots were initially free\n";

    runner.clockMs=0;runner.PrepareLearning(flags,++frame,true);
    runner.LearningSamples(lights,1,41,0,flags);
    for(UINT i=0;i<128u;++i) {
        auto values=runner.LearningSamples(lights,4096,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        for(const auto& v:values) Require(v.x>0 && std::abs(v.x-v.y)<v.x*3e-5f,"Inherited-scale sampling and MIS disagree");
        runner.clockMs=frame*16u;runner.PrepareLearning(flags,++frame);
    }
    auto scale=runner.LearningSamples(lights,1,42,0,flags)[0];
    auto learned=runner.LearningSamples(lights,1,23,127u<<8u,flags)[0];
    const bool calibrated=scale.y==0 && scale.x<100 && learned.x>.3f;
    std::cout<<"Bright-parent / dim-child: local Q sum "<<scale.x<<", target PDF "<<learned.x<<std::endl;
    Require(allocatedFine && calibrated,"Persistent grid allocation or inherited-scale failure");

    auto fineBlocked=runner.Upload(collisions(0u,1u));runner.Srv(fineBlocked.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.PrepareLearning(flags,++frame,true);runner.LearningSamples(lights,1,39,0,flags);
    for(UINT i=0;i<32u;++i) {
        Require(runner.LearningSamples(lights,1,40,0,flags)[0].x==1,"Fallback overwrote active colliding cells");
        runner.LearningSamples(lights,64,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        runner.PrepareLearning(flags,++frame);
    }
    auto fallback=runner.LearningSamples(lights,1,42,0,flags)[0];
    Require(fallback.y==1 && fallback.z>0,"Blocked fine buckets did not obtain a learning spatial fallback");
    runner.LearningSamples(lights,1,43,0,flags);
    runner.LearningSamples(lights,64,24,(127u<<16u)|((frame*7919u)&65535u),flags);runner.PrepareLearning(flags,++frame);
    auto reclaimed=runner.LearningSamples(lights,1,42,0,flags)[0];
    Require(reclaimed.y==0,"Freed fine buckets did not regain requested LOD on the next prepare");
    std::cout<<"Blocked fine buckets: learning LOD "<<fallback.y<<", recovered LOD "<<reclaimed.y<<" without reset\n";

    const UINT pressureStart=0xfffffe00u;
    runner.clockMs=pressureStart;runner.PrepareLearning(flags,++frame,true);
    runner.LearningSamples(lights,1,39,0,flags);
    for(UINT i=0;i<8u;++i) {
        runner.LearningSamples(lights,64,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        runner.clockMs+=16u;runner.PrepareLearning(flags,++frame);
    }
    auto grace=runner.LearningSamples(lights,1,42,0,flags)[0];
    Require(grace.y>0,"Nearby cells were evicted during the short inactivity grace period");
    UINT firstFineMs=0u;
    for(UINT i=0;i<48u;++i) {
        runner.LearningSamples(lights,64,24,(127u<<16u)|((frame*7919u)&65535u),flags);
        runner.clockMs+=16u;runner.PrepareLearning(flags,++frame);
        if(firstFineMs==0u && runner.LearningSamples(lights,1,42,0,flags)[0].y==0)
            firstFineMs=runner.clockMs-pressureStart;
    }
    auto shortMove=runner.LearningSamples(lights,1,42,0,flags)[0];
    std::cout<<"Nearby idle bucket pressure after "<<runner.clockMs-pressureStart<<" ms: LOD "<<shortMove.y
        <<", local updates "<<shortMove.z<<", first fine cell at "<<firstFineMs<<" ms"<<std::endl;
    Require(shortMove.y==0 && shortMove.z>8 && firstFineMs<=320u,
        "Short camera movement leaves local detail blocked by ten-second nearby history");

    runner.clockMs=0;runner.PrepareLearning(flags,++frame,true);
    runner.LearningSamples(lights,1,39,0,flags);
    for(UINT i=0;i<8u;++i) {
        runner.LearningSamples(lights,64,24,frame,flags);
        runner.clockMs+=16u;runner.PrepareLearning(flags,++frame);
    }
    runner.clockMs=300u;runner.PrepareLearning(flags,++frame);
    runner.LearningSamples(lights,64,24,frame,flags);
    Require(runner.LearningSamples(lights,1,44,0,flags)[0].x==1,
        "Late-feedback fixture did not queue exactly one pressure replacement");
    Require(runner.LearningSamples(lights,1,40,0,flags)[0].x==1,"Request mutated a frozen cell");
    runner.clockMs+=16u;runner.PrepareLearning(flags,++frame);
    Require(runner.LearningSamples(lights,1,40,0,flags)[0].x==1,
        "Pressure replacement discarded feedback arriving after its request");
    Require(runner.LearningSamples(lights,1,42,0,flags)[0].y>0,
        "Pressure replacement evicted a receiver active in the last frame");
    std::cout<<"Late feedback cancelled queued pressure replacement; active keys survived\n";

    runner.PrepareLearning(flags,++frame,true);runner.testCamera={.25f,.25f,-53.75f};runner.rewardScale=1e6f;
    for(UINT i=0;i<64u;++i) {
        runner.LearningSamples(lights,4096,24,(frame*7919u)&65535u,flags);runner.PrepareLearning(flags,++frame);
    }
    const auto bright=runner.LearningSamples(lights,1,42,0,flags)[0];
    Require(bright.y==1 && bright.w==1 && bright.x>1e5,"Movement fixture did not learn its bright spatial parent");
    runner.testCamera={0,0,0};runner.rewardScale=1e-6f;
    for(UINT i=0;i<256u;++i) {
        runner.LearningSamples(lights,64,24,(127u<<16u)|((frame*7919u)&65535u),flags);runner.PrepareLearning(flags,++frame);
    }
    const auto dim=runner.LearningSamples(lights,1,42,0,flags)[0];
    const auto dimPdf=runner.LearningSamples(lights,1,23,127u<<8u,flags)[0];
    Require(dim.y==0 && dim.w==1 && dim.x<100*runner.rewardScale && dimPdf.x>.5f,
        "Bright-to-dim movement retained inherited scale despite local production feedback");
    std::cout<<"Bright-to-dim movement: Q sum "<<bright.x<<" -> "<<dim.x<<", new emitter PDF "<<dimPdf.x<<" without reset\n";
    runner.rewardScale=1;
    runner.clockMs=0;runner.testCamera={0,0,0};
}

void VerifySurfaceLearning(Runner& runner) {
    constexpr UINT lights=128,flags=LT_FLAG_LEARNING;
    std::vector<LightTriangle> tris(lights);
    for(UINT i=0;i<lights;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;t.meshID=i/16u;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));
    auto bind=[&](float roughness,float coat=0.0f,float coatRoughness=1.0f) {
        auto input=runner.Upload(std::vector<XMFLOAT4>{{.25f,.25f,-9.75f,roughness},{coat,coatRoughness,0,0}});
        runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();return input;
    };
    auto input=bind(.8f);runner.testCamera={0,0,0};runner.lodScale=.05f;runner.clockMs=0;runner.rewardScale=1;
    UINT frame=1;runner.PrepareLearning(flags,frame,true);
    for(UINT i=0;i<96;++i) {
        runner.LearningSamples(lights,4096,24,(frame*7919u)&65535u,flags);runner.PrepareLearning(flags,++frame);
    }
    Require(runner.LearningSamples(lights,1,23,0,flags)[0].x>.5f,"Surface fixture did not establish a distinct learned proposal");
    auto check=[&](const char* name,float roughness,float distance,UINT testFlags,float coat=0,float coatRoughness=1) {
        input=bind(roughness,coat,coatRoughness);runner.testCamera={.25f,.25f,-9.75f-distance};
        auto values=runner.LearningSamples(lights,131072,45,48391,testFlags);
        double mean=0,rate=0;
        for(const auto& v:values) {
            Require(v.x>0 && std::abs(v.x-v.y)<=v.x*3e-5f,"Light-tree sampling and PDF evaluation disagree");
            Require(v.w==(testFlags!=0u?1.0f:0.0f),"Surface-learning enable state was not propagated");
            mean+=v.z/values.size();rate+=v.w/values.size();
        }
        std::cout<<name<<": learned fraction "<<rate<<", MIS integral "<<mean<<std::endl;
        Require(std::abs(mean-1)<.015,"Surface-learning toggle changed the known MIS integral");
    };
    check("Glossy receiver",.2f,10,flags);
    check("Nearby rough receiver",.8f,10,flags);
    check("Roughness transition",.5f,10,flags);
    auto wrong=runner.LearningSamples(lights,131072,46,48391,flags);double wrongMean=0;
    for(const auto& v:wrong) wrongMean+=v.z/wrong.size();
    Require(std::abs(wrongMean-1)>.1,"MIS negative control was insensitive to a mismatched proposal");
    std::cout<<"Mismatched-MIS negative control: "<<wrongMean<<" (must differ from one)\n";
    check("Glossy clearcoat",.8f,10,flags,1,.2f);
    check("Rough clearcoat",.8f,10,flags,1,.8f);
    check("Far diffuse receiver",1,100,flags);
    check("Diffuse LOD transition",1,20.0f*std::exp2(1.5f),flags);
    check("Far roughness transition",.5f,100,flags);
    check("Very distant diffuse receiver",1,10000,flags);
    check("Distant glossy receiver",.2f,100,flags);
    check("Disabled fallback",.8f,10,0);
    runner.testCamera={0,0,0};runner.clockMs=0;
}

void VerifyAtomicAccumulation(Runner& runner) {
    using Exact = std::array<uint32_t,10>;
    auto add=[](Exact& total,float value) {
        uint32_t bits;memcpy(&bits,&value,4);
        UINT exponent=(bits>>23u)&255u;
        uint64_t mantissa=(bits&0x7fffffu)|(exponent?0x800000u:0u);
        UINT shift=std::max(exponent,1u)-1u,index=shift/32u;
        uint64_t pending=mantissa<<(shift%32u);
        while(pending) {
            uint64_t next=uint64_t(total[index])+(pending&0xffffffffu);
            total[index++]=uint32_t(next);pending=(pending>>32u)+(next>>32u);
        }
    };
    auto fromBits=[](UINT bits) {float value;memcpy(&value,&bits,4);return value;};
    auto check=[&](const std::vector<XMFLOAT4>& input,UINT work,bool carrySeed) {
        auto resource=runner.Upload(input);runner.Srv(resource.Get(),19,sizeof(XMFLOAT4));runner.Flush();
        runner.LearningSamples(1,1,carrySeed?33:29,0,LT_FLAG_LEARNING);
        runner.LearningSamples(UINT(input.size()),work,30,0,LT_FLAG_LEARNING);
        auto words=runner.LearningSamples(1,10,31,0,LT_FLAG_LEARNING);
        auto decoded=runner.LearningSamples(1,1,32,0,LT_FLAG_LEARNING)[0];
        Require(decoded.z==float(work),"Contended feedback lost zero or positive observations");
        for(UINT stat=0;stat<2;++stat) {
            Exact expected{};
            if(carrySeed) for(UINT j=0;j<8;++j) expected[j]=~0u;
            double sum=carrySeed?std::ldexp(1.0,107):0.0;
            for(UINT i=0;i<work;++i) {const auto& v=input[i%input.size()];float value=stat?v.y:v.x;add(expected,value);sum+=value;}
            for(UINT j=0;j<5;++j) {
                const auto& v=words[stat*5+j];
                UINT low=UINT(v.x)|(UINT(v.y)<<16u),high=UINT(v.z)|(UINT(v.w)<<16u);
                Require(low==expected[2*j] && high==expected[2*j+1],"Native atomic accumulation lost significand bits or a carry");
            }
            float actual=stat?decoded.y:decoded.x;double capped=std::min(sum,double(3e38f));
            if(!(std::isfinite(actual) && std::abs(actual-capped)<=std::max(1e-37,capped*3e-7)))
                std::cerr<<"Accumulator decode: work="<<work<<" seed="<<carrySeed<<" stat="<<stat<<" actual="<<actual<<" expected="<<capped<<std::endl;
            Require(std::isfinite(actual) && std::abs(actual-capped)<=std::max(1e-37,capped*3e-7),"Accumulated feedback decoded incorrectly");
        }
    };
    std::vector<XMFLOAT4> range;
    for(UINT exponent=0;exponent<255;++exponent) {
        UINT fraction=(exponent*7919u)&0x7fffffu;
        range.push_back({fromBits((exponent<<23u)|fraction),fromBits((exponent<<23u)|0x7fffffu),0,0});
    }
    range.push_back({0,0,0,0});check(range,65539,false);
    check({{0,0,0,0},{1,1,0,0},{3.25f,10.5625f,0,0},{6,36,0,0}},2073603,false);
    check({{fromBits(1u),fromBits(0x7fffffu),0,0}},65539,false);
    check({{fromBits(1u),fromBits(1u),0,0}},4099,true);
    check({{0,0,0,0}},4099,false);
    std::cout<<"Native atomic sums: full FP32 exponent range, exact carries, 1080p contention, zeros and record reuse passed\n";
}

void BenchmarkLearning(Runner& runner, bool hotOnly=false) {
    constexpr UINT lights=128,receiversCount=512,work=1920*1080;
    std::vector<LightTriangle> tris(lights);
    for(UINT i=0;i<lights;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;t.meshID=0;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));builder.WriteSlotSrv(runner.device.Get(),runner.Handle(7));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));
    runner.lodScale=.001f;
    std::vector<XMFLOAT4> receivers;
    for(UINT i=0;i<receiversCount;++i) receivers.push_back({float(i%32)+.25f,float(i/32)+.25f,-9.75f,0});
    auto input=runner.Upload(receivers);runner.Srv(input.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.PrepareLearning(LT_FLAG_LEARNING,1,true);
    for(UINT frame=2;frame<140;++frame) {
        runner.LearningSamples(receiversCount,receiversCount*256u,15,frame*7919u,LT_FLAG_LEARNING);
        runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    auto coverage=runner.LearningSamples(receiversCount,receiversCount,12,0,LT_FLAG_LEARNING);
    for(const auto& v:coverage) Require(v.x==1,"Benchmark receiver was not allocated");
    ComPtr<IDXGIFactory4> factory;Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    ComPtr<IDXGIAdapter1> adapter;Check(factory->EnumAdapterByLuid(runner.device->GetAdapterLuid(),IID_PPV_ARGS(&adapter)));
    DXGI_ADAPTER_DESC1 desc{};Check(adapter->GetDesc1(&desc));std::wcout<<L"GPU: "<<desc.Description<<L"\n";
    double cut=0;for(const auto& v:coverage) cut+=v.z/receiversCount;
    std::cout<<"1080p microbenchmark: 2,073,600 queries, 512 receivers, mean mature cut "<<cut<<"; median of five GPU dispatches\n";
    if(hotOnly) {
        for(UINT mode:{17u,15u}) for(UINT pixels:{65536u,262144u,1048576u,2073600u,8294400u}) {
            std::array<double,3> ms{};
            for(UINT i=0;i<ms.size();++i) {
                runner.LearningSamples(1,pixels,mode,54321u+i,LT_FLAG_LEARNING,&ms[i]);
                runner.PrepareLearning(LT_FLAG_LEARNING,runner.currentFrame+1u);
            }
            std::sort(ms.begin(),ms.end());
            std::cout<<"one cell "<<(mode==17u?"one-cluster feedback":"sample+feedback")<<" "<<pixels<<": "<<ms[1]<<" ms"<<std::endl;
            if(ms[1]>100.0) break;
        }
        return;
    }
    for(bool transition:{false,true}) {
    runner.lodScale=transition?.05f:.001f;runner.testCamera=transition?XMFLOAT3{15.5f,7.5f,-45.75f}:XMFLOAT3{};
    if(transition) for(UINT i=0;i<138u;++i) {
        runner.LearningSamples(receiversCount,receiversCount*256u,15,(i+140u)*7919u,LT_FLAG_LEARNING);
        runner.PrepareLearning(LT_FLAG_LEARNING,runner.currentFrame+1u);
    }
    std::cout<<(transition?"LOD transition band\n":"Stable LOD\n");
    for(UINT divergent:{0u,0x80000000u}) for(UINT mode : {14u,15u,16u}) for(UINT flags : {0u,UINT(LT_FLAG_LEARNING)}) {
        auto warmup=runner.LearningSamples(receiversCount,work,mode,123u|divergent,flags);
        if(mode==16u) for(const auto& value:warmup)
            Require(value.y>0 && std::abs(value.y-value.z)<value.y*3e-5f,
                "Mature-cut sampling and matching PDF disagree");
        std::array<double,5> ms{};
        for(UINT i=0;i<ms.size();++i) runner.LearningSamples(receiversCount,work,mode,(54321+i)|divergent,flags,&ms[i]);
        std::sort(ms.begin(),ms.end());
        const char* label=mode==14?"sample":mode==13?"feedback":mode==15?"sample+feedback":"sample+PDF";
        std::cout<<(divergent?"scattered ":"coherent ")<<label<<" "<<(flags?"learning":"ordinary")<<": "<<ms[2]<<" ms\n";
    }
    for(bool idle:{false,true}) {
        std::array<double,5> ms{};
        for(UINT i=0;i<ms.size();++i) {
            if(!idle) runner.LearningSamples(receiversCount,work,15,54321u+i,LT_FLAG_LEARNING);
            runner.PrepareLearning(LT_FLAG_LEARNING,runner.currentFrame+1u,false,&ms[i]);
        }
        std::sort(ms.begin(),ms.end());std::cout<<(idle?"idle":"active")<<" update+initialize: "<<ms[2]<<" ms\n";
    }
    }
    constexpr UINT crowdedCount=65536u;
    receivers.clear();
    for(UINT i=0;i<crowdedCount;++i) receivers.push_back({float(i%256)+.25f,float(i/256)+.25f,-9.75f,0});
    auto crowded=runner.Upload(receivers);runner.Srv(crowded.Get(),19,sizeof(XMFLOAT4));runner.Flush();
    runner.lodScale=.001f;runner.testCamera={0,0,0};runner.PrepareLearning(LT_FLAG_LEARNING,1,true);
    for(UINT frame=2;frame<10;++frame) {
        runner.LearningSamples(crowdedCount,crowdedCount,11,0,LT_FLAG_LEARNING);runner.PrepareLearning(LT_FLAG_LEARNING,frame);
    }
    std::cout<<"Capacity pressure: 65,536 receivers, scattered 1080p queries\n";
    for(UINT mode:{15u,16u}) {
        std::array<double,5> ms{};
        for(UINT i=0;i<ms.size();++i) {
            runner.LearningSamples(crowdedCount,work,mode,(54321u+i)|0x80000000u,LT_FLAG_LEARNING,&ms[i]);
            runner.PrepareLearning(LT_FLAG_LEARNING,runner.currentFrame+1u);
        }
        std::sort(ms.begin(),ms.end());
        std::cout<<(mode==15u?"sample+feedback":"sample+PDF")<<": "<<ms[2]<<" ms\n";
    }
}
int main(int argc, char** argv) {
    try {
        Require(argc == 2 || argc == 3, "Expected shader directory and optional --benchmark");
        for (uint32_t d = 0; d < LT_TRAIL_MAX_DEPTH; ++d) {
            const uint32_t capacity = uint32_t(uint64_t{1} << (31u - d));
            Require(!lt::LightTreeNeedsBalancedSplit(capacity, d), "Unnecessary median split");
            Require(lt::LightTreeNeedsBalancedSplit(capacity + 1u, d), "Missing depth budget split");
        }
        Require(lt::LightTreeNeedsBalancedSplit(UINT32_MAX, 0), "Maximum count must be balanced");
        bool rejected = false;
        try { lt::AppendLightTreeTrail(0, 1, 32); } catch (const std::logic_error&) { rejected = true; }
        Require(rejected, "Overflow must not silently truncate");
        Runner runner(argv[1]);
        if(argc==3 && std::string(argv[2])=="--surface") {VerifySurfaceLearning(runner);return 0;}
        if(argc==3 && std::string(argv[2])=="--cold-lod") {VerifyColdLodTransition(runner);return 0;}
        if(argc==3) {std::string option=argv[2];if(option=="--grid-review"){VerifyGridReview(runner);return 0;}if(option=="--lod"){VerifyIntermediateDistance(runner);return 0;}if(option=="--recovery"){VerifyCutRecovery(runner);VerifyCutRecovery(runner,true);return 0;}Require(option=="--benchmark" || option=="--hot-benchmark","Unknown option");BenchmarkLearning(runner,option=="--hot-benchmark");return 0;}
        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv{};
        nullSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        nullSrv.Buffer.NumElements = 1;
        nullSrv.Buffer.StructureByteStride = 16;
        for (UINT i = 0; i < 21; ++i) runner.device->CreateShaderResourceView(nullptr, &nullSrv, runner.Handle(i));
        runner.VerifyPdf({0.0f}, false, true);
        runner.Srv(runner.instanceBuffer.Get(), 3, sizeof(InstanceProperties));
        std::cout << "Empty mesh-light sampling with null resources passed\n";
        VerifyLightPacking(runner);
        for (uint32_t d : {0u, 16u, 17u, 31u, 32u, 33u})
            for (bool tlas : {false, true}) VerifyBoundaryTree(runner, d, tlas);

        lt::LightTreeBuilder::Settings settings;
        settings.buildBins = 4;
        for (bool instances : {false, true}) {
            std::vector<LightTriangle> tris(180);
            for (uint32_t i = 0; i < tris.size(); ++i) {
                const float x = std::ldexp(1.0f, int(i) - 60);
                tris[i].x = {x, 0, 0}; tris[i].y = {x, 1, 0}; tris[i].z = {x, 0, 1};
                tris[i].meshID = instances ? i : 0;
            }
            lt::LightTreeBuilder builder;
            // Reuse the builder and descriptor heap across both directions of a layout change.
            for (bool compact : {false, true, false, true}) {
                settings.compactGpuNodes = compact;
                builder.Build(tris, settings);
                VerifyBuiltTree(runner, builder, tris, false, true);
                if (instances) VerifyBuiltTree(runner, builder, tris, true, true);
            }
            for (auto& t : tris) { t.x = {0,0,0}; t.y = {0,1,0}; t.z = {0,0,1}; }
            builder.Build(tris, settings);
            VerifyBuiltTree(runner, builder, tris, instances, false);
        }
        VerifyConeGeometry(runner);
        runner.compactNodes=false;
        VerifyLearning(runner,false);
        VerifyLearning(runner,true);
        runner.compactNodes=true;
        VerifyLearning(runner,false);
        VerifyLearning(runner,true);
        runner.rewardScale=1e-6f;
        VerifyLearning(runner,false);
#ifdef LT_ACCUMULATOR_WORDS
        VerifyAtomicAccumulation(runner);
#endif
        VerifyCutRecovery(runner);
        VerifyCutRecovery(runner,true);
        VerifyIntermediateDistance(runner);
        VerifyGridReview(runner);
        VerifySurfaceLearning(runner);
        VerifyColdLodTransition(runner);
        std::cout << "All light-tree tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
