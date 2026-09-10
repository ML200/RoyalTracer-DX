// Headless tests of the production builder, refitter, GPU uploads and HLSL PDFs.
#define NOMINMAX
#include <windows.h>
#include <dxgi1_6.h>
#include <array>
#include <fstream>
#include <functional>
#include "Lighting/LightTreeRefit.h"

void Check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D12 operation failed"); }
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Runner {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso, learningPso;
    ComPtr<ID3D12Resource> learningBuffer;
    std::vector<ComPtr<ID3D12Resource>> uploads;
    UINT increment = 0;
    uint64_t serial = 0;
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
        hd.NumDescriptors = 19;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
        increment = device->GetDescriptorHandleIncrementSize(hd.Type);

        CD3DX12_DESCRIPTOR_RANGE range;
        range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 19, 0);
        CD3DX12_ROOT_PARAMETER params[5];
        params[0].InitAsDescriptorTable(1, &range);
        params[1].InitAsUnorderedAccessView(0, 1);
        params[2].InitAsConstants(4, 0, 1);
        params[3].InitAsConstants(47, 1);
        params[4].InitAsUnorderedAccessView(27);
        CD3DX12_ROOT_SIGNATURE_DESC desc(5, params);
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
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Format = format;
        d.Buffer.StructureByteStride = stride;
        const UINT bytes = stride ? stride : (format == DXGI_FORMAT_R32G32_UINT ? 8u : 4u);
        d.Buffer.NumElements = UINT(resource->GetDesc().Width / bytes);
        device->CreateShaderResourceView(resource, &d, Handle(slot));
    }

    void BindLearning(UINT flags, UINT frame=1u, bool reset=false) {
        ID3D12DescriptorHeap* heaps[]={heap.Get()};commands->SetDescriptorHeaps(1,heaps);
        commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootDescriptorTable(0,heap->GetGPUDescriptorHandleForHeapStart());
        commands->SetComputeRootUnorderedAccessView(4,learningBuffer->GetGPUVirtualAddress());
        UINT push[47]{};push[9]=flags;float cellSize=1.0f;memcpy(push+24,&cellSize,4);
        push[45]=reset?LT_RESET_BIT:0u;push[46]=frame;
        commands->SetComputeRoot32BitConstants(3,47,push,0);
    }
    void PrepareLearning(UINT flags, UINT frame, bool reset=false) {
        BindLearning(flags,frame,reset);commands->SetPipelineState(learningPso.Get());
        auto barrier=CD3DX12_RESOURCE_BARRIER::UAV(learningBuffer.Get());commands->ResourceBarrier(1,&barrier);
        commands->Dispatch(LT_LEARNING_GROUPS,1,1);commands->ResourceBarrier(1,&barrier);Flush();
    }
    std::vector<XMFLOAT4> LearningSamples(UINT lights,UINT work,UINT mode,UINT seed,UINT flags) {
        auto output=Buffer(work*sizeof(XMFLOAT4),D3D12_HEAP_TYPE_DEFAULT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        BindLearning(flags);commands->SetPipelineState(pso.Get());
        commands->SetComputeRootUnorderedAccessView(1,output->GetGPUVirtualAddress());
        UINT test[]={work,mode,lights,seed};commands->SetComputeRoot32BitConstants(2,4,test,0);
        commands->Dispatch((work+63u)/64u,1,1);
        auto barrier=CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_GENERIC_READ);
        commands->ResourceBarrier(1,&barrier);return Read<XMFLOAT4>(output.Get());
    }

    void VerifyPdf(const std::vector<float>& expected, bool invalidTree = false, bool noLights = false) {
        auto output = Buffer(expected.size() * sizeof(XMFLOAT4), D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        commands->SetDescriptorHeaps(1, heaps);
        commands->SetComputeRootSignature(root.Get());
        commands->SetPipelineState(pso.Get());
        commands->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        commands->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        UINT test[4]={UINT(expected.size()),0u,0u,0u};
        commands->SetComputeRoot32BitConstants(2,4,test,0);
        commands->SetComputeRootUnorderedAccessView(4,learningBuffer->GetGPUVirtualAddress());
        UINT push[SHARC_ROOT_CONSTANTS] = {};
        if (noLights) push[9] = RS_FLAG_NO_MESH_LIGHTS;
        commands->SetComputeRoot32BitConstants(3, 47, push, 0);
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

// Walk topology independently of the encoded trails; check every leaf, including
// rare deep leaves that random light sampling is unlikely to visit.
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

void VerifyBuiltTree(Runner& runner, lt::LightTreeBuilder& builder, const std::vector<LightTriangle>& tris,
    bool refit, bool requireDeep) {
    builder.UploadAll(runner.device.Get(), runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(), runner.Handle(9));
    builder.WriteLookupSrvs(runner.device.Get(), runner.Handle(16));
    auto gpu = builder.GetGpu();
    if (refit) {
        lt::TLASRebuilder rebuilder;
        auto result = rebuilder.Build(lt::ComputeBLASLocalRoots(tris), {}, 4);
        gpu.TLASNodes = runner.Upload(result.nodes);
        gpu.BLASBitTrail = runner.Upload(result.blasBitTrails);
        runner.Srv(gpu.TLASNodes.Get(), 9, sizeof(lt::LightTLASNodeGpu));
        runner.Srv(gpu.BLASBitTrail.Get(), 18, 0, DXGI_FORMAT_R32G32_UINT);
    }
    auto emissive = runner.Upload(tris);
    runner.Srv(emissive.Get(), 6, sizeof(LightTriangle));
    const auto tlas = runner.Read<lt::LightTLASNodeGpu>(gpu.TLASNodes.Get());
    const auto blas = runner.Read<lt::LightBLASNodeGpu>(gpu.BLASNodes.Get());
    const auto ranges = runner.Read<lt::BlasRangeGpu>(gpu.BLASRanges.Get());
    const auto indices = runner.Read<uint32_t>(gpu.LeafTriIndex.Get());
    const auto triTrails = runner.Read<lt::LightTreeTrail>(gpu.TriBitTrail.Get());
    const auto blasTrails = runner.Read<lt::LightTreeTrail>(gpu.BLASBitTrail.Get());
    Require(triTrails.size() == tris.size() && blasTrails.size() == ranges.size(), "Trail buffer has wrong stride/count");
    std::vector<float> expected(tris.size());
    uint32_t maxBlasDepth = 0;
    const auto maxTlasDepth = Walk(tlas, 0, 0, 0, 1.0f, [&](const auto& t, uint64_t path, float pdf) {
        Require(path == blasTrails.at(t.blasIndex), "TLAS trail does not reach its BLAS");
        const auto& r = ranges.at(t.blasIndex);
        std::vector<lt::LightBLASNodeGpu> local(blas.begin() + r.nodeOffset, blas.begin() + r.nodeOffset + r.nodeCount);
        maxBlasDepth = std::max(maxBlasDepth, Walk(local, 0, 0, 0, pdf, [&](const auto& b, uint64_t bp, float p) {
            Require(b.triCount == 1, "One-triangle leaf invariant broken");
            const auto tri = indices.at(r.triIndexOffset + b.triFirst);
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

// A four-way comb exercises every child value and the top two bits of the trail
// at exactly depth 32, without needing 4^32 triangles. Identical zero-power
// bounds make the production importance fallback uniform at every branch.
void VerifyBoundaryTree(Runner& runner, uint32_t depth, bool deepTlas) {
    // Beyond capacity, use a malformed unary chain so both sampling and PDF
    // evaluation must reject the only path, rather than reporting a partial PDF.
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
            tlas[i].blasIndex = nodes[i].triFirst;
        }
        nodes.assign(indices.size(), {});
        ranges.resize(indices.size());
        blasTrails = trails; trails.assign(indices.size(), 0);
        for (UINT i = 0; i < indices.size(); ++i) {
            nodes[i].triCount = 1; ranges[i].nodeOffset = i; ranges[i].triIndexOffset = i; triToBlas[i] = i;
        }
    }
    for (auto& r : ranges) XMStoreFloat4x4(&r.worldToLocal, XMMatrixIdentity());
    auto tg = runner.Upload(tlas); runner.Srv(tg.Get(), 9, sizeof(tlas[0]));
    auto bg = runner.Upload(nodes); runner.Srv(bg.Get(), 10, sizeof(nodes[0]));
    auto rg = runner.Upload(ranges); runner.Srv(rg.Get(), 11, sizeof(ranges[0]));
    auto ig = runner.Upload(indices); runner.Srv(ig.Get(), 12, 0, DXGI_FORMAT_R32_UINT);
    auto mg = runner.Upload(triToBlas); runner.Srv(mg.Get(), 16, 0, DXGI_FORMAT_R32_UINT);
    auto bt = runner.Upload(blasTrails); runner.Srv(bt.Get(), 18, 0, DXGI_FORMAT_R32G32_UINT);
    auto tt = runner.Upload(trails); runner.Srv(tt.Get(), 17, 0, DXGI_FORMAT_R32G32_UINT);
    auto eg = runner.Upload(std::vector<LightTriangle>(indices.size())); runner.Srv(eg.Get(), 6, sizeof(LightTriangle));
    runner.VerifyPdf(expected, invalidTree);
    std::cout << (deepTlas ? "TLAS" : "BLAS") << " boundary depth " << depth << " passed\n";
}

void VerifySGGeometry(Runner& runner) {
    auto metrics=runner.LearningSamples(2,2,5,0,0);
    Require(std::abs(metrics[0].x-.5f)<1e-6 && metrics[0].y==0,"Back-facing cone importance is incorrect");
    Require(std::abs(metrics[0].z-4*metrics[0].w)<1e-4,"Near-field distance regularization is not scale consistent");
    const auto normal=lt::normalize3({2,3,4});
    Require(lt::length3(lt::sub3({metrics[1].x,metrics[1].y,metrics[1].z},normal))<1e-6f && std::abs(metrics[1].w-4*lt::LT_PI)<1e-5,"SG isotropic limit or receiver-normal transform incorrect");
    lt::Cone ca,cb;ca.axis={1,0,0};ca.theta_o=.7f;cb.axis={-1,0,0};cb.theta_o=.1f;
    auto cu=lt::coneUnion(ca,cb);
    Require(lt::safe_acosf(lt::dot3(ca.axis,cu.axis))+ca.theta_o<=cu.theta_o+1e-5f &&
        lt::safe_acosf(lt::dot3(cb.axis,cu.axis))+cb.theta_o<=cu.theta_o+1e-5f,"Antiparallel cone union lost support");
    LightTriangle tri{};tri.x={0,0,0};tri.y={0,2,0};tri.z={3,0,0};tri.weight=2;tri.instanceID=0;
    std::vector<LightTriangle> tris{tri};auto roots=lt::ComputeBLASLocalRoots(tris);
    Require(roots.size()==1 && roots[0].localCone.theta_o==0,"Refit cone lost the emitter orientation");
    auto sg=lt::triangleSG(tri);
    Require(std::abs(sg.mean[0]-1)<1e-12 && std::abs(sg.mean[1]-2.0/3.0)<1e-12,"Wrong triangle centroid");
    Require(std::abs(sg.covariance[0][0]-.5)<1e-12 && std::abs(sg.covariance[1][1]-2.0/9.0)<1e-12,"Wrong triangle spatial variance");
    std::vector<InstanceXformCPU> transforms(1);
    XMStoreFloat4x4(&transforms[0].objectToWorld,XMMatrixScaling(-2,3,4)*XMMatrixRotationY(.4f)*XMMatrixTranslation(100,200,-300));
    lt::LightTreeBuilder builder;builder.Build(tris,transforms);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    auto initial=runner.Read<lt::LightTLASNodeGpu>(builder.GetGpu().TLASNodes.Get());
    lt::TLASRebuilder rebuilder;auto refit=rebuilder.Build(roots,transforms);
    const auto& a=initial[0];const auto& b=refit.nodes[0];
    Require(std::memcmp(&a.bmin,&b.bmin,sizeof(XMFLOAT3))==0 && std::memcmp(&a.bmax,&b.bmax,sizeof(XMFLOAT3))==0,"Initial/refit world bounds disagree");
    Require(std::abs(a.sgMean.x-b.sgMean.x)<1e-5 && std::abs(a.sgVariance-b.sgVariance)<1e-5,"Initial/refit SG moments disagree");
    auto expected=lt::transformPointW({1,2.0f/3.0f,0},transforms[0].objectToWorld);
    Require(lt::length3(lt::sub3(expected,a.sgMean))<1e-4f && a.cosTheta_o==-1 && a.sgSharpness==0,"Affine transform fallback or SG center incorrect");
    lt::LightTreeRefitManager manager;
    manager.RequestRefit(roots,transforms);manager.DiscardPending();
    lt::TLASRefitResult discarded;Require(!manager.IsPending() && !manager.PollResult(discarded),"Obsolete refit survived an emission rebuild");
    std::cout<<"SG moments and initial/refit affine transforms passed\n";
}
void VerifyLearning(Runner& runner, bool sgEnabled, bool twoLevel) {
    constexpr UINT count=128;
    std::vector<LightTriangle> tris(count);
    for(UINT i=0;i<count;++i) {
        auto& t=tris[i];float x=float(i%16)*.025f,y=float(i/16)*.025f;
        t.x={x,y,0};t.y={x,y+.01f,0};t.z={x+.01f,y,0};t.weight=1;
        t.instanceID=twoLevel?i/16:0;
    }
    lt::LightTreeBuilder builder;builder.Build(tris);builder.UploadAll(runner.device.Get(),runner.commands.Get());
    builder.WriteSrvs(runner.device.Get(),runner.Handle(9));builder.WriteLookupSrvs(runner.device.Get(),runner.Handle(16));
    auto emission=runner.Upload(tris);runner.Srv(emission.Get(),6,sizeof(LightTriangle));runner.Flush();
    auto roots=lt::ComputeBLASLocalRoots(tris);
    auto mapping=runner.Read<UINT>(builder.GetGpu().TriToBLAS.Get());
    for(UINT i=0;i<count;++i) Require(roots.at(mapping[i]).instanceID==tris[i].instanceID,"Initial and refit BLAS identities disagree");
    UINT flags=LT_FLAG_LEARNING|(sgEnabled?LT_FLAG_SG:0u);
    runner.PrepareLearning(flags,1,true);
    auto before=runner.LearningSamples(count,32768,1,123,flags); // cold prior + requests
    double baselineSecond=0;
    for(const auto& v:before) baselineSecond+=double(v.w)*v.w/before.size();
    runner.PrepareLearning(flags,2);
    auto initial=runner.LearningSamples(count,count,2,0,flags);
    Require(initial[0].z>=LT_CUT_INITIAL,"Initial light cut was not allocated");
    runner.LearningSamples(count,2048,4,234,flags); // every shadow blocked
    auto frozen=runner.LearningSamples(count,count,2,0,flags);
    for(UINT i=0;i<count;++i) Require(initial[i].x==frozen[i].x,"Training changed the current frame's PDF");
    runner.PrepareLearning(flags,3);
    auto black=runner.LearningSamples(count,count,2,0,flags);
    Require(std::abs(black[0].w-.75f*initial[0].w)<initial[0].w*1e-5f,"Occluded samples were omitted from the Q update");
    double mean=0;uint64_t samples=0;
    for(UINT frame=4;frame<100;++frame) {
        auto values=runner.LearningSamples(count,2048,1,frame*7919,flags);
        for(const auto& v:values) {
            Require(v.y>0 && std::isfinite(v.z) && std::abs(v.y-v.z)<v.y*3e-5f,"Learned sampling and MIS PDF disagree");
            mean+=v.w;++samples;
        }
        runner.PrepareLearning(flags,frame);
    }
    mean/=double(samples);Require(std::abs(mean-1.0)<.05,"Learned estimator changed the known unit integral");
    auto pdfs=runner.LearningSamples(count,count,2,0,flags);double total=0;
    for(const auto& v:pdfs) { total+=v.x;Require(v.x>0 && v.y==1 && v.z<=LT_CUT_MAX,"Learned cut lost support, overlaps, or exceeded capacity"); }
    Require(std::abs(total-1)<2e-5,"Learned leaf probabilities do not sum to one");
    Require(pdfs[0].z>initial[0].z,"Variance-based cut refinement never occurred");
    auto after=runner.LearningSamples(count,32768,3,45343,flags);double second=0,finalMean=0;
    for(const auto& v:after) { second+=double(v.w)*v.w/after.size();finalMean+=double(v.w)/after.size(); }
    Require(std::abs(finalMean-1)<.03 && second<baselineSecond*.3,"Learning did not reduce variance at equal sample count");
    std::cout<<"Learning "<<(sgEnabled?"SG":"cone")<<(twoLevel?" TLAS+BLAS":" BLAS")<<": mean "<<mean<<", cut "<<initial[0].z<<" -> "<<pdfs[0].z<<", second moment "<<baselineSecond<<" -> "<<second<<"\n";
    runner.PrepareLearning(flags,101,true);
    auto reset=runner.LearningSamples(count,count,2,0,flags);Require(reset[0].z==0,"Reset retained learned node references");
}

int main(int argc, char** argv) {
    try {
        Require(argc == 2, "Expected shader directory");
        // Budget boundaries include the largest representable item count.
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
        // Empty scenes must not dereference even a light-tree root. Bind null
        // descriptors and query the production sampler/PDF sentinel path.
        D3D12_SHADER_RESOURCE_VIEW_DESC nullSrv{};
        nullSrv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        nullSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        nullSrv.Buffer.NumElements = 1;
        nullSrv.Buffer.StructureByteStride = 16;
        for (UINT i = 0; i < 19; ++i) runner.device->CreateShaderResourceView(nullptr, &nullSrv, runner.Handle(i));
        runner.VerifyPdf({0.0f}, false, true);
        std::cout << "Empty mesh-light sampling with null resources passed\n";
        for (uint32_t d : {0u, 16u, 17u, 31u, 32u, 33u})
            for (bool tlas : {false, true}) VerifyBoundaryTree(runner, d, tlas);

        lt::LightTreeBuilder::Settings settings;
        settings.buildBins = 4;
        for (bool instances : {false, true}) {
            std::vector<LightTriangle> tris(180);
            for (uint32_t i = 0; i < tris.size(); ++i) {
                const float x = std::ldexp(1.0f, int(i) - 60);
                tris[i].x = {x, 0, 0}; tris[i].y = {x, 1, 0}; tris[i].z = {x, 0, 1};
                tris[i].instanceID = instances ? i : 0;
            }
            lt::LightTreeBuilder builder;
            builder.Build(tris, settings);
            VerifyBuiltTree(runner, builder, tris, false, true);
            if (instances) VerifyBuiltTree(runner, builder, tris, true, true);
            for (auto& t : tris) { t.x = {0,0,0}; t.y = {0,1,0}; t.z = {0,0,1}; }
            builder.Build(tris, settings);
            VerifyBuiltTree(runner, builder, tris, instances, false);
        }
        VerifySGGeometry(runner);
        VerifyLearning(runner,false,false);
        VerifyLearning(runner,true,false);
        VerifyLearning(runner,true,true);
        std::cout << "All light-tree tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
