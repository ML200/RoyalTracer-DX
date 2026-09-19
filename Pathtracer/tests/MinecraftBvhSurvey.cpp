#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "../src/Util/stb_image.h"
#include <windows.h>
#include <dxgi1_6.h>
#include <d3dx12.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include "minecraft/mc_world.h"
#include "minecraft/mc_materials.h"
#include "minecraft/mc_zip.h"
#include "minecraft/voxel_mesher.h"
#include "planet/worker_pool.h"
#include "planet/tlas_builder.h"
#include "Scene/BvhQuality.h"

using namespace mc;
using namespace planet;
using Clock = std::chrono::steady_clock;
static void Check(HRESULT hr) { if (FAILED(hr)) throw std::runtime_error("D3D12 operation failed: " + std::to_string((uint32_t)hr)); }
static uint64_t Align(uint64_t n) { return (n+255)&~255ull; }

struct Geometry {
    NodeKey key;
    uint32_t vertexOffset, vertexCount, indexOffset, tris, opaque;
    uint64_t blasOffset = 0, blasBytes = 0;
};

struct Gpu {
    ComPtr<ID3D12Device5> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList4> commands;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12QueryHeap> queries;
    ComPtr<ID3D12Resource> output, readback, vertices, indices, pool, scratch;
    TlasBuilder tlas;
    uint64_t serial = 0, frequency = 0;
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    static constexpr uint32_t Rays = 262144;
    explicit Gpu(const char* shader) {
        Check(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));
        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter1> adapter;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        Check(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter)));
        DXGI_ADAPTER_DESC1 adapterDesc{};
        Check(adapter->GetDesc1(&adapterDesc));
        printf("Survey GPU: %ls\n", adapterDesc.Description);
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
        Check(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options)));
        if (options.RaytracingTier < D3D12_RAYTRACING_TIER_1_1) throw std::runtime_error("DXR 1.1 required");
        D3D12_COMMAND_QUEUE_DESC q{};
        Check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
        Check(queue->GetTimestampFrequency(&frequency));
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));
        Check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        if (!event) throw std::runtime_error("Event creation failed");
        CD3DX12_ROOT_PARAMETER params[3];
        params[0].InitAsShaderResourceView(0); params[1].InitAsUnorderedAccessView(0); params[2].InitAsConstants(8,0);
        CD3DX12_ROOT_SIGNATURE_DESC rs(3,params);
        ComPtr<ID3DBlob> blob, errors;
        Check(D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&errors));
        Check(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)));
        std::ifstream file(shader,std::ios::binary);
        if (!file) throw std::runtime_error("Missing survey shader");
        std::vector<char> code((std::istreambuf_iterator<char>(file)),{});
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{}; pd.pRootSignature=root.Get(); pd.CS={code.data(),code.size()};
        Check(device->CreateComputePipelineState(&pd,IID_PPV_ARGS(&pso)));
        D3D12_QUERY_HEAP_DESC hd{}; hd.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP; hd.Count=2;
        Check(device->CreateQueryHeap(&hd,IID_PPV_ARGS(&queries)));
        output=create_buffer(device.Get(),Rays*8,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,HEAP_DEFAULT);
        readback=create_buffer(device.Get(),Rays*8+256,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST,HEAP_READBACK);
    }
    ~Gpu() { if(event) CloseHandle(event); }
    void flush() {
        Check(commands->Close()); ID3D12CommandList* lists[]{commands.Get()}; queue->ExecuteCommandLists(1,lists);
        Check(queue->Signal(fence.Get(),++serial)); Check(fence->SetEventOnCompletion(serial,event));
        if(WaitForSingleObject(event,60000)!=WAIT_OBJECT_0) throw std::runtime_error("GPU timeout");
        Check(allocator->Reset()); Check(commands->Reset(allocator.Get(),nullptr));
    }
    template<class T> ComPtr<ID3D12Resource> upload(const std::vector<T>& values) {
        auto r=create_buffer(device.Get(),values.size()*sizeof(T),D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ,HEAP_UPLOAD);
        void* p=nullptr; Check(r->Map(0,nullptr,&p)); memcpy(p,values.data(),values.size()*sizeof(T)); r->Unmap(0,nullptr); return r;
    }
    uint32_t descriptions(const Geometry& g, D3D12_RAYTRACING_GEOMETRY_DESC gd[2]) {
        uint32_t n=0;
        auto add=[&](uint32_t tris,bool opaque,uint32_t offset) {
            if(!tris)return;
            auto& d=gd[n++]; d.Type=D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            d.Flags=opaque?D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE:D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
            d.Triangles.VertexBuffer={vertices->GetGPUVirtualAddress()+uint64_t(g.vertexOffset)*sizeof(MeshVertex),sizeof(MeshVertex)};
            d.Triangles.VertexCount=g.vertexCount; d.Triangles.VertexFormat=DXGI_FORMAT_R32G32B32_FLOAT;
            d.Triangles.IndexBuffer=indices->GetGPUVirtualAddress()+uint64_t(g.indexOffset+offset)*4;
            d.Triangles.IndexCount=tris*3; d.Triangles.IndexFormat=DXGI_FORMAT_R32_UINT;
        };
        add(g.opaque,true,0); add(g.tris-g.opaque,false,g.opaque*3); return n;
    }
    uint64_t build(std::vector<Geometry>& geometry,const std::vector<MeshVertex>& v,const std::vector<uint32_t>& idx,const double camera[3]) {
        vertices=upload(v); indices=upload(idx);
        uint64_t total=0,maxScratch=0;
        for(auto& g:geometry) {
            D3D12_RAYTRACING_GEOMETRY_DESC gd[2]{};
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
            in.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL; in.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;
            in.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            in.NumDescs=descriptions(g,gd); in.pGeometryDescs=gd;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{}; device->GetRaytracingAccelerationStructurePrebuildInfo(&in,&info);
            g.blasOffset=total; g.blasBytes=Align(info.ResultDataMaxSizeInBytes); total+=g.blasBytes;
            maxScratch=std::max(maxScratch,info.ScratchDataSizeInBytes);
        }
        if(total>(uint64_t(6)<<30))throw std::runtime_error("Survey BLAS exceeds 6 GB; lower detail factor");
        pool=create_buffer(device.Get(),total,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,HEAP_DEFAULT);
        scratch=create_buffer(device.Get(),maxScratch,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,HEAP_DEFAULT);
        uint32_t built=0;
        for(const auto& g:geometry) {
            D3D12_RAYTRACING_GEOMETRY_DESC gd[2]{};
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC d{};
            d.Inputs.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL; d.Inputs.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;
            d.Inputs.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            d.Inputs.NumDescs=descriptions(g,gd); d.Inputs.pGeometryDescs=gd;
            d.DestAccelerationStructureData=pool->GetGPUVirtualAddress()+g.blasOffset; d.ScratchAccelerationStructureData=scratch->GetGPUVirtualAddress();
            commands->BuildRaytracingAccelerationStructure(&d,0,nullptr);
            auto barrier=CD3DX12_RESOURCE_BARRIER::UAV(scratch.Get()); commands->ResourceBarrier(1,&barrier);
            if(++built%128==0)flush();
        }
        auto barrier=CD3DX12_RESOURCE_BARRIER::UAV(pool.Get()); commands->ResourceBarrier(1,&barrier); flush();
        tlas.init(device.Get(),(uint32_t)geometry.size()); tlas.begin();
        for(uint32_t i=0;i<geometry.size();++i) {
            int64_t o[3]; node_origin_blocks(geometry[i].key,o);
            const float transform[12]{1,0,0,(float)(o[0]-camera[0]),0,1,0,(float)(o[1]-camera[1]),0,0,1,(float)(o[2]-camera[2])};
            tlas.add_instance(pool->GetGPUVirtualAddress()+geometry[i].blasOffset,transform,i,0,D3D12_RAYTRACING_INSTANCE_FLAG_NONE);
        }
        tlas.build(commands.Get()); flush(); vertices.Reset(); indices.Reset(); scratch.Reset(); return total;
    }
    double probe(uint32_t mode,uint32_t seed,uint32_t& hits) {
        struct Constants { float origin[3]; uint32_t mode,count; float tMax; uint32_t seed,pad; } c{{0,0,0},mode,Rays,65536.0f,seed,0};
        commands->SetPipelineState(pso.Get()); commands->SetComputeRootSignature(root.Get());
        commands->SetComputeRootShaderResourceView(0,tlas.tlas_address()); commands->SetComputeRootUnorderedAccessView(1,output->GetGPUVirtualAddress());
        commands->SetComputeRoot32BitConstants(2,8,&c,0);
        constexpr uint32_t rounds=16;
        commands->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        for(uint32_t i=0;i<rounds;++i) { commands->Dispatch(Rays/64,1,1); auto b=CD3DX12_RESOURCE_BARRIER::UAV(output.Get()); commands->ResourceBarrier(1,&b); }
        commands->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
        commands->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,readback.Get(),0);
        auto b=CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE); commands->ResourceBarrier(1,&b);
        commands->CopyBufferRegion(readback.Get(),256,output.Get(),0,Rays*8);
        b=CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS); commands->ResourceBarrier(1,&b); flush();
        void* p=nullptr; Check(readback->Map(0,nullptr,&p)); const auto ticks=(uint64_t*)p;
        const double ms=(ticks[1]-ticks[0])*1000.0/frequency/rounds;
        hits=0; const auto words=(uint32_t*)((char*)p+256); for(uint32_t i=0;i<Rays;++i)if(words[2*i])++hits;
        readback->Unmap(0,nullptr); return ms;
    }
};

int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if(argc<5 || (argc>6 && argc!=9)) { printf("Usage: MinecraftBvhSurvey world pack shader output.json [detail=128] [cameraX cameraY cameraZ]\n"); return 2; }
    try {
        const float detail=argc>5?std::stof(argv[5]):128.f;
        WorkerPool workers;
        World world; WorldLoadConfig config; config.worldDir=argv[1]; std::string error;
        if(!world.load(config,&workers,&error))throw std::runtime_error(error);
        ResourceStack resources; std::vector<std::unique_ptr<ZipArchive>> archives;
        for(const std::string& path:{std::string(argv[2]),find_minecraft_jar()}) {
            auto z=std::make_unique<ZipArchive>(); if(!z->open(path,&error))throw std::runtime_error("Required assets: "+path+": "+error);
            resources.push(z.get()); archives.push_back(std::move(z));
        }
        MaterialSoA materials; std::vector<std::string> names; std::vector<TextureData> textures; MaterialBuildStats stats;
        if(!MaterialBuilder().build(world.registry(),resources,materials,names,textures,0,stats,&error))throw std::runtime_error(error);
        world.build_lod(&workers);
        const double camera[3]{argc==9?std::stod(argv[6]):(double)world.level().spawnX,
            argc==9?std::stod(argv[7]):(double)world.level().spawnY+40,
            argc==9?std::stod(argv[8]):(double)world.level().spawnZ};
        LodCut cut; const auto selected=Clock::now(); world.lod_tree().select(camera,detail,cut,&workers);
        const double selectMs=std::chrono::duration<double,std::milli>(Clock::now()-selected).count();
        printf("Exact cut: %zu nodes, detail %.1f, camera %.0f %.0f %.0f\n",cut.leafList.size(),detail,camera[0],camera[1],camera[2]);
        std::vector<Geometry> geometry; std::vector<MeshVertex> vertices; std::vector<uint32_t> indices;
        std::vector<bvh::Bounds> bounds; std::vector<uint32_t> counts;
        uint64_t alpha=0,lights=0,thin=0,degenerate=0; const auto start=Clock::now();
        ChunkMesher mesher(world.registry(),world.store()); ChunkMesh mesh;
        std::ofstream chunks(std::filesystem::path(argv[4]).replace_extension(".csv")); chunks<<"level,x,y,z,triangles,alpha,lights\n";
        for(uint32_t i=0;i<cut.leafList.size();++i) {
            const NodeKey key=unpack_node(cut.leafList[i]); MeshParams params; params.flatMaterials=key.level>=6;
            mesher.mesh(key,params,mesh); if(mesh.empty())continue;
            if(vertices.size()+mesh.vertices.size()>UINT32_MAX || indices.size()+mesh.indices.size()>UINT32_MAX)throw std::runtime_error("Survey exceeds index range");
            geometry.push_back({key,(uint32_t)vertices.size(),(uint32_t)mesh.vertices.size(),(uint32_t)indices.size(),mesh.triangle_count(),mesh.opaqueTriCount});
            int64_t o[3]; node_origin_blocks(key,o); bvh::Bounds bb;
            for(const auto& v:mesh.vertices)bb.point(v.px+(float)o[0],v.py+(float)o[1],v.pz+(float)o[2]);
            bounds.push_back(bb); counts.push_back(mesh.triangle_count());
            for(uint32_t t=0;t<mesh.triangle_count();++t) {
                const auto& a=mesh.vertices[mesh.indices[3*t]], &b=mesh.vertices[mesh.indices[3*t+1]], &c=mesh.vertices[mesh.indices[3*t+2]];
                const Vec3f ab{b.px-a.px,b.py-a.py,b.pz-a.pz},ac{c.px-a.px,c.py-a.py,c.pz-a.pz},bc{c.px-b.px,c.py-b.py,c.pz-b.pz};
                const Vec3f cr=cross(ab,ac); const float twiceArea=std::sqrt(dot(cr,cr));
                if(twiceArea<=1e-10f)++degenerate;
                else if(std::max({dot(ab,ab),dot(ac,ac),dot(bc,bc)})/twiceArea>=16.f)++thin;
            }
            alpha+=mesh.alphaTriCount; lights+=mesh.light_tri_count();
            chunks<<(int)key.level<<','<<key.x<<','<<key.y<<','<<key.z<<','<<mesh.triangle_count()<<','<<mesh.alphaTriCount<<','<<mesh.light_tri_count()<<'\n';
            vertices.insert(vertices.end(),mesh.vertices.begin(),mesh.vertices.end()); indices.insert(indices.end(),mesh.indices.begin(),mesh.indices.end());
            if(i%1000==0)printf("Meshed %u/%zu, %.1fM triangles\n",i,cut.leafList.size(),indices.size()/3e6);
        }
        const double meshSeconds=std::chrono::duration<double>(Clock::now()-start).count();
        const auto survey=bvh::survey(bounds,counts);
        printf("Exact geometry: %u BLAS, %llu triangles, %llu alpha, %llu lights, %llu root overlaps, %llu thin triangles\n",survey.instances,survey.triangles,alpha,lights,survey.overlapPairs,thin);
        Gpu gpu(argv[3]); const uint64_t blasBytes=gpu.build(geometry,vertices,indices,camera);
        vertices.clear(); vertices.shrink_to_fit(); indices.clear(); indices.shrink_to_fit();
        double medians[2]{}; uint32_t hits=0;
        for(uint32_t mode=0;mode<2;++mode) {
            gpu.probe(mode,0,hits); std::vector<double> samples;
            for(uint32_t i=0;i<9;++i)samples.push_back(gpu.probe(mode,0,hits));
            std::sort(samples.begin(),samples.end()); medians[mode]=samples[4];
            printf("%s: %.4f ms / %u rays, %u hits (force opaque, shading excluded)\n",mode?"Occlusion":"Closest",medians[mode],Gpu::Rays,hits);
        }
        std::ofstream out(argv[4]);
        out<<"{\n\"camera\":["<<camera[0]<<','<<camera[1]<<','<<camera[2]<<"],\n\"detail\":"<<detail<<",\n\"chunks\":"<<world.stats().chunks<<",\n\"selected\":"<<cut.leafList.size()
            <<",\n\"blas\":"<<survey.instances<<",\n\"triangles\":"<<survey.triangles<<",\n\"alpha\":"<<alpha<<",\n\"light_triangles\":"<<lights
            <<",\n\"overlap_pairs\":"<<survey.overlapPairs<<",\n\"thin_triangles\":"<<thin<<",\n\"degenerate_triangles\":"<<degenerate
            <<",\n\"blas_bytes_uncompacted\":"<<blasBytes<<",\n\"selection_ms\":"<<selectMs<<",\n\"mesh_seconds\":"<<meshSeconds
            <<",\n\"rays\":"<<Gpu::Rays<<",\n\"hit_rays\":"<<hits<<",\n\"closest_ms\":"<<medians[0]<<",\n\"occlusion_ms\":"<<medians[1]<<"\n}\n";
    } catch(const std::exception& e) { fprintf(stderr,"Survey failed: %s\n",e.what()); return 1; }
    return 0;
}
