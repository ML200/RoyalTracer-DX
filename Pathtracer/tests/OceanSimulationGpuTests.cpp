#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dx12.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <random>
#include <stdexcept>
#define OCEAN_FFT_SIZE 16
#define OCEAN_FFT_LOG2 4
#include "../shaders/OceanLayout.h"
using Microsoft::WRL::ComPtr;
using DirectX::XMFLOAT4;
constexpr int N=OCEAN_FFT_SIZE,C=OCEAN_CASCADES;
constexpr double pi=3.14159265358979323846;
void Check(HRESULT hr){if(FAILED(hr))throw std::runtime_error("D3D12 failure " + std::to_string(unsigned(hr)));}
void Require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
struct Runner {
    ComPtr<ID3D12Device> dev;ComPtr<ID3D12CommandQueue> queue;ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cmd;ComPtr<ID3D12Fence> fence;ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12DescriptorHeap> heap;UINT stride=0;UINT64 serial=0;HANDLE event=CreateEvent(nullptr,FALSE,FALSE,nullptr);
    std::vector<ComPtr<ID3D12Resource>> keep;
    std::array<ComPtr<ID3D12PipelineState>,7> psos;
    struct Push {UINT u0=0,u1=0,u2=0,u3=0;float time=0,dt=0,f2=0,f3=.55f;} push;
    ComPtr<ID3D12Resource> Buffer(UINT64 size,D3D12_HEAP_TYPE type){
        CD3DX12_HEAP_PROPERTIES hp(type);auto d=CD3DX12_RESOURCE_DESC::Buffer(size);ComPtr<ID3D12Resource> r;
        Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,type==D3D12_HEAP_TYPE_UPLOAD?D3D12_RESOURCE_STATE_GENERIC_READ:D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&r)));keep.push_back(r);return r;
    }
    D3D12_CPU_DESCRIPTOR_HANDLE At(UINT slot){auto h=heap->GetCPUDescriptorHandleForHeapStart();h.ptr+=slot*stride;return h;}
    ComPtr<ID3D12Resource> Texture(int slices,int levels,DXGI_FORMAT format,UINT srv,UINT uav,bool write=true){
        auto d=CD3DX12_RESOURCE_DESC::Tex2D(format,N,N,UINT16(slices),UINT16(levels));if(write)d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        CD3DX12_HEAP_PROPERTIES hp(D3D12_HEAP_TYPE_DEFAULT);ComPtr<ID3D12Resource> r;
        Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,write?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&r)));
        if(srv){D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.Format=format;s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2DARRAY;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;s.Texture2DArray.ArraySize=slices;s.Texture2DArray.MipLevels=levels;dev->CreateShaderResourceView(r.Get(),&s,At(srv));}
        if(uav)for(int i=0;i<levels;++i){D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2DARRAY;u.Texture2DArray.ArraySize=slices;u.Texture2DArray.MipSlice=i;dev->CreateUnorderedAccessView(r.Get(),nullptr,&u,At(uav+i));}
        keep.push_back(r);return r;
    }
    void Upload(ID3D12Resource* r,const std::vector<XMFLOAT4>& data){
        auto d=r->GetDesc();std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layout(d.DepthOrArraySize);UINT64 bytes;
        dev->GetCopyableFootprints(&d,0,d.DepthOrArraySize,0,layout.data(),nullptr,nullptr,&bytes);auto staging=Buffer(bytes,D3D12_HEAP_TYPE_UPLOAD);
        char* p;Check(staging->Map(0,nullptr,(void**)&p));
        for(UINT i=0;i<d.DepthOrArraySize;++i){for(int y=0;y<N;++y)memcpy(p+layout[i].Offset+y*layout[i].Footprint.RowPitch,data.data()+(i*N+y)*N,N*sizeof(XMFLOAT4));
            CD3DX12_TEXTURE_COPY_LOCATION src(staging.Get(),layout[i]),dst(r,i);cmd->CopyTextureRegion(&dst,0,0,0,&src,nullptr);}
        staging->Unmap(0,nullptr);D3D12_RESOURCE_BARRIER b=CD3DX12_RESOURCE_BARRIER::Transition(r,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);cmd->ResourceBarrier(1,&b);
    }
    void Barrier(ID3D12Resource*r){auto b=CD3DX12_RESOURCE_BARRIER::UAV(r);cmd->ResourceBarrier(1,&b);}
    void Flush(){Check(cmd->Close());ID3D12CommandList* list[]={cmd.Get()};queue->ExecuteCommandLists(1,list);Check(queue->Signal(fence.Get(),++serial));Check(fence->SetEventOnCompletion(serial,event));Require(WaitForSingleObject(event,30000)==WAIT_OBJECT_0,"GPU timeout");Check(alloc->Reset());Check(cmd->Reset(alloc.Get(),nullptr));}
    std::vector<XMFLOAT4> Read(ID3D12Resource*r,int mip=0){
        auto d=r->GetDesc();UINT sub=mip;D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;UINT64 bytes;dev->GetCopyableFootprints(&d,sub,1,0,&layout,nullptr,nullptr,&bytes);
        auto read=Buffer(bytes,D3D12_HEAP_TYPE_READBACK);D3D12_RESOURCE_BARRIER b=CD3DX12_RESOURCE_BARRIER::Transition(r,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);cmd->ResourceBarrier(1,&b);
        CD3DX12_TEXTURE_COPY_LOCATION dst(read.Get(),layout),src(r,sub);cmd->CopyTextureRegion(&dst,0,0,0,&src,nullptr);std::swap(b.Transition.StateBefore,b.Transition.StateAfter);cmd->ResourceBarrier(1,&b);Flush();
        const int n=N>>mip;std::vector<XMFLOAT4> result(n*n);char* p;Check(read->Map(0,nullptr,(void**)&p));for(int y=0;y<n;++y)memcpy(result.data()+y*n,p+layout.Offset+y*layout.Footprint.RowPitch,n*sizeof(XMFLOAT4));read->Unmap(0,nullptr);return result;
    }
    void Dispatch(int pso,UINT x,UINT y,UINT z){ID3D12DescriptorHeap* h[]={heap.Get()};cmd->SetDescriptorHeaps(1,h);cmd->SetComputeRootSignature(root.Get());cmd->SetPipelineState(psos[pso].Get());cmd->SetComputeRoot32BitConstants(0,8,&push,0);cmd->Dispatch(x,y,z);}
    Runner(const std::string& path){
        Check(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&dev)));D3D12_COMMAND_QUEUE_DESC q{};Check(dev->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)));Check(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc.Get(),nullptr,IID_PPV_ARGS(&cmd)));Check(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=256;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;Check(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));stride=dev->GetDescriptorHandleIncrementSize(hd.Type);
        CD3DX12_ROOT_PARAMETER rp;rp.InitAsConstants(8,0);CD3DX12_STATIC_SAMPLER_DESC sampler(0,D3D12_FILTER_MIN_MAG_MIP_LINEAR,D3D12_TEXTURE_ADDRESS_MODE_WRAP,D3D12_TEXTURE_ADDRESS_MODE_WRAP,D3D12_TEXTURE_ADDRESS_MODE_WRAP);
        CD3DX12_ROOT_SIGNATURE_DESC rd(1,&rp,1,&sampler,D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);ComPtr<ID3DBlob> blob,err;Check(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&err));Check(dev->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)));
        const char* names[]={"OceanEvolve","OceanFftH","OceanFftV","OceanAssemble","OceanFoam","OceanMip","OceanCondition"};
        for(int i=0;i<7;++i){std::ifstream f(path+"/small-"+names[i]+".dxil",std::ios::binary);Require(bool(f),"Missing test shader");std::vector<char> code((std::istreambuf_iterator<char>(f)),{});D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};pd.pRootSignature=root.Get();pd.CS={code.data(),code.size()};Check(dev->CreateComputePipelineState(&pd,IID_PPV_ARGS(&psos[i])));}
    }
    ~Runner(){CloseHandle(event);}
    void Test(){
        auto h0=Texture(C,1,DXGI_FORMAT_R32G32B32A32_FLOAT,OCEAN_SRV_H0,0,false),wave=Texture(C,1,DXGI_FORMAT_R32G32B32A32_FLOAT,OCEAN_SRV_WAVE,0,false);
        auto fft=Texture(C*2,1,DXGI_FORMAT_R32G32B32A32_FLOAT,0,OCEAN_UAV_FFT);
        auto disp=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R32G32B32A32_FLOAT,OCEAN_SRV_DISP,OCEAN_UAV_DISP_MIPS);
        auto deriv=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R32G32B32A32_FLOAT,OCEAN_SRV_DERIV,OCEAN_UAV_DERIV_MIPS);
        auto moments=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R32G32B32A32_FLOAT,0,OCEAN_UAV_MOMENT_MIPS);
        auto surface=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R32G32B32A32_FLOAT,0,OCEAN_UAV_SURFACE_MIPS);
        auto fa=Texture(C,1,DXGI_FORMAT_R32G32_FLOAT,0,OCEAN_UAV_FOAM_A),fb=Texture(C,1,DXGI_FORMAT_R32G32_FLOAT,0,OCEAN_UAV_FOAM_B);
        OceanParamsGPU P{};P.choppiness=8.0f;P.displacementScale=1;P.foamCoverageScale=0;P.cascadeLength={64,64,64,64};
        auto params=Buffer(sizeof(P),D3D12_HEAP_TYPE_UPLOAD);void* ptr;Check(params->Map(0,nullptr,&ptr));memcpy(ptr,&P,sizeof(P));params->Unmap(0,nullptr);
        D3D12_SHADER_RESOURCE_VIEW_DESC ps{};ps.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;ps.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;ps.Buffer.NumElements=1;ps.Buffer.StructureByteStride=sizeof(P);dev->CreateShaderResourceView(params.Get(),&ps,At(OCEAN_SRV_PARAMS));
        std::vector<XMFLOAT4> initial(C*N*N),waves(C*N*N);std::vector<std::complex<double>> a(N*N);std::mt19937 gen(11);std::normal_distribution<double> random;
        for(int z=1;z<N;++z)for(int x=1;x<N;++x){if(x==N/2&&z==N/2)continue;a[z*N+x]=.015*std::complex<double>(random(gen),random(gen));}
        for(int z=0;z<N;++z)for(int x=0;x<N;++x){int i=z*N+x;auto b=std::conj(a[((N-z)%N)*N+(N-x)%N]);initial[i]={float(a[i].real()),float(a[i].imag()),float(b.real()),float(b.imag())};double kx=(x-N/2)*2*pi/64,kz=(z-N/2)*2*pi/64,k=std::hypot(kx,kz);waves[i]={float(kx),float(kz),float(k>0?1/k:0),float(std::sqrt(9.80665*k))};}
        Upload(h0.Get(),initial);Upload(wave.Get(),waves);push.time=2.3f;
        Dispatch(0,N/8,N/8,C);Barrier(fft.Get());Dispatch(1,N,C*2,1);Barrier(fft.Get());Dispatch(2,N,C*2,1);Barrier(fft.Get());Dispatch(3,N/8,N/8,C);Barrier(disp.Get());Barrier(deriv.Get());Barrier(moments.Get());
        auto D=Read(disp.Get()),G=Read(deriv.Get()),M=Read(moments.Get());double err2=0,ref2=0,meanSquare=0,spectral=0,imag2=0;
        std::vector<std::complex<double>> H(N*N);
        for(int i=0;i<N*N;++i){auto h=initial[i];auto phase=std::polar(1.,-double(waves[i].w)*push.time);H[i]=std::complex<double>(h.x,h.y)*phase+std::complex<double>(h.z,h.w)*std::conj(phase);spectral+=std::norm(H[i]);}
        for(int z=0;z<N;++z)for(int x=0;x<N;++x){std::array<std::complex<double>,8> ref{};
            for(int j=0;j<N*N;++j){const auto w=waves[j];auto h=H[j]*std::polar(1.,2*pi*((j%N-N/2)*x+(j/N-N/2)*z)/N);const std::complex<double> I(0,1);
                const double k=std::hypot(double(w.x),double(w.y));
                const double chop=P.choppiness/(1+std::pow(k/(4*pi),4));
                ref[0]+=I*(chop*w.x*w.z)*h;ref[1]+=h;ref[2]+=I*(chop*w.y*w.z)*h;ref[3]-=(chop*w.x*w.y*w.z)*h;
                ref[4]+=I*double(w.x)*h;ref[5]+=I*double(w.y)*h;ref[6]-=(chop*w.x*w.x*w.z)*h;ref[7]-=(chop*w.y*w.y*w.z)*h;}
            int i=z*N+x;const float values[]={D[i].x,D[i].y,D[i].z,D[i].w,G[i].x,G[i].y,G[i].z,G[i].w};
            for(int j=0;j<8;++j){err2+=std::pow(values[j]-ref[j].real(),2);ref2+=std::norm(ref[j]);imag2+=std::pow(ref[j].imag(),2);}
            meanSquare+=D[i].y*D[i].y;
            Require(std::abs(M[i].x-G[i].x*G[i].x)<1e-6 && std::abs(M[i].y-G[i].x*G[i].y)<1e-6 && std::abs(M[i].z-G[i].y*G[i].y)<1e-6,"Raw moment storage");
        }
        double relative=std::sqrt(err2/ref2),parseval=std::abs(meanSquare/(N*N)/spectral-1);
        std::cout<<"Production GPU FFT versus double DFT relative L2 "<<relative<<", Parseval "<<parseval<<", reference imaginary residual "<<std::sqrt(imag2/ref2)<<'\n';
        Require(relative<1e-5 && parseval<1e-5 && std::sqrt(imag2/ref2)<1e-5,"FFT/derivative/Parseval/Hermitian check");
        for(UINT mip=1;mip<OCEAN_MIP_LEVELS;++mip){push.u0=2;push.u1=mip;Dispatch(5,1,1,C);Barrier(moments.Get());}
        const auto bound=Read(moments.Get(),OCEAN_MIP_LEVELS-1);
        const float gain=std::min(1.0f,.85f/std::max(bound[0].w,1e-8f));
        Require(gain<1,"Stress fixture must exercise conditioning");
        Dispatch(6,N/8,N/8,C);Barrier(disp.Get());Barrier(deriv.Get());
        auto safeD=Read(disp.Get()),safeG=Read(deriv.Get());
        for(int i=0;i<N*N;++i){
            Require(std::abs(safeD[i].x-D[i].x*gain)<1e-5 && safeD[i].y==D[i].y,"Conditioning preserves height and scales horizontal shape");
            const float a=1+safeG[i].z,b=1+safeG[i].w,c=safeD[i].w;
            Require(.5f*(a+b-std::hypot(a-b,2*c))>=.14999f,"Composite conditioning margin");
        }
        std::cout<<"Conditioning stress gain "<<gain<<", minimum stretch >= 0.15\n";
        push.u0=0;push.u1=1;push.dt=1.f/60;Dispatch(4,N/8,N/8,C);Barrier(surface.Get());auto F=Read(surface.Get());for(auto v:F)Require(v.x==0&&v.z==0&&std::isfinite(v.w),"Disabled foam reset must be zero/finite");
        // First moment mip verifies production averaging (no threshold after filtering).
        push.u0=2;push.u1=1;Dispatch(5,1,1,C);Barrier(moments.Get());auto MM=Read(moments.Get(),1);
        for(int z=0;z<N/2;++z)for(int x=0;x<N/2;++x){XMFLOAT4 sum{};for(int j=0;j<2;++j)for(int i=0;i<2;++i){auto v=M[(z*2+j)*N+x*2+i];sum.x+=v.x*.25f;sum.y+=v.y*.25f;sum.z+=v.z*.25f;}auto v=MM[z*N/2+x];Require(std::abs(v.x-sum.x)+std::abs(v.y-sum.y)+std::abs(v.z-sum.z)<1e-6,"Moment mip average");}
        std::cout<<"PASS: actual production Evolve/FFT/Assemble/Foam/Mip kernels (16x16 test lattice)\n";
    }
};
int main(int argc,char**argv){try{Require(argc==2,"Shader directory argument required");Runner r(argv[1]);r.Test();return 0;}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}

