#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dx12.h>
#include <DirectXMath.h>
#include <DirectXPackedVector.h>
#include <wrl/client.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <cmath>
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
    std::array<ComPtr<ID3D12PipelineState>,3> psos;
    struct Push {UINT u0=0,u1=0,u2=0,u3=0;float time=0,dt=0,f2=0,f3=0;} push;
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
    // Every slice of one mip of a half-float array, widened to float.
    std::vector<XMFLOAT4> ReadHalf(ID3D12Resource*r,int mip=0){
        auto d=r->GetDesc();const int n=N>>mip;std::vector<XMFLOAT4> result;
        for(UINT slice=0;slice<d.DepthOrArraySize;++slice){
            const UINT sub=D3D12CalcSubresource(mip,slice,0,d.MipLevels,d.DepthOrArraySize);D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout;UINT64 bytes;dev->GetCopyableFootprints(&d,sub,1,0,&layout,nullptr,nullptr,&bytes);
            auto read=Buffer(bytes,D3D12_HEAP_TYPE_READBACK);D3D12_RESOURCE_BARRIER b=CD3DX12_RESOURCE_BARRIER::Transition(r,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);cmd->ResourceBarrier(1,&b);
            CD3DX12_TEXTURE_COPY_LOCATION dst(read.Get(),layout),src(r,sub);cmd->CopyTextureRegion(&dst,0,0,0,&src,nullptr);std::swap(b.Transition.StateBefore,b.Transition.StateAfter);cmd->ResourceBarrier(1,&b);Flush();
            char* p;Check(read->Map(0,nullptr,(void**)&p));
            for(int y=0;y<n;++y){auto* h=reinterpret_cast<const DirectX::PackedVector::HALF*>(p+layout.Offset+y*layout.Footprint.RowPitch);
                for(int x=0;x<n;++x)result.push_back({DirectX::PackedVector::XMConvertHalfToFloat(h[x*4]),DirectX::PackedVector::XMConvertHalfToFloat(h[x*4+1]),
                    DirectX::PackedVector::XMConvertHalfToFloat(h[x*4+2]),DirectX::PackedVector::XMConvertHalfToFloat(h[x*4+3])});}
            read->Unmap(0,nullptr);
        }
        return result;
    }
    void Dispatch(int pso,UINT x,UINT y,UINT z){ID3D12DescriptorHeap* h[]={heap.Get()};cmd->SetDescriptorHeaps(1,h);cmd->SetComputeRootSignature(root.Get());cmd->SetPipelineState(psos[pso].Get());cmd->SetComputeRoot32BitConstants(0,8,&push,0);cmd->Dispatch(x,y,z);}
    Runner(const std::string& path){
        Check(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&dev)));D3D12_COMMAND_QUEUE_DESC q{};Check(dev->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)));Check(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc.Get(),nullptr,IID_PPV_ARGS(&cmd)));Check(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=256;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;Check(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));stride=dev->GetDescriptorHandleIncrementSize(hd.Type);
        CD3DX12_ROOT_PARAMETER rp;rp.InitAsConstants(8,0);CD3DX12_STATIC_SAMPLER_DESC sampler(0,D3D12_FILTER_MIN_MAG_MIP_LINEAR,D3D12_TEXTURE_ADDRESS_MODE_WRAP,D3D12_TEXTURE_ADDRESS_MODE_WRAP,D3D12_TEXTURE_ADDRESS_MODE_WRAP);
        CD3DX12_ROOT_SIGNATURE_DESC rd(1,&rp,1,&sampler,D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED);ComPtr<ID3DBlob> blob,err;Check(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&err));Check(dev->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)));
        const char* names[]={"OceanFftH","OceanFftV","OceanMip"};
        for(int i=0;i<3;++i){std::ifstream f(path+"/small-"+names[i]+".dxil",std::ios::binary);Require(bool(f),"Missing test shader");std::vector<char> code((std::istreambuf_iterator<char>(f)),{});D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};pd.pRootSignature=root.Get();pd.CS={code.data(),code.size()};Check(dev->CreateComputePipelineState(&pd,IID_PPV_ARGS(&psos[i])));}
    }
    ~Runner(){CloseHandle(event);}
    void Test(){
        constexpr float L=64;
        auto h0=Texture(C,1,DXGI_FORMAT_R32G32B32A32_FLOAT,OCEAN_SRV_H0,0,false);
        auto fft=Texture(C*2,1,DXGI_FORMAT_R32G32B32A32_FLOAT,0,OCEAN_UAV_FFT);
        // Written as parity 1, so the kernels have to follow the parity rather than a fixed array.
        auto history=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R16G16B16A16_FLOAT,OCEAN_SRV_DISP0,OCEAN_UAV_DISP0_MIPS);
        auto disp=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R16G16B16A16_FLOAT,OCEAN_SRV_DISP1,OCEAN_UAV_DISP1_MIPS);
        auto deriv=Texture(C,OCEAN_MIP_LEVELS,DXGI_FORMAT_R16G16B16A16_FLOAT,OCEAN_SRV_DERIV,OCEAN_UAV_DERIV_MIPS);
        OceanParamsGPU P{};P.choppiness=0.8f;P.cascadeLength={L,L,L,L};P.dispParity=1;
        // The short-wave chop ramps up across the test lattice and fades again past it, so every
        // part of the profile is exercised.
        P.chopBand={float(std::log2(.15)),float(std::log2(.5)),float(std::log2(.8)),1.5f};
        auto params=Buffer(sizeof(P),D3D12_HEAP_TYPE_UPLOAD);void* ptr;Check(params->Map(0,nullptr,&ptr));memcpy(ptr,&P,sizeof(P));params->Unmap(0,nullptr);
        D3D12_SHADER_RESOURCE_VIEW_DESC ps{};ps.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;ps.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;ps.Buffer.NumElements=1;ps.Buffer.StructureByteStride=sizeof(P);dev->CreateShaderResourceView(params.Get(),&ps,At(OCEAN_SRV_PARAMS));
        // Every cascade gets its own random spectrum; the kernels rebuild the wave vector of node
        // (x, z) as ((x, z) - N/2) 2 pi / L and its frequency from deep-water dispersion.
        std::vector<XMFLOAT4> initial(C*N*N);std::mt19937 gen(11);std::normal_distribution<double> random;
        for(int c=0;c<C;++c){std::vector<std::complex<double>> a(N*N);
            for(int z=1;z<N;++z)for(int x=1;x<N;++x){if(x==N/2&&z==N/2)continue;a[z*N+x]=.015*std::complex<double>(random(gen),random(gen));}
            for(int z=0;z<N;++z)for(int x=0;x<N;++x){int i=z*N+x;auto b=std::conj(a[((N-z)%N)*N+(N-x)%N]);initial[c*N*N+i]={float(a[i].real()),float(a[i].imag()),float(b.real()),float(b.imag())};}}
        Upload(h0.Get(),initial);push.time=2.3f;
        Dispatch(0,N,C,1);Barrier(fft.Get());Dispatch(1,N,C,1);Barrier(disp.Get());Barrier(deriv.Get());
        auto D=ReadHalf(disp.Get()),G=ReadHalf(deriv.Get()),Hist=ReadHalf(history.Get());
        double err2=0,ref2=0,meanSquare=0,spectral=0,imag2=0;
        for(int c=0;c<C;++c){
            std::vector<std::complex<double>> H(N*N);
            for(int i=0;i<N*N;++i){auto h=initial[c*N*N+i];const double kx=(i%N-N/2)*2*pi/L,kz=(i/N-N/2)*2*pi/L;
                auto phase=std::polar(1.,-std::sqrt(9.80665*std::hypot(kx,kz))*push.time);H[i]=std::complex<double>(h.x,h.y)*phase+std::complex<double>(h.z,h.w)*std::conj(phase);spectral+=std::norm(H[i]);}
            for(int z=0;z<N;++z)for(int x=0;x<N;++x){std::array<std::complex<double>,8> ref{};
                for(int j=0;j<N*N;++j){const double kx=(j%N-N/2)*2*pi/L,kz=(j/N-N/2)*2*pi/L,k=std::hypot(kx,kz),invK=k>0?1/k:0;
                    auto h=H[j]*std::polar(1.,2*pi*((j%N-N/2)*x+(j/N-N/2)*z)/N);const std::complex<double> I(0,1);
                    auto smooth=[](double t){t=std::clamp(t,0.0,1.0);return t*t*(3-2*t);};const double l=std::log2(std::max(k,1e-9));
                    const double chop=P.choppiness*(1+P.chopBand.w*smooth((l-P.chopBand.x)/(P.chopBand.y-P.chopBand.x))*(1-smooth(l-P.chopBand.z)));
                    ref[0]+=I*(chop*kx*invK)*h;ref[1]+=h;ref[2]+=I*(chop*kz*invK)*h;ref[3]-=(chop*kx*kz*invK)*h;
                    ref[4]+=I*kx*h;ref[5]+=I*kz*h;ref[6]-=(chop*kx*kx*invK)*h;ref[7]-=(chop*kz*kz*invK)*h;}
                int i=c*N*N+z*N+x;const float values[]={D[i].x,D[i].y,D[i].z,D[i].w,G[i].x,G[i].y,G[i].z,G[i].w};
                for(int j=0;j<8;++j){err2+=std::pow(values[j]-ref[j].real(),2);ref2+=std::norm(ref[j]);imag2+=std::pow(ref[j].imag(),2);}
                meanSquare+=D[i].y*D[i].y;
            }
        }
        for(auto v:Hist)Require(v.x==0&&v.y==0&&v.z==0&&v.w==0,"The history array must not be written");
        // The fields are stored in half precision: 11 significant bits bound the error near 5e-4.
        double relative=std::sqrt(err2/ref2),parseval=std::abs(meanSquare/(N*N)/spectral-1);
        std::cout<<"Production GPU FFT versus double DFT relative L2 "<<relative<<", Parseval "<<parseval<<", reference imaginary residual "<<std::sqrt(imag2/ref2)<<'\n';
        Require(relative<2e-3 && parseval<3e-3 && std::sqrt(imag2/ref2)<1e-5,"FFT/derivative/Parseval/Hermitian check");
        // First mip level of both pyramids is the box average of the level below.
        push.u0=1;push.u1=1;Dispatch(2,1,1,C);Barrier(disp.Get());Barrier(deriv.Get());
        auto DM=ReadHalf(disp.Get(),1),GM=ReadHalf(deriv.Get(),1);const int n=N/2;
        for(int c=0;c<C;++c)for(int z=0;z<n;++z)for(int x=0;x<n;++x){XMFLOAT4 d{},g{};
            for(int j=0;j<2;++j)for(int i=0;i<2;++i){auto a=D[c*N*N+(z*2+j)*N+x*2+i],b=G[c*N*N+(z*2+j)*N+x*2+i];
                d.x+=a.x*.25f;d.y+=a.y*.25f;d.z+=a.z*.25f;d.w+=a.w*.25f;g.x+=b.x*.25f;g.y+=b.y*.25f;g.z+=b.z*.25f;g.w+=b.w*.25f;}
            auto v=DM[c*n*n+z*n+x],w=GM[c*n*n+z*n+x];
            Require(std::abs(v.x-d.x)+std::abs(v.y-d.y)+std::abs(v.z-d.z)+std::abs(v.w-d.w)<4e-3 &&
                    std::abs(w.x-g.x)+std::abs(w.y-g.y)+std::abs(w.z-g.z)+std::abs(w.w-g.w)<4e-3,"Mip average");}
        std::cout<<"PASS: actual production FftH/FftV/Mip kernels (16x16 test lattice, four cascades, parity 1)\n";
    }
};
int main(int argc,char**argv){try{Require(argc==2,"Shader directory argument required");Runner r(argv[1]);r.Test();return 0;}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
