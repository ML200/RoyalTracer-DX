// Headless production-shader smoke test and linear-radiance preview. No DLSS runtime needed.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "../shaders/DlssGuideLayout.h"
#include "../shaders/CumulusLayout.h"
using Microsoft::WRL::ComPtr;
using namespace DirectX;
void Check(HRESULT hr) { if(FAILED(hr)) {char s[80];sprintf_s(s,"D3D12: %08lx",hr);throw std::runtime_error(s);} }
void Require(bool v,const char* s) {if(!v)throw std::runtime_error(s);}
struct GPU {
    ComPtr<ID3D12Device> dev; ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12InfoQueue> diagnostics;
    ComPtr<ID3D12CommandAllocator> alloc; ComPtr<ID3D12GraphicsCommandList> cmd;
    ComPtr<ID3D12RootSignature> root; ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Fence> fence; HANDLE event=CreateEvent(nullptr,FALSE,FALSE,nullptr); UINT64 serial=0;
    std::array<ComPtr<ID3D12Resource>,9> tex;
    std::array<ComPtr<ID3D12PipelineState>,13> pso;
    ComPtr<ID3D12PipelineState> splitPreview;
    ComPtr<ID3D12Resource> cb,out,readback,cloudQueries,scratch;
    ComPtr<ID3D12QueryHeap> queries; ComPtr<ID3D12Resource> ticks;
    std::array<float,192> camera{}; std::array<UINT,57> constants{};
    UINT width=640,height=360;
    std::array<float,9> densityKey{};UINT densityEpoch=0;
    ~GPU(){CloseHandle(event);}
    ComPtr<ID3D12Resource> Buffer(UINT64 bytes,D3D12_HEAP_TYPE type) {
        D3D12_HEAP_PROPERTIES h{};h.Type=type; D3D12_RESOURCE_DESC d{};
        d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;d.DepthOrArraySize=1;
        d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if(type==D3D12_HEAP_TYPE_DEFAULT)d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        auto state=type==D3D12_HEAP_TYPE_DEFAULT?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:
            type==D3D12_HEAP_TYPE_UPLOAD?D3D12_RESOURCE_STATE_GENERIC_READ:D3D12_RESOURCE_STATE_COPY_DEST;
        ComPtr<ID3D12Resource> r;Check(dev->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_PPV_ARGS(&r)));return r;
    }
    GPU(const std::string& dir,UINT w,UINT h):width(w),height(h) {
        ComPtr<ID3D12Debug> debug;if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))debug->EnableDebugLayer();
        Check(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&dev)));
        ComPtr<IDXGIFactory4> factory;ComPtr<IDXGIAdapter1> adapter;
        Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        Check(factory->EnumAdapterByLuid(dev->GetAdapterLuid(),IID_PPV_ARGS(&adapter)));
        DXGI_ADAPTER_DESC1 adapterDesc{};Check(adapter->GetDesc1(&adapterDesc));
        std::wstring adapterName=adapterDesc.Description;
        std::cout<<"GPU adapter: "<<std::string(adapterName.begin(),adapterName.end())<<"\n";
        dev.As(&diagnostics);
        D3D12_COMMAND_QUEUE_DESC q{};Check(dev->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)));
        Check(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)));
        Check(dev->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc.Get(),nullptr,IID_PPV_ARGS(&cmd)));Check(cmd->Close());
        Check(dev->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        // Production noise/light/environment/ambient LUT bindings; UAV63 is test output.
        const UINT regs[]={52,53,49,51,28,29,25,27,54,30,40,55,31,32,8,56,33,57,58,45,46};
        D3D12_DESCRIPTOR_RANGE ranges[21]{};
        for(UINT i=0;i<21;i++) {ranges[i].RangeType=(i<4||i==8||i==10||i==11||i==15||i==17||i==18)?D3D12_DESCRIPTOR_RANGE_TYPE_SRV:D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[i].NumDescriptors=1;ranges[i].BaseShaderRegister=regs[i];ranges[i].OffsetInDescriptorsFromTableStart=i;}
        D3D12_ROOT_PARAMETER p[4]{};p[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;p[0].DescriptorTable={21,ranges};
        p[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;p[1].Descriptor.ShaderRegister=0;
        p[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;p[2].Constants={1,0,57};
        p[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;p[3].Descriptor.ShaderRegister=63;
        D3D12_STATIC_SAMPLER_DESC samplers[2]{};
        for(UINT i=0;i<2;i++) {auto& s=samplers[i];s.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
            s.AddressU=s.AddressV=s.AddressW=i?D3D12_TEXTURE_ADDRESS_MODE_CLAMP:D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            s.ShaderRegister=i;s.MaxLOD=D3D12_FLOAT32_MAX;s.MaxAnisotropy=1;s.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS;}
        D3D12_ROOT_SIGNATURE_DESC rd{4,p,2,samplers,D3D12_ROOT_SIGNATURE_FLAG_NONE};
        ComPtr<ID3DBlob> blob,error;Check(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error));
        Check(dev->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)));
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.NumDescriptors=21;hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(dev->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)));
        const UINT sizes[4][3]={{CUMULUS_NOISE_SIZE,CUMULUS_NOISE_SIZE,CUMULUS_NOISE_SIZE},{CUMULUS_LIGHT_XZ,CUMULUS_LIGHT_Y,CUMULUS_LIGHT_XZ*CUMULUS_LIGHT_CASCADES},{256,64,1},{32,32,1}};
        #if CUMULUS_REFERENCE
        const DXGI_FORMAT noiseFormat=DXGI_FORMAT_R8G8B8A8_UNORM;
#else
        const DXGI_FORMAT noiseFormat=DXGI_FORMAT_R8G8_UNORM;
#endif
        const DXGI_FORMAT formats[]={noiseFormat,DXGI_FORMAT_R16G16_FLOAT,DXGI_FORMAT_R16G16B16A16_FLOAT,DXGI_FORMAT_R16G16B16A16_FLOAT};
        for(UINT i=0;i<4;i++) {
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC d{};
            d.Dimension=i<2?D3D12_RESOURCE_DIMENSION_TEXTURE3D:D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            d.Width=sizes[i][0];d.Height=sizes[i][1];d.DepthOrArraySize=(UINT16)sizes[i][2];d.MipLevels=1;d.Format=formats[i];d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&tex[i])));
            auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT inc=dev->GetDescriptorHandleIncrementSize(hd.Type);handle.ptr+=i*inc;
            D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.Format=d.Format;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if(i<2){s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE3D;s.Texture3D.MipLevels=1;}else{s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;s.Texture2D.MipLevels=1;}
            dev->CreateShaderResourceView(tex[i].Get(),&s,handle);handle.ptr+=4*inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=d.Format;
            if(i<2){u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE3D;u.Texture3D.WSize=d.DepthOrArraySize;}else u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
            dev->CreateUnorderedAccessView(tex[i].Get(),nullptr,&u,handle);
        }
        {
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC d{};
            d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=CUMULUS_ENV_W;d.Height=CUMULUS_ENV_H;d.DepthOrArraySize=CUMULUS_ENV_LAYERS;
            d.MipLevels=1;d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&tex[4])));
            auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT inc=dev->GetDescriptorHandleIncrementSize(hd.Type);handle.ptr+=8*inc;
            D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.Format=d.Format;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2DARRAY;s.Texture2DArray.MipLevels=1;s.Texture2DArray.ArraySize=CUMULUS_ENV_LAYERS;
            dev->CreateShaderResourceView(tex[4].Get(),&s,handle);handle.ptr+=inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=d.Format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2DARRAY;u.Texture2DArray.ArraySize=CUMULUS_ENV_LAYERS;
            dev->CreateUnorderedAccessView(tex[4].Get(),nullptr,&u,handle);handle.ptr+=inc;
            // Stars are disabled in this test. Supply a valid 2D texture for GetDimensions nonetheless.
            s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;s.Texture2D={0,1,0,0};dev->CreateShaderResourceView(tex[2].Get(),&s,handle);
        }
        {
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC d{};
            d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=CUMULUS_AMBIENT_W;d.Height=CUMULUS_AMBIENT_H;d.DepthOrArraySize=1;
            d.MipLevels=1;d.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&tex[5])));
            auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT inc=dev->GetDescriptorHandleIncrementSize(hd.Type);handle.ptr+=11*inc;
            D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.Format=d.Format;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;s.Texture2D.MipLevels=1;dev->CreateShaderResourceView(tex[5].Get(),&s,handle);handle.ptr+=inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=d.Format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;dev->CreateUnorderedAccessView(tex[5].Get(),nullptr,&u,handle);
        }
        {
            cloudQueries=Buffer(UINT64(width)*height*CUMULUS_QUERY_BYTES,D3D12_HEAP_TYPE_DEFAULT);
            auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT inc=dev->GetDescriptorHandleIncrementSize(hd.Type);handle.ptr+=13*inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=DXGI_FORMAT_R32_TYPELESS;u.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
            u.Buffer.NumElements=UINT(width*height*CUMULUS_QUERY_BYTES/4);u.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
            dev->CreateUnorderedAccessView(cloudQueries.Get(),nullptr,&u,handle);handle.ptr+=inc;
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC d{};
            d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=width;d.Height=height;d.DepthOrArraySize=3;
            d.MipLevels=1;d.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&scratch)));
            u={};u.Format=d.Format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2DARRAY;u.Texture2DArray.ArraySize=3;
            dev->CreateUnorderedAccessView(scratch.Get(),nullptr,&u,handle);
        }
        for (UINT i=0;i<1;i++) {
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
            auto d=tex[0]->GetDesc();d.Format=DXGI_FORMAT_R8G8_UNORM;
            Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&tex[6+i])));
            auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT inc=dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);handle.ptr+=(15+i)*inc;
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};view.Format=d.Format;view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE3D;view.Texture3D.MipLevels=1;
            dev->CreateShaderResourceView(tex[6+i].Get(),&view,handle);handle.ptr+=inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=d.Format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE3D;u.Texture3D.WSize=CUMULUS_NOISE_SIZE;
            dev->CreateUnorderedAccessView(tex[6+i].Get(),nullptr,&u,handle);
        }
        for(UINT i=7;i<9;i++) {
            bool tags=i==8;UINT stride=tags?1:CUMULUS_DENSITY_BRICK_VERTICES;
            D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;D3D12_RESOURCE_DESC d{};
            d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE3D;d.Width=CUMULUS_DENSITY_BRICKS_XZ*stride;
            d.Height=CUMULUS_DENSITY_BRICKS_Y*stride;d.DepthOrArraySize=UINT16(d.Width);d.MipLevels=1;
            d.Format=tags?DXGI_FORMAT_R32G32B32A32_SINT:DXGI_FORMAT_R16_FLOAT;d.SampleDesc.Count=1;d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Check(dev->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&d,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&tex[i])));
            auto handle=heap->GetCPUDescriptorHandleForHeapStart();UINT inc=dev->GetDescriptorHandleIncrementSize(hd.Type);handle.ptr+=(17+i-7)*inc;
            D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.Format=d.Format;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE3D;s.Texture3D.MipLevels=1;dev->CreateShaderResourceView(tex[i].Get(),&s,handle);handle.ptr+=2*inc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=d.Format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE3D;u.Texture3D.WSize=d.DepthOrArraySize;
            dev->CreateUnorderedAccessView(tex[i].Get(),nullptr,&u,handle);
        }
        {std::ifstream f(dir+"/densityCache.dxil",std::ios::binary);
        if(f){std::vector<char> code((std::istreambuf_iterator<char>(f)),{});
            D3D12_COMPUTE_PIPELINE_STATE_DESC d{};d.pRootSignature=root.Get();d.CS={code.data(),code.size()};
            Check(dev->CreateComputePipelineState(&d,IID_PPV_ARGS(&pso[12])));}}
        const char* names[]={"noise","light","trans","multi","preview","environment","ambient","seedSecondary","secondary","checkSecondary","material"};
        for(UINT i=0;i<11;i++){std::ifstream f(dir+"/"+names[i]+".dxil",std::ios::binary);Require(bool(f),"missing shader");
            std::vector<char> code((std::istreambuf_iterator<char>(f)),{});D3D12_COMPUTE_PIPELINE_STATE_DESC d{};d.pRootSignature=root.Get();d.CS={code.data(),code.size()};
            HRESULT hr=dev->CreateComputePipelineState(&d,IID_PPV_ARGS(&pso[i]));if(FAILED(hr)){std::cerr<<names[i]<<" PSO failed\n";Check(hr);}}
#if !CUMULUS_REFERENCE
        {std::ifstream f(dir+"/guides.dxil",std::ios::binary);
        if(f){std::vector<char> code((std::istreambuf_iterator<char>(f)),{});
            D3D12_COMPUTE_PIPELINE_STATE_DESC d{};d.pRootSignature=root.Get();d.CS={code.data(),code.size()};
            Check(dev->CreateComputePipelineState(&d,IID_PPV_ARGS(&pso[11])));splitPreview=pso[4];}}
#endif
        cb=Buffer(768,D3D12_HEAP_TYPE_UPLOAD);out=Buffer(UINT64(width)*height*64,D3D12_HEAP_TYPE_DEFAULT);
        readback=Buffer(UINT64(width)*height*64,D3D12_HEAP_TYPE_READBACK);ticks=Buffer(16,D3D12_HEAP_TYPE_READBACK);
        D3D12_QUERY_HEAP_DESC qd{};qd.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;qd.Count=2;Check(dev->CreateQueryHeap(&qd,IID_PPV_ARGS(&queries)));
        constants[0]=width;constants[1]=height;
        camera[99]=1e9f;camera[104]=48.52f;camera[105]=11.405f;camera[106]=172;camera[108]=11;
        camera[109]=1;camera[110]=2;camera[111]=5;camera[112]=1;camera[115]=10;
        camera[121]=12;camera[122]=8;camera[123]=4;camera[124]=4;camera[125]=1;camera[126]=.005f;
        const float c[]={1,.28f,1.5f,3.6f,.8f,14,1.0f,0,0,1,1,64,12,17,-1,.5f,2,1};
        std::memcpy(camera.data()+133,c,sizeof(c));Pose(2,0.3f);
#if CUMULUS_REFERENCE
        camera[145]=48;
#endif
    }
    void SunElevation(float degrees,bool setting) {
        // Invert the engine's sunrise-based phase clock (night speedup is one here).
        const double pi=3.141592653589793,gamma=2*pi/365*(camera[106]-1),lat=camera[104]*pi/180;
        const double dec=.006918-.399912*cos(gamma)+.070257*sin(gamma)-.006758*cos(2*gamma)+.000907*sin(2*gamma)-.002697*cos(3*gamma)+.001480*sin(3*gamma);
        auto hourAngle=[&](double el){return acos((sin(el*pi/180)-sin(lat)*sin(dec))/(cos(lat)*cos(dec)))*12/pi;};
        const double sunrise=12-hourAngle(-.833),target=12+(setting?1:-1)*hourAngle(degrees);
        camera[108]=float(fmod(target-sunrise-camera[105]/15+24,24));
    }
    void Pose(float altitudeMeters,float pitch,float yaw=0) {
        XMVECTOR eye=XMVectorSet(0,altitudeMeters,0,1),at=XMVectorSet(sin(yaw)*4000,altitudeMeters+pitch*4000,cos(yaw)*4000,1);
        XMMATRIX v=XMMatrixLookAtRH(eye,at,XMVectorSet(0,1,0,0)),p=XMMatrixPerspectiveFovRH(XM_PI/3.0f,float(width)/height,.01f,1e9f);
        XMMATRIX m[]={v,p,XMMatrixInverse(nullptr,v),XMMatrixInverse(nullptr,p),v,p};std::memcpy(camera.data(),m,sizeof(m));
    }
    void Begin(){void* data;Check(cb->Map(0,nullptr,&data));std::memcpy(data,camera.data(),768);cb->Unmap(0,nullptr);
        Check(alloc->Reset());Check(cmd->Reset(alloc.Get(),nullptr));ID3D12DescriptorHeap* h[]={heap.Get()};cmd->SetDescriptorHeaps(1,h);
        cmd->SetComputeRootSignature(root.Get());cmd->SetComputeRootDescriptorTable(0,heap->GetGPUDescriptorHandleForHeapStart());
        cmd->SetComputeRootConstantBufferView(1,cb->GetGPUVirtualAddress());cmd->SetComputeRoot32BitConstants(2,57,constants.data(),0);cmd->SetComputeRootUnorderedAccessView(3,out->GetGPUVirtualAddress());}
    void Barrier(ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};cmd->ResourceBarrier(1,&b);}
    void Dispatch(UINT i,UINT x,UINT y=1){cmd->SetPipelineState(pso[i].Get());cmd->Dispatch(x,y,1);D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;cmd->ResourceBarrier(1,&b);}
    void End(){Check(cmd->Close());ID3D12CommandList* c[]={cmd.Get()};queue->ExecuteCommandLists(1,c);Check(queue->Signal(fence.Get(),++serial));Check(fence->SetEventOnCompletion(serial,event));
        Require(WaitForSingleObject(event,30000)==WAIT_OBJECT_0,"GPU timeout");Check(dev->GetDeviceRemovedReason());
        if(diagnostics){bool errors=false;for(UINT64 i=0;i<diagnostics->GetNumStoredMessages();i++){
            SIZE_T size=0;Check(diagnostics->GetMessage(i,nullptr,&size));std::vector<char> storage(size);auto* message=(D3D12_MESSAGE*)storage.data();
            Check(diagnostics->GetMessage(i,message,&size));if(message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR){std::cerr<<message->pDescription<<"\n";errors=true;}}
            diagnostics->ClearStoredMessages();Require(!errors,"D3D12 validation errors");}}
    void TestSecondary(UINT n,float roughness=0){constants[40]=n;camera[96]=0;float savedDebug=camera[147];camera[147]=roughness;Begin();
        Dispatch(7,(width+7)/8,(height+7)/8);
        cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        Dispatch(8,(width+7)/8,(height+7)/8);
        cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);
        cmd->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,ticks.Get(),0);
        Dispatch(9,(width+7)/8,(height+7)/8);
        Barrier(out.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);cmd->CopyResource(readback.Get(),out.Get());Barrier(out.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);End();
        void* data;Check(ticks->Map(0,nullptr,&data));UINT64 frequency;Check(queue->GetTimestampFrequency(&frequency));
        auto* timestamps=(UINT64*)data;std::cout<<"Secondary resolve (roughness "<<roughness<<"): "<<double(timestamps[1]-timestamps[0])*1000/frequency<<" ms\n";ticks->Unmap(0,nullptr);
        Check(readback->Map(0,nullptr,&data));const float* v=(const float*)data;double error=0,weights[3]={};size_t nonempty=0;
        for(UINT i=0;i<width*height;i++)for(UINT c=4;c<8;c++){Require(std::isfinite(v[i*16+c]),"nonfinite secondary resolve");error=std::max(error,double(v[i*16+c]));}
        for(UINT i=0;i<width*height;i++)if(v[i*16+12]>0){nonempty++;for(UINT c=0;c<3;c++)weights[c]+=v[i*16+12+c];}
        Require(std::abs(weights[0]/nonempty-(n+1)*.5)<.03 && std::abs(weights[1]/nonempty-(n+1)*.25)<.03,"biased cloud miss reservoir selection");
        readback->Unmap(0,nullptr);Require(error<.01,"deferred secondary radiance, normalization or guide mismatch");
        camera[147]=savedDebug;
        std::cout<<"PASS: actual secondary origin, empty queue, RGB throughput, "<<n<<"-sample reservoir normalization, roughness and reflection guide ownership, unbiased selection (max error "<<error<<")\n";
    }
    void Prepare(){Begin();Dispatch(0,32768);for(UINT i=6;i<7;i++)Barrier(tex[i].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);Barrier(tex[0].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        for(UINT i=7;i<9;i++)Barrier(tex[i].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Dispatch(2,32,8);Dispatch(3,32,32);Barrier(tex[2].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Barrier(tex[3].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);Dispatch(6,32);Barrier(tex[5].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);Dispatch(1,8192+1024*(CUMULUS_LIGHT_CASCADES-1));
        Barrier(tex[1].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);Dispatch(5,2048);End();}
    void BakeDensity(UINT phases=CUMULUS_DENSITY_PHASES,UINT firstPhase=0) {
        if(!pso[12]||camera[152]<.5f||camera[140]!=0||camera[141]!=0)return;
        std::array<float,9> key={camera[134],camera[135],camera[136],camera[137],camera[139],camera[140],camera[141],camera[146],camera[150]};
        if(densityEpoch==0||key!=densityKey){densityKey=key;++densityEpoch;}
        std::memcpy(&camera[153],&densityEpoch,4);float frame=camera[96];double ms=0;
        for(UINT phase=0;phase<phases;phase++) {
            camera[96]=float(firstPhase+phase);Begin();
            cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
            for(UINT i=7;i<9;i++)Barrier(tex[i].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Dispatch(12,CUMULUS_DENSITY_BATCH);
            for(UINT i=7;i<9;i++)Barrier(tex[i].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);cmd->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,ticks.Get(),0);End();
            void* data;Check(ticks->Map(0,nullptr,&data));UINT64 frequency;Check(queue->GetTimestampFrequency(&frequency));auto* t=(UINT64*)data;
            ms+=double(t[1]-t[0])*1000/frequency;ticks->Unmap(0,nullptr);
        }
        camera[96]=frame;std::cout<<"Density cache update: "<<ms<<" ms, "<<phases<<" phases\n";
    }
    std::vector<float> Render(const std::string& path,UINT samples=1,UINT firstFrame=0,bool save=true){double ms=0.0;
        BakeDensity();
        Begin();cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);
        Barrier(tex[1].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Barrier(tex[5].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        Dispatch(6,32);Barrier(tex[5].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Dispatch(1,8192+1024*(CUMULUS_LIGHT_CASCADES-1));Barrier(tex[1].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);Dispatch(5,2048);
        cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);cmd->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,ticks.Get(),0);End();
        void* cacheData;Check(ticks->Map(0,nullptr,&cacheData));UINT64 cacheFrequency;Check(queue->GetTimestampFrequency(&cacheFrequency));auto* cacheTicks=(UINT64*)cacheData;
        std::cout<<"Light + secondary cache: "<<double(cacheTicks[1]-cacheTicks[0])*1000/cacheFrequency<<" ms\n";ticks->Unmap(0,nullptr);
        for(UINT frame=0;frame<samples;frame++) {
            camera[96]=float(frame+firstFrame);Begin();cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0);Dispatch(4,(width+7)/8,(height+7)/8);
            if(pso[4].Get()==splitPreview.Get() && pso[11])Dispatch(11,(width+7)/8,(height+7)/8);
            cmd->EndQuery(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1);cmd->ResolveQueryData(queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,2,ticks.Get(),0);End();
            void* data;Check(ticks->Map(0,nullptr,&data));UINT64 frequency;Check(queue->GetTimestampFrequency(&frequency));auto* t=(UINT64*)data;
            ms+=double(t[1]-t[0])*1000/frequency;ticks->Unmap(0,nullptr);
        }
        Begin();
        Barrier(out.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);cmd->CopyResource(readback.Get(),out.Get());Barrier(out.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);End();
        void* data;Check(readback->Map(0,nullptr,&data));std::vector<float> values(width*height*16);std::memcpy(values.data(),data,values.size()*4);readback->Unmap(0,nullptr);
        std::cout<<path<<": "<<ms/samples<<" ms at "<<width<<"x"<<height<<" (volume pass only, "<<samples<<" samples)\n";
        if(save){std::ofstream image(path,std::ios::binary);image.write((char*)values.data(),values.size()*4);}return values;}
};
void TestDetail(GPU& g,const std::string& dir) {
    const auto original=g.camera;const auto preview=g.pso[4];g.pso[4]=g.pso[10];
    size_t outward=0,broadChanged[2]={};double mass[3]={},variation[3]={};
    for(UINT slice=0;slice<4;slice++)for(UINT mode=0;mode<3;mode++) {
        g.camera[147]=.12f+.22f*slice;g.camera[150]=float(mode);
        if(mode==2)g.camera[139]=1.5f;else g.camera[139]=1;
        auto v=g.Render(dir+"/material-"+std::to_string(slice)+"-"+std::to_string(mode)+".bin");
        // Wind now deforms the enclosing body. Distant rays and shadow caches
        // must follow that deformation instead of retaining the original shape.
        g.camera[150]=0;auto base=g.Render("material baseline",1,0,false);
        for(size_t i=0;i<v.size();i+=16) {
            for(UINT c=0;c<6;c++)Require(std::isfinite(v[i+c])&&v[i+c]>=0&&v[i+c]<=1,"invalid detailed material");
            for(UINT c=0;c<2;c++)if(mode==1 && std::abs(v[i+4+c]-base[i+4+c])>.005f)broadChanged[c]++;
            if(mode==1 && base[i]==0 && v[i]>.005f)outward++;
            mass[mode]+=v[i];
            if((i/16)%g.width){double d=double(v[i])-v[i-16];variation[mode]+=d*d;}
        }
    }
    Require(outward>100,"fine density failed to create outward structures");
    for(UINT mode=0;mode<3;mode++)std::cout<<"Material detail "<<mode<<": integrated density "<<mass[mode]<<", spatial variation "<<variation[mode]<<"\n";
    Require(broadChanged[0]>100 && broadChanged[1]>100,"body deformation is missing from distant rays or shadow caches");
    std::cout<<"PASS: "<<outward<<" outward density samples, bounded material; distant/shadow body changes "<<broadChanged[0]<<" / "<<broadChanged[1]<<".\n";
    // At time zero, reversing wind changes only directional deformation. The
    // recovered shape uses the wind axis, so its zero-detail control must agree.
    g.camera=original;g.camera[100]=0;g.camera[147]=.34f;g.camera[141]=0;
    std::vector<float> forward,undeformed;
    size_t windChanged=0,windBroadChanged[2]={};
    for(UINT direction=0;direction<2;direction++) {
        g.camera[140]=direction ? -20.0f:20.0f;g.camera[150]=1;
        auto v=g.Render("wind-directed material",1,0,false);
        g.camera[150]=0;auto base=g.Render("undeformed material",1,0,false);
        if(direction==0){forward=std::move(v);undeformed=std::move(base);continue;}
        for(size_t i=0;i<v.size();i+=16) {
            Require(std::isfinite(v[i])&&v[i]>=0&&v[i]<=1,"invalid wind-deformed density");
            Require(std::memcmp(base.data()+i,undeformed.data()+i,6*sizeof(float))==0,"wind reversal changed the undeformed control");
            for(UINT c=0;c<2;c++)if(std::abs(v[i+4+c]-forward[i+4+c])>.005f)windBroadChanged[c]++;
            if(std::abs(v[i]-forward[i])>.005f)windChanged++;
        }
    }
    Require(windChanged>100,"billow deformation failed to follow wind direction");
    Require(windBroadChanged[0]>100 && windBroadChanged[1]>100,"distant/shadow body deformation failed to follow wind direction");
    std::cout<<"PASS: wind reversal deforms "<<windChanged<<" material samples and both broad representations; undeformed controls remain exact.\n";
    // Check the projected silhouette, not just density variations inside a cloud.
    g.pso[4]=preview;
    for(UINT pose=0;pose<3;pose++) {
        g.camera=original;
        if(pose==1)g.Pose(1600,.2f);
        if(pose==2)g.Pose(500,.8f);
        g.camera[150]=0;auto base=g.Render("undeformed silhouette",1,0,false);
        for(UINT strength=1;strength<=2;strength++) {
            g.camera[150]=float(strength);
            auto v=g.Render("wind silhouette",1,0,false);
            size_t changed=0,cloudPixels=0;
            for(size_t i=0;i<v.size();i+=16) {
                // The deterministic guide opacity is independent of lighting RNG.
                bool a=base[i+15]>.5f,b=v[i+15]>.5f;
                cloudPixels+=a||b;changed+=a!=b;
            }
            Require(cloudPixels>100 && changed>cloudPixels/100,"billow control changes detail without moving the projected outline");
            std::cout<<"PASS: silhouette pose "<<pose<<" strength "<<strength<<": "<<changed<<" mask pixels changed ("<<100.0*changed/cloudPixels<<"% of union).\n";
        }
    }
    g.camera=original;g.pso[4]=preview;
}
void BenchmarkLighting(GPU& g,const std::string& dir) {
    const auto current=g.pso[4];
    // Optional previous production shader, same RG8 resources. Compare kernels in
    // one process to avoid attributing changing GPU clocks/load to this revision.
    ComPtr<ID3D12PipelineState> previous;
    std::ifstream f("out/cumulus/paired-final1080/preview.dxil",std::ios::binary);
    if(f){std::vector<char> code((std::istreambuf_iterator<char>(f)),{});
        D3D12_COMPUTE_PIPELINE_STATE_DESC p{};p.pRootSignature=g.root.Get();p.CS={code.data(),code.size()};
        Check(g.dev->CreateComputePipelineState(&p,IID_PPV_ARGS(&previous)));}
    for(UINT trial=0;trial<3;trial++) {
        if(previous){g.pso[4]=previous;g.camera[150]=0;g.Render(dir+"/previous-"+std::to_string(trial),32,0,false);}
        g.pso[4]=current;
        for(UINT budget:{0u,1u,2u,4u}) {
            g.camera[149]=float(budget);g.camera[150]=0;
            g.Render(dir+"/budget-"+std::to_string(budget)+"-"+std::to_string(trial),32,0,false);
        }
        g.camera[149]=2;g.camera[150]=1;
        g.Render(dir+"/detail-"+std::to_string(trial),32,0,false);
    }
}
void TestAtmosphere(GPU& g) {
    const auto original=g.camera;
    g.Pose(2,.28f);g.SunElevation(30.0f,true);g.camera[134]=.75f;g.camera[147]=-8;
    auto shadow=g.Render("cloud shadows in air",1,0,false);
    size_t reduced=0;
    for(size_t i=0;i<shadow.size();i+=16)for(UINT c=0;c<3;c++) {
        Require(std::isfinite(shadow[i+c])&&shadow[i+c]>=0,"invalid shadowed haze");
        Require(shadow[i+c]<=shadow[i+4+c]+1e-7,"cloud shadow added direct atmospheric light");
        Require(std::abs(shadow[i+8+c]-shadow[i+12+c])<1e-6,"lighting changed atmospheric extinction");
        reduced+=shadow[i+c]<shadow[i+4+c]*.95f;
    }
    Require(reduced>100,"cloud cache did not shadow atmospheric scattering");
    g.camera[138]=0;
    auto clear=g.Render("zero-extinction atmosphere shadow control",1,0,false);
    for(size_t i=0;i<clear.size();i+=16)for(UINT c=0;c<3;c++)
        Require(std::abs(clear[i+c]-clear[i+4+c])<1e-7,"empty clouds changed atmospheric lighting");
    // A shell with zero cloud extinction must not turn haze into a sparse
    // random cloud-light estimate. Density jitter is identical in both runs.
    g.camera[147]=-3;g.camera[149]=0;
    auto full=g.Render("empty shell haze, full lighting",1,123,false);
    g.camera[149]=2;
    auto sparse=g.Render("empty shell haze, sparse lighting",1,123,false);
    for(size_t i=0;i<full.size();i++)Require(std::abs(full[i]-sparse[i])<1e-6,"cloud light budget changed empty-shell haze");
    g.camera=original;g.camera[147]=-9;g.SunElevation(-5.0f,true);
    auto diffuse=g.Render("diffuse field and atmosphere coordinates",1,0,false);
    for(size_t i=0;i<diffuse.size();i+=16) {
        for(UINT c=0;c<3;c++) {
            Require(std::isfinite(diffuse[i+c])&&diffuse[i+c]>=0,"invalid diffuse field");
            Require(diffuse[i+c]==diffuse[i+4+c],"local cloud profile imprinted diffuse illumination");
        }
        Require(diffuse[i+8]<.001f&&diffuse[i+9]<1e-6f,"atmosphere LUT coordinates do not round trip");
    }
    g.camera=original;g.camera[134]=1;g.camera[138]=80;g.camera[147]=-7;
    auto thick=g.Render("thick cloud column depth",1,0,false);float maxColumn=0;
    for(size_t i=0;i<thick.size();i+=16) {
        maxColumn=std::max(maxColumn,thick[i+1]);
        Require(std::isfinite(thick[i+1])&&thick[i+1]>=thick[i+2]-.01f,"invalid thick cloud column");
        Require(std::abs(thick[i+1]-thick[i+2]-thick[i+3])<.003f+.001f*thick[i+1],"thick column depth was clipped");
    }
    Require(maxColumn>80,"thick column control failed to exceed direct-light cutoff");
    g.camera=original;
    std::cout<<"PASS: cloud-shadowed haze, unchanged extinction, empty-cloud control, haze independent of sparse lighting, broad diffuse response and LUT round trip.\n";
}
void TestTwilight(GPU& g) {
    const auto original=g.camera;
    g.camera[138]=0;g.camera[147]=-6;
    auto v=g.Render("solar-disk and dense-twilight regression",1,0,false);
    for(UINT y=0;y<g.height;y++)for(UINT x=0;x<g.width;x++) {
        size_t i=(size_t(y)*g.width+x)*16;
        for(UINT c=0;c<7;c++)Require(std::isfinite(v[i+c]),"nonfinite twilight lighting");
        Require(v[i]==0 && v[i+1]==0,"planet shadow contaminated zero cloud optical depth");
        Require(v[i+2]>=0 && v[i+2]<=1 && v[i+3]>=0,"invalid solar-disk transmission");
        if(x>0) {
            Require(v[i+2]>=v[i-16+2],"solar disk visibility is not monotonic");
            Require(v[i+2]-v[i-16+2]<8.0/g.width,"solar disk has a hard visibility edge");
            Require(v[i+3]+1e-6>=v[i-16+3],"atmospheric sun transmission popped at the horizon");
        }
        if(x==0)Require(v[i+2]==0 && v[i+3]==0,"direct light leaked through the planet");
        if(x==g.width-1)Require(v[i+2]==1,"fully risen solar disk is still shadowed");
        if(x<g.width/4)Require(v[i+4]+v[i+5]+v[i+6]>1e-8,"dense clouds lost diffuse twilight light");
    }
    g.camera=original;
    g.camera[147]=-7;
    v=g.Render("vertical light-cache integration",1,0,false);
    for(size_t i=0;i<v.size();i+=16) {
        for(UINT c=0;c<4;c++)Require(std::isfinite(v[i+c]),"nonfinite vertical cloud light cache");
        Require(v[i]>=0 && v[i]<=80 && v[i+1]>=0 && v[i+1]<=g.camera[138]*g.camera[136]+.1f,"light-cache depth out of bounds");
        Require(v[i+1]+1e-4>=v[i+2],"upward cloud optical depth increased with altitude");
        // Check adjacent cells against their densities, allowing RG16F storage
        // roundoff. This also catches swapped axes, mixed columns and wrong units.
        Require(std::abs(v[i+1]-v[i+2]-v[i+3])<.003f+.001f*v[i+1],"vertical light-cache integration mismatch");
    }
    // At fixed geometry optical depth is linear in extinction until its stored
    // cap. The old early-out at 40 returned unsaturated values up to 80, causing
    // jumps in high scattering orders as a sunset shadow moved across cells.
    g.SunElevation(-1.3f,false);g.camera[138]=4;
    auto thin=g.Render("twilight cache linearity reference",1,0,false);
    g.camera[138]=16;
    auto thick=g.Render("twilight cache saturation continuity",1,0,false);
    double maxCacheError=0;
    for(size_t i=0;i<thin.size();i+=16) {
        double expected=std::min(80.0,4.0*thin[i]);
        maxCacheError=std::max(maxCacheError,std::abs(thick[i]-expected));
    }
    Require(maxCacheError<.13,"shadow cache early-out caused discontinuous multiple-scattering depth");
    g.camera=original;
    std::cout<<"PASS: shadow-cache extinction scaling and saturation continuity (max half-float error "<<maxCacheError<<").\n";
    std::cout<<"PASS: cloud extinction excludes planet occlusion, continuous solar-disk transmission, dense twilight fill.\n";
    std::cout<<"PASS: bounded, monotonic vertical light-cache integration.\n";
}
void TestLighting(GPU& g,const std::string& dir) {
    const auto original=g.camera;
    for(UINT scene=0;scene<10;scene++) {
        g.camera=original;g.camera[147]=-4;
        const char* names[]={"ground","dusk","dawn","above","inside","wisps","backlit","sunset-bottom","sunrise-bottom","after-sunset"};
        if(scene==1||scene==2)g.SunElevation(-.8f,scene==1);
        if(scene==3)g.Pose(5500,-.25f);
        if(scene==4)g.Pose(2500,.1f);
        if(scene==5){g.camera[134]=.16f;g.camera[139]=1.5f;g.Pose(1600,.2f);}
        if(scene==6){g.camera[147]=-5;auto sun=g.Render("sun direction",1,0,false);
            float horizontal=std::sqrt(sun[0]*sun[0]+sun[2]*sun[2]);
            Require(sun[1]>0 && horizontal>1e-4f,"expected an above-horizon sun for backlighting test");
            g.Pose(2,sun[1]/horizontal,std::atan2(sun[0],sun[2]));g.camera[147]=-4;}
        if(scene>=7) {
            g.SunElevation(scene==9?-3.0f:-1.3f,scene!=8);
            g.camera[147]=-5;auto sun=g.Render("twilight sun direction",1,0,false);
            g.Pose(2,.35f,std::atan2(sun[0],sun[2]));g.camera[147]=-4;
        }
        std::string prefix=dir+"/lighting-"+names[scene];
        g.camera[149]=0;
        auto reference=g.Render(prefix+"-full.bin");
        for(UINT budget:{1u,2u,4u}) {
            if(budget!=2 && scene!=0 && scene!=1 && scene!=6)continue;
            double firstError=0;
            for(UINT samples:{8u,32u,128u,512u}) {
                if(budget!=2 && (samples==32||samples==128))continue;
                g.camera[149]=float(budget);
                std::string variant=budget==2 ? prefix : prefix+"-budget"+std::to_string(budget);
                auto candidate=g.Render(variant+"-"+std::to_string(samples)+".bin",samples);
                double squaredError=0,energy=0,signedError=0,total=0,channelError[3]={},channelSum[3]={};
                for(size_t i=0;i<candidate.size();i+=16) {
                    // All non-radiance channels must remain bit-exact. No tolerance or
                    // refreshed reference can hide sampling-induced guide changes.
                    Require(std::memcmp(reference.data()+i+3,candidate.data()+i+3,13*sizeof(float))==0,"lighting budget changed extinction or RR guides");
                    for(UINT c=0;c<3;c++) {
                        Require(std::isfinite(candidate[i+c]),"nonfinite sparse lighting");
                        double delta=double(candidate[i+c])-reference[i+c];
                        squaredError+=delta*delta;energy+=double(reference[i+c])*reference[i+c];
                        signedError+=delta;total+=reference[i+c];
                        channelError[c]+=delta;channelSum[c]+=reference[i+c];
                    }
                }
                double relative=std::sqrt(squaredError/std::max(energy,1e-20));
                double bias=0;for(UINT c=0;c<3;c++)bias=std::max(bias,std::abs(channelError[c])/std::max(channelSum[c],1e-20));
                std::cout<<"Lighting convergence "<<names[scene]<<" budget "<<budget<<" "<<samples<<": relative RMSE "<<relative<<", signed mean error "<<signedError/std::max(total,1e-20)<<", max channel bias "<<bias<<"\n";
                if(samples==8)firstError=relative;
                if(samples==512)Require((firstError==0 ? relative==0 : relative<firstError*.3) && relative<.04 && bias<.005,"sparse cloud lighting failed convergence/energy check");
            }
        }
    }
    g.camera=original;g.camera[147]=-1;
    std::cout<<"PASS: sparse lighting converges to full shading; extinction and all RR guides remain bit-identical.\n";
}
void TestTemporal(GPU& g,const std::string& dir) {
    const auto original=g.camera;g.camera[147]=-3;g.camera[149]=0;
    auto first=g.Render(dir+"/temporal-first.bin",1,0);
    double changed=0;size_t eligible=0;
    for(UINT frame=1;frame<8;frame++) {
        auto v=g.Render("temporal density",1,frame,false);
        for(size_t i=0;i<v.size();i+=16) {
            for(UINT c:{4u,5u,6u,7u,11u,12u,13u,14u,15u})
                Require(v[i+c]==first[i+c],"temporal colour sampling changed an RR guide");
            if(first[i+15]>.02f && first[i+15]<.98f){changed+=std::abs(v[i+3]-first[i+3]);eligible++;}
        }
    }
    Require(eligible>100 && changed/eligible>.002,"cloud silhouette quadrature is frozen across frames");
    std::cout<<"PASS: moving density quadrature with bit-identical RR guides; edge opacity change "<<changed/eligible<<"\n";
    g.camera=original;g.Pose(3500,-.18f);g.SunElevation(30,true);g.camera[147]=-12;
    auto reference=g.Render(dir+"/haze-reference.bin",1,0);
    g.camera[147]=-14;auto frozen=g.Render(dir+"/haze-frozen.bin",1,0);
    g.camera[147]=-13;auto a=g.Render(dir+"/haze-first.bin",1,0),b=g.Render("fresh haze",1,1,false);
    double difference=0;
    for(size_t i=0;i<a.size();i+=16){for(UINT c=0;c<3;c++)difference+=std::abs(a[i+c]-b[i+c]);
        for(UINT c=8;c<11;c++)Require(a[i+c]==b[i+c],"air source jitter changed camera extinction");}
    Require(difference/(g.width*g.height)>.00001,"atmospheric shadow samples are frozen");
    auto error=[&](const std::vector<float>& v){double e=0,n=0;for(size_t i=0;i<v.size();i+=16)for(UINT c=0;c<3;c++) {
        double d=v[i+c]-reference[i+c];e+=d*d;n+=double(reference[i+c])*reference[i+c];}return sqrt(e/std::max(n,1e-20));};
    double firstError=error(a);g.camera[147]=-11;auto accumulated=g.Render(dir+"/haze-accumulated.bin",256);
    double finalError=error(accumulated),frozenError=error(frozen);
    std::cout<<"Haze reference RMS first "<<firstError<<", accumulated "<<finalError<<", frozen "<<frozenError<<"\n";
    Require(finalError<firstError*.55 && finalError<frozenError*.75 && finalError<.08,"temporal haze did not converge toward dense integration");
    g.camera=original;std::cout<<"PASS: fresh atmospheric sources, stationary air extinction and haze convergence.\n";
}
void TestSunOcclusion(GPU& g,const std::string& dir) {
    std::ifstream file(dir+"/sunOcclusion.dxil",std::ios::binary);
    Require(bool(file),"missing sun occlusion probe shader");
    std::vector<char> code((std::istreambuf_iterator<char>(file)),{});
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};desc.pRootSignature=g.root.Get();desc.CS={code.data(),code.size()};
    const auto original=g.pso[4];const auto camera=g.camera;
    Check(g.dev->CreateComputePipelineState(&desc,IID_PPV_ARGS(&g.pso[4])));
    g.Pose(2,0);g.SunElevation(35,true);g.camera[152]=0;
    g.camera[134]=.75f;g.camera[138]=14;
    const char* names[]={"opaque-sun","thin-sun","disabled-sun","empty-sun","zero-extinction-sun"};
    for(UINT mode=0;mode<5;mode++) {
        g.camera[133]=mode==2?0.0f:1.0f;
        g.camera[134]=mode==3?0.0f:.75f;
        g.camera[138]=mode==4?0.0f:mode==1?.3f:14.0f;
        auto v=g.Render(dir+"/"+names[mode]+".bin");
        size_t opaque=0,thin=0;double residual=0,thinError=0;
        for(size_t i=0;i<v.size();i+=16) {
            for(UINT c=0;c<16;c++)Require(std::isfinite(v[i+c]),"nonfinite sun occlusion output");
            Require(v[i+4]>1000,"sun occlusion test missed the bright disk");
            if(v[i+3]>25) {
                opaque++;
                for(UINT c=0;c<3;c++)residual=std::max(residual,double(v[i+c]));
            }
            if(mode==1 && v[i+3]>.1f && v[i+3]<3) {
                thin++;thinError=std::max(thinError,std::abs(v[i+7]-std::exp(-double(v[i+3]))));
            }
            if(mode>=2)Require(v[i+7]==1 && v[i]>1000,"clear sky unexpectedly obscures the sun");
        }
        std::cout<<names[mode]<<": opaque rays "<<opaque<<", maximum residual disk radiance "<<residual
            <<", thin rays "<<thin<<", maximum thin transmission error "<<thinError<<"\n";
        if(mode==0){Require(opaque>100,"sun occlusion test needs dense cloud cover");Require(residual<.002,"opaque clouds leak the solar disk after early termination");}
        if(mode==1){Require(thin>100,"sun occlusion test needs thin clouds");Require(thinError<.03,"thin clouds lost gradual solar transmission");}
    }
    g.pso[4]=original;g.camera=camera;
    std::cout<<"PASS: dense clouds obscure the sun; thin clouds and clear-sky controls retain transmission.\n";
}
void TestIntegration(GPU& g,const std::string& dir) {
    std::ifstream file(dir+"/integrationProbe.dxil",std::ios::binary);
    Require(bool(file),"missing integration probe shader");
    std::vector<char> code((std::istreambuf_iterator<char>(file)),{});
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};desc.pRootSignature=g.root.Get();desc.CS={code.data(),code.size()};
    auto original=g.pso[4];Check(g.dev->CreateComputePipelineState(&desc,IID_PPV_ARGS(&g.pso[4])));
    auto v=g.Render("integration probes",1,0,false);g.pso[4]=original;
    double meanError=0,secondError=0,inverseError=0;
    for(UINT row=0;row<8;row++) {
        const float* c=v.data()+size_t(row)*g.width*16;
        double s0=c[0],s1=c[1],length=c[2],opacity=-std::expm1(-.5*(s0+s1)*length);
        // Independent physical-space integration of sigma(t) T(t), not the GPU's
        // inverse-CDF quadrature. Dense midpoint integration resolves even tau=100.
        double m1=0,m2=0;const UINT n=65536;
        for(UINT j=0;j<n;j++) {
            double t=length*(j+.5)/n,sigma=s0+(s1-s0)*t/length;
            double w=sigma*std::exp(-s0*t-.5*(s1-s0)*t*t/length)*length/n;
            m1+=w*t;m2+=w*t*t;
        }
        if(opacity>0){m1/=opacity;m2/=opacity;}
        meanError=std::max(meanError,std::abs(c[5]-m1)/std::max(m1,1e-8));
        secondError=std::max(secondError,std::abs(c[6]-m2)/std::max(m2,1e-8));
    }
    for(size_t i=0;i<v.size();i+=16) {
        for(UINT k=0;k<16;k++)Require(std::isfinite(v[i+k]),"nonfinite integration/sun probe");
        double s0=v[i],s1=v[i+1],length=v[i+2],t=v[i+4],u=v[i+7];
        Require(t>=0 && t<=length,"sampled cloud event outside its cell");
        double target=-std::log1p(-u*v[i+3]);
        double actual=s0*t+.5*(s1-s0)*t*t/length;
        inverseError=std::max(inverseError,std::abs(target-actual));
        Require(std::abs(v[i+11]-1)<2e-6,"visible sun direction not normalized");
        if(v[i+8]>1e-4)Require(v[i+9]>=v[i+10]-2e-6,"visible sun centroid intersects the planet");
        if(i>=16 && (i/16)%g.width)Require(v[i+9]>=v[i-16+9]-2e-6,"solar centroid reverses while the disk rises");
    }
    std::cout<<"Integration max errors: inverse tau "<<inverseError<<", mean "<<meanError<<", second moment "<<secondError<<"\n";
    Require(inverseError<3e-4,"linear cloud-event inversion failed");
    Require(meanError<.01 && secondError<.04,"representative cloud depth moments exceed quadrature tolerance");
    std::ifstream airFile(dir+"/airQuadrature.dxil",std::ios::binary);
    Require(bool(airFile),"missing atmospheric quadrature probe");
    std::vector<char> airCode((std::istreambuf_iterator<char>(airFile)),{});desc.CS={airCode.data(),airCode.size()};
    Check(g.dev->CreateComputePipelineState(&desc,IID_PPV_ARGS(&g.pso[4])));
    auto air=g.Render("RGB air-source quadrature",1,0,false);g.pso[4]=original;
    double airError=0;
    for(UINT y=0;y<g.height;y++)for(UINT c=0;c<3;c++) {
        double mean=0;
        for(UINT x=0;x<g.width;x++){size_t i=(size_t(y)*g.width+x)*16;
            Require(std::isfinite(air[i+c]) && air[i+c]>=0 && air[i+3]>=0 && air[i+3]<=air[i+7],"invalid atmospheric quadrature weight/position");
            mean+=air[i+c]/g.width;}
        size_t i=size_t(y)*g.width*16;double ext=air[i+4+c],ds=air[i+7];
        double analytic=-std::expm1(-ext*ds)/ext;
        airError=std::max(airError,std::abs(mean-analytic)/analytic);
    }
    Require(airError<.003,"RGB atmospheric source weights failed analytic integration");
    std::cout<<"PASS: atmospheric source quadrature matches analytic RGB integral; max relative error "<<airError<<"\n";
    std::cout<<"PASS: analytic cell extinction, bounded event/depth moments, visible solar centroid and degenerate night direction.\n";
    TestSunOcclusion(g,dir);
}
void TestDensityParity(GPU& g,const std::string& dir) {
    std::ifstream f(dir+"/densityParity.dxil",std::ios::binary);Require(bool(f),"missing density parity shader");
    std::vector<char> code((std::istreambuf_iterator<char>(f)),{});
    D3D12_COMPUTE_PIPELINE_STATE_DESC d{};d.pRootSignature=g.root.Get();d.CS={code.data(),code.size()};
    auto pso=g.pso[4];auto original=g.camera;
    ComPtr<ID3D12PipelineState> candidate,reference;
    Check(g.dev->CreateComputePipelineState(&d,IID_PPV_ARGS(&candidate)));
    std::ifstream refFile(dir+"/densityReference.dxil",std::ios::binary);Require(bool(refFile),"missing density reference shader");
    std::vector<char> refCode((std::istreambuf_iterator<char>(refFile)),{});d.CS={refCode.data(),refCode.size()};
    Check(g.dev->CreateComputePipelineState(&d,IID_PPV_ARGS(&reference)));
    size_t samples=0,nonempty=0;
    const float coverage[]={0,.01f,.16f,.28f,.75f,1};
    const float scale[]={.25f,.8f,3},thickness[]={.3f,3.6f,8},base[]={.2f,1.5f,8};
    const float detail[]={0,.3f,1,1.5f};
    for(UINT scenario=0;scenario<48;scenario++) {
        g.camera=original;g.camera[134]=coverage[scenario%6];g.camera[137]=scale[(scenario/2)%3];
        g.camera[135]=base[(scenario/3)%3];g.camera[136]=thickness[(scenario/5)%3];
        g.camera[139]=detail[(scenario/6)%4];g.camera[150]=float((scenario/7)%3);
        g.camera[140]=(scenario%3-1.0f)*40;g.camera[141]=(scenario%5-2.0f)*20;
        g.camera[146]=float(scenario)*2.07f;g.camera[100]=scenario*37.0f;
        g.pso[4]=reference;auto ref=g.Render("density reference scenario "+std::to_string(scenario),1,0,false);
        g.pso[4]=candidate;auto v=g.Render("density parity scenario "+std::to_string(scenario),1,0,false);
        for(size_t i=0;i<v.size();i+=16) {
            Require(std::isfinite(v[i]) && v[i]>=0 && v[i]<=1,"invalid optimized density");
            if(std::memcmp(v.data()+i,ref.data()+i,4*sizeof(float))!=0) {
                std::cerr<<"Density mismatch scenario "<<scenario<<", pixel "<<i/16<<": "
                    <<v[i]<<","<<v[i+1]<<","<<v[i+2]<<" versus "<<ref[i]<<","<<ref[i+1]<<","<<ref[i+2]<<"\n";
                std::ofstream dump(dir+"/density-parity-failure.bin",std::ios::binary);
                dump.write((char*)v.data(),v.size()*sizeof(float));
                std::ofstream refDump(dir+"/density-parity-reference.bin",std::ios::binary);
                refDump.write((char*)ref.data(),ref.size()*sizeof(float));
                Require(false,"conservative density optimization changed the approved field");
            }
            samples++;nonempty+=v[i]>0;
        }
    }
    Require(nonempty>1000,"density parity did not exercise cloud interiors");
    g.camera=original;g.pso[4]=pso;
    std::cout<<"PASS: "<<samples<<" bit-identical material samples, including "<<nonempty
        <<" nonempty samples; UI extremes, footprint boundaries, coarse/fine paths, wind and seeds.\n";
}
void TestDensityCache(GPU& g,const std::string& dir) {
    std::ifstream f(dir+"/densityCacheProbe.dxil",std::ios::binary);Require(bool(f),"missing density cache probe");
    std::vector<char> code((std::istreambuf_iterator<char>(f)),{});
    D3D12_COMPUTE_PIPELINE_STATE_DESC d{};d.pRootSignature=g.root.Get();d.CS={code.data(),code.size()};
    auto preview=g.pso[4];Check(g.dev->CreateComputePipelineState(&d,IID_PPV_ARGS(&g.pso[4])));
    g.camera[152]=1;g.BakeDensity(0);
    // Render normally warms all phases. Temporarily suppress automatic bakes so
    // missing, stale and partially moved windows can actually be exercised.
    auto bake=g.pso[12];
    auto probe=[&](const char* label,bool expectedHit) {
        g.pso[12]=nullptr;auto v=g.Render(label,1,0,false);g.pso[12]=bake;
        size_t hits=0,negative=0,worstVertex=0;double mse=0,energy=0;float maxVertexError=0;
        for(size_t i=0;i<v.size();i+=16) {
            Require(std::isfinite(v[i+1])&&v[i+1]>=0&&v[i+1]<=1,"invalid cached density");
            hits+=v[i]>0;negative+=v[i+7]>0;
            mse+=double(v[i+3])*v[i+3];energy+=double(v[i+2])*v[i+2];
            if(v[i+6]>maxVertexError){maxVertexError=v[i+6];worstVertex=i;}
        }
        std::cout<<"Density cache hits "<<hits<<" / "<<v.size()/16<<", expected "<<expectedHit<<", vertex max "<<maxVertexError<<"\n";
        Require(hits==(expectedHit?v.size()/16:0),"density cache accepted a stale block or missed a valid block");
        if(expectedHit) {
            std::cout<<"Worst vertex sample "<<worstVertex/16<<", stored "<<v[worstVertex+4]<<", reference "<<v[worstVertex+5]<<", P "<<v[worstVertex+8]<<","<<v[worstVertex+9]<<","<<v[worstVertex+10]<<"\n";
            Require(maxVertexError<.0006f,"baked vertex mismatch, including negative world coordinates");
            Require(negative>1000,"cache regression did not cover negative world blocks");
            std::cout<<"Density cache interpolation relative RMS "<<sqrt(mse/std::max(energy,1e-20))<<", vertex max "<<maxVertexError<<"\n";
        }
    };
    probe("cold density cache",false);
    g.BakeDensity(1);
    g.pso[12]=nullptr;auto partial=g.Render("partially populated density cache",1,0,false);g.pso[12]=bake;
    for(size_t i=0;i<partial.size()/16;i++)
        Require((partial[i*16]>0)==(i%CUMULUS_DENSITY_BRICK_COUNT<CUMULUS_DENSITY_BATCH),"partial cache exposed an unfinished brick");
    g.BakeDensity();probe("warm density cache",true);
    g.camera[147]=20;probe("wrapped slot has different world identity",false);g.camera[147]=-1;
    g.camera[147]=21;probe("filtered footprint fallback",false);g.camera[147]=-1;
    g.camera[140]=20;probe("animated wind fallback",false);g.camera[140]=0;
    g.camera[152]=0;probe("disabled density cache",false);g.camera[152]=1;
    g.camera[134]=.48f;g.BakeDensity(0);probe("changed weather invalidates old epoch",false);
    g.BakeDensity();probe("rebuilt weather",true);
    g.camera[44]-=1500;g.BakeDensity();probe("camera moved into negative world coordinates",true);
    g.BakeDensity();probe("steady cache reuse",true);
    g.pso[4]=preview;
    std::cout<<"PASS: density brick identities, quantized vertices, weather invalidation, movement, disabled/filtered/wind fallback.\n";
}
int main(int argc,char** argv)try {
    Require(argc==2||argc==4||argc==5,"expected output directory [width height [--lighting|--atmosphere|--twilight|--detail|--benchmark|--restored|--integration|--temporal|--density-parity]]");std::string dir=argv[1];
    GPU g(dir,argc>=4?std::stoul(argv[2]):640,argc>=4?std::stoul(argv[3]):360);g.Prepare();
    if(argc==5){std::string mode=argv[4];
        if(mode=="--density-cache"){TestDensityCache(g,dir);return 0;}
        if(mode=="--density-parity"){TestDensityParity(g,dir);return 0;}
        if(mode=="--temporal"){TestTemporal(g,dir);return 0;}
        if(mode=="--integration"){TestIntegration(g,dir);return 0;}
        if(mode=="--lighting"){TestLighting(g,dir);return 0;}
        if(mode=="--twilight"){TestTwilight(g);return 0;}
        if(mode=="--atmosphere"){TestAtmosphere(g);return 0;}
        if(mode=="--detail"){TestDetail(g,dir);return 0;}
        if(mode=="--benchmark"){BenchmarkLighting(g,dir);return 0;}
        if(mode=="--cached"){g.camera[152]=1;g.BakeDensity();}
        else {Require(mode=="--restored","unknown test mode");g.camera[149]=0;g.camera[150]=0;}
    }
    auto check=[&](const std::vector<float>& v,bool clear){size_t cloudy=0,distant=0;for(size_t i=0;i<v.size();i+=16){for(size_t c=0;c<16;c++)Require(std::isfinite(v[i+c]),"nonfinite cloud output");
        Require(v[i+3]>=0&&v[i+3]<=1,"opacity out of range");for(size_t c=8;c<11;c++)Require(v[i+c]>=0&&v[i+c]<=1,"transmittance out of range");
        if(v[i+15]>.02f){cloudy++;Require(v[i+7]>0,"missing cloud depth");float n=v[i+4]*v[i+4]+v[i+5]*v[i+5]+v[i+6]*v[i+6];Require(std::abs(n-1)<.01f,"non-unit cloud normal");
            double nZ=DLSS_GUIDE_DEPTH_NEAR,fZ=DLSS_GUIDE_DEPTH_FAR,z=nZ*fZ/(nZ+v[i+12]*(fZ-nZ));
            size_t pixel=i/16;double px=(double(pixel%g.width)+.5)/g.width*2-1,py=(double(pixel/g.width)+.5)/g.height*2-1;
            double focal=1.0/std::tan(XM_PI/6.0),aspect=double(g.width)/g.height;
            double rayZ=1000.0*v[i+7]/std::sqrt(1+px*px*aspect*aspect/(focal*focal)+py*py/(focal*focal));
            Require(v[i+12]>0&&v[i+12]<=1,"cloud depth clipped to sky");Require(std::abs(z-rayZ)<.002*rayZ,"device-depth projection mismatch");
            double expectedMV=g.camera[140]*g.camera[151]*focal*g.height*.5/rayZ;
            Require(std::abs(v[i+13]-expectedMV)<.002&&std::abs(v[i+14])<.002,"wind/stationary motion mismatch");
            if(v[i+7]>10)distant++;
        }}
        Require(clear?cloudy==0:cloudy>100,"unexpected cloud coverage");std::cout<<"PASS: radiance, transmittance, normals, reverse-Z, motion; cloudy "<<cloudy<<", beyond 10 km "<<distant<<"\n";};
    auto daylight=g.Render(dir+"/ground.bin",32);check(daylight,false);
    g.TestSecondary(1);g.TestSecondary(8);g.TestSecondary(8,.5f);g.TestSecondary(8,1.0f);
    g.camera[147]=-3;auto sample0=g.Render(dir+"/sample0.bin",1,0),sample1=g.Render(dir+"/sample1.bin",1,1);g.camera[147]=-1;
    double radianceChange=0,guideChange=0;
    for(size_t i=0;i<sample0.size();i+=16){for(size_t c=0;c<3;c++)radianceChange+=std::abs(sample0[i+c]-sample1[i+c]);
        for(size_t c=4;c<8;c++)guideChange=std::max(guideChange,double(std::abs(sample0[i+c]-sample1[i+c])));}
    Require(radianceChange/(g.width*g.height)>1e-4,"volume integration has no temporal sampling variance");
    Require(guideChange<1e-6,"stationary cloud guides follow noisy samples");std::cout<<"PASS: fresh stochastic radiance with stable normals/depth\n";
    auto checkTwilight=[&](const std::vector<float>& image){check(image,false);double dayR=0,dayB=0,red=0,blue=0;
        for(size_t i=0;i<image.size();i+=16){dayR+=daylight[i];dayB+=daylight[i+2];red+=image[i];blue+=image[i+2];}
        Require(red>1e-4 && red<dayR && red/std::max(blue,1e-10)>dayR/std::max(dayB,1e-10),"twilight lost illumination or atmospheric colour");};
    g.SunElevation(-.8f,true);checkTwilight(g.Render(dir+"/dusk.bin",32));
    g.SunElevation(-.8f,false);checkTwilight(g.Render(dir+"/dawn.bin",32));g.camera[108]=11;
    std::cout<<"PASS: below-horizon dawn/dusk retain coloured cloud illumination\n";
    g.camera[147]=1;check(g.Render(dir+"/finite-before-cloud.bin"),true);g.camera[147]=-1;
    g.camera[140]=20;g.camera[151]=.016f;check(g.Render(dir+"/wind.bin"),false);g.camera[140]=0;g.camera[151]=0;
    g.Pose(5500,-.25f);auto above=g.Render(dir+"/above.bin",32);check(above,false);
    g.Pose(0,-.25f);g.camera[102]=5500;auto rebased=g.Render(dir+"/rebased.bin",32);check(rebased,false);
    double rebaseError=0;for(size_t i=0;i<above.size();i+=16)for(size_t c=0;c<3;c++)rebaseError+=std::abs(above[i+c]-rebased[i+c]);
    Require(rebaseError/(g.width*g.height*3)<1e-4,"floating-origin rebase changed cloud radiance");g.camera[102]=0;
    std::cout<<"PASS: floating-origin radiance continuity\n";
    g.Pose(1600,.2f);check(g.Render(dir+"/side.bin",32),false);
    g.Pose(2500,.1f);check(g.Render(dir+"/inside.bin"),false);
    g.Pose(100000,-2.0f);check(g.Render(dir+"/high-altitude.bin",32),false);
    g.camera[133]=0;check(g.Render(dir+"/disabled.bin"),true);
    g.camera[133]=1;g.camera[134]=0;check(g.Render(dir+"/zero-coverage.bin"),true);
    std::cout<<"Cumulus GPU checks passed.\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
