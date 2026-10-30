#pragma once
#include "../planet/stream_orchestrator.h"
#include "../Lighting/DynamicLights.h"

struct DeviceContext;
struct MaterialSoA;

// Geometry a host application streams in at runtime (the Unity plugin). It takes the voxel world's place: the first
// external stream slot, the voxel share of the global vertex/index/material-ID buffers and instance records, and the
// opaque/alpha hit group pair. It also gets a block of material slots and of bindless texture slots, since the
// material buffer and the descriptor heap are sized once. Its emissive triangles join the light tree as the voxel
// world's do.
struct IHostStream : planet::IExternalStream, IDynamicLights {
    struct Reservation {
        uint32_t vertexElems = 0;
        uint32_t indexElems = 0;
        uint32_t matIdElems = 0; // one per triangle
        uint32_t instanceSlots = 0;
        uint32_t materials = 0;
    };
    struct Binding {
        ID3D12Device5* device = nullptr;
        DeviceContext* ctx = nullptr;
        ID3D12Resource* vertexGlobal = nullptr; // NON_PIXEL_SHADER_RESOURCE between frames
        ID3D12Resource* indexGlobal = nullptr;  // NON_PIXEL_SHADER_RESOURCE between frames
        uint32_t combinedVertexCount = 0;
        uint32_t vertexBase = 0, indexBase = 0;     // elements
        ID3D12Resource* materialIds = nullptr;      // upload heap
        uint32_t matIdBase = 0;
        uint32_t propsBase = 0;                     // first instance record
        MaterialSoA* materials = nullptr;           // the scene's, kept in step with materialBuffer
        ID3D12Resource* materialBuffer = nullptr;   // upload heap, packed
        uint32_t materialBase = 0;
        ID3D12DescriptorHeap* heap = nullptr;       // shader visible; texture IDs are absolute heap indices
        uint32_t textureBase = 0, textureCount = 0;
    };
    virtual Reservation reservation() const = 0;
    virtual void bind(const Binding& b) = 0;
};
