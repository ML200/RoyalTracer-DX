#pragma once
#include <cstdint>
#include <vector>
#include "../planet/coordinate_system.h"
#include "LightTreeRefit.h"

// The voxel world's or a host's buffer regions for streamed emissive triangles: dense records with their light
// BLAS nodes, leaf indices and traversal trails, after the static scene's part of each buffer.
struct DynamicLightBinding {
    ID3D12Resource* records = nullptr;
    ID3D12Resource* nodes = nullptr;
    ID3D12Resource* leafIndex = nullptr;
    ID3D12Resource* trails = nullptr;
    uint32_t recordBase = 0, recordCapacity = 0;
    uint32_t nodeBase = 0, nodeCapacity = 0;
    uint32_t slotBase = 0;
};

// Emissive geometry that streams in at runtime and joins the light tree as extra TLAS leaves, one per light slot.
// The renderer binds the buffer regions (Renderer::BindVoxelLights, which leaves them in
// NON_PIXEL_SHADER_RESOURCE), snapshots the leaves for each light-tree refit, and reports which version the
// published tree contains, so a slot only enters instance records once the tree knows it.
struct IDynamicLights {
    virtual ~IDynamicLights() = default;
    virtual bool lights_wanted() const = 0;
    virtual uint32_t light_record_capacity() const = 0;
    virtual uint32_t light_node_capacity() const = 0;
    virtual void bind_lights(const DynamicLightBinding& b) = 0;
    virtual bool lights_bound() const = 0;
    virtual void set_light_slot_base(uint32_t base) = 0;
    virtual uint32_t light_version() const = 0;
    virtual void light_snapshot(const planet::DVec3& sceneOrigin, std::vector<lt::TLASExtraLeaf>& out,
                                uint32_t& version, uint32_t& slotCount) const = 0;
    virtual void on_light_tlas_published(uint32_t version, bool hasLeaves) = 0;
};
