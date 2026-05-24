#pragma once

#include <cstdint>
#include <vector>

#include "core/cubed_sphere.h"
#include "gpu/device_buffer.h"

namespace pb {

//====================================
//A ghost copy operation. dst_idx and src_idx are linear cell indices within
//the full 6-face buffer of a Field (see CubedSphereGrid::global_index).
//src lies in the interior of src_face, dst lies in the halo region of
//dst_face. dst_face and src_face are stored for completeness and to ease
//introspection during testing; the linear indices are sufficient for a
//halo gather kernel since they already encode the face.
//====================================

struct GhostCopy {
    int dst_face;
    int dst_idx;
    int src_face;
    int src_idx;
};

//====================================
//NeighborTable enumerates, at construction time, every ghost cell on every
//face and resolves it to a source interior cell on a neighbor face. The
//mapping is computed by treating the ghost cell as a small angular
//extrapolation past the face edge: the cell center is fed through
//face_uv_to_sphere (the equiangular projection extends past u,v=+/-1 by a
//small amount for HALO=2) and then sphere_to_face_uv determines which face
//and interior cell the direction lands in.
//
//Corner ghost cells (both i and j out of [0, N)) are resolved the same way;
//sphere_to_face_uv simply picks whichever neighbor face wins the absolute
//component tie. The resulting source value is one cell on that face and is
//good enough for finite-difference stencils that may read corners.
//====================================

class NeighborTable {
public:
    NeighborTable() = default;
    explicit NeighborTable(const CubedSphereGrid& grid);

    const CubedSphereGrid& grid() const { return grid_; }

    const std::vector<GhostCopy>& host_copies() const { return host_copies_; }
    const DeviceBuffer<GhostCopy>& device_copies() const { return device_copies_; }
    std::size_t copy_count() const { return host_copies_.size(); }

    //CPU helper: 4-neighbor lookup for any cell, interior or edge. Interior
    //neighbors stay on the same face. Edge neighbors cross to an adjacent
    //face using the same projection rule the halo table uses.
    Neighbors4 neighbors(int face, int i, int j) const;

private:
    Neighbor resolve(int face, int i, int j) const;

    CubedSphereGrid         grid_;
    std::vector<GhostCopy>  host_copies_;
    DeviceBuffer<GhostCopy> device_copies_;
};

}
