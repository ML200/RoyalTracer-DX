#pragma once

#include <cstdint>
#include <vector>

#include "core/cubed_sphere.h"
#include "gpu/device_buffer.h"

namespace pb {

class NeighborTable;

//====================================
//Field<T> is the working data type for the simulator. It owns one device
//allocation that holds (N + 2*HALO)^2 cells for each of the six cube faces.
//Faces are laid out back-to-back in the buffer. Cell (face, i, j) maps to
//global linear index grid().global_index(face, i, j).
//
//Interior i, j run from 0 to N-1. Halo cells are accessible at i or j in
//[-HALO, -1] or [N, N+HALO-1]. Halo cells start out undefined and are
//filled by halo_exchange(), which gathers from precomputed source cells
//in the supplied NeighborTable.
//
//Field is move-only and trivially constructible (default ctor produces an
//empty field with grid().n == 0).
//====================================

template <typename T>
class Field {
public:
    Field() = default;
    explicit Field(const CubedSphereGrid& grid)
        : grid_(grid), buffer_(static_cast<std::size_t>(grid.total_cells())) {}

    const CubedSphereGrid& grid() const { return grid_; }

    T*       data()       { return buffer_.data(); }
    const T* data() const { return buffer_.data(); }

    std::size_t total_cells() const { return buffer_.count(); }
    std::size_t bytes()       const { return buffer_.bytes(); }
    bool        empty()       const { return buffer_.empty(); }

    void zero() { buffer_.zero(); }

    void upload(const std::vector<T>& host)   { buffer_.upload(host); }
    void download(std::vector<T>& host) const { buffer_.download(host); }

    //Fill ghost cells on every face from the corresponding interior cells on
    //neighbor faces. Requires the supplied table was built against the same
    //grid this field was constructed with; the caller is responsible.
    void halo_exchange(const NeighborTable& table);

private:
    CubedSphereGrid grid_;
    DeviceBuffer<T> buffer_;
};

}
