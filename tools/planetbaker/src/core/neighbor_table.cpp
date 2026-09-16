#include "core/neighbor_table.h"

namespace pb {

//Map a (face, i, j) cell, which may be in the halo region, to an interior
//(face, i, j) on whichever face actually owns that direction. For interior
//inputs the result is the input unchanged.
Neighbor NeighborTable::resolve(int face, int i, int j) const {
    const int n = grid_.n;
    if (i >= 0 && i < n && j >= 0 && j < n) {
        return Neighbor{face, i, j};
    }

    float u = grid_.u_of_i(i);
    float v = grid_.v_of_j(j);
    Vec3f p = face_uv_to_sphere(face, u, v);

    int   nf = 0;
    float nu = 0.0f, nv = 0.0f;
    sphere_to_face_uv(p, nf, nu, nv);

    int ni = grid_.i_of_u(nu);
    int nj = grid_.j_of_v(nv);
    return Neighbor{nf, ni, nj};
}

Neighbors4 NeighborTable::neighbors(int face, int i, int j) const {
    return Neighbors4{
        resolve(face, i + 1, j),
        resolve(face, i - 1, j),
        resolve(face, i, j + 1),
        resolve(face, i, j - 1),
    };
}

NeighborTable::NeighborTable(const CubedSphereGrid& grid) : grid_(grid) {
    const int n      = grid_.n;
    const int halo   = CubedSphereGrid::HALO;
    const int stride = grid_.stride();

    //Reserve roughly the right amount: per face the halo is the full
    //(N+2H)^2 minus the N^2 interior.
    const std::size_t per_face = static_cast<std::size_t>(stride) * stride
                               - static_cast<std::size_t>(n) * n;
    host_copies_.reserve(6 * per_face);

    for (int face = 0; face < 6; ++face) {
        for (int j = -halo; j < n + halo; ++j) {
            for (int i = -halo; i < n + halo; ++i) {
                if (i >= 0 && i < n && j >= 0 && j < n) continue;

                Neighbor src = resolve(face, i, j);

                GhostCopy c;
                c.dst_face = face;
                c.dst_idx  = grid_.global_index(face, i, j);
                c.src_face = src.face;
                c.src_idx  = grid_.global_index(src.face, src.i, src.j);
                host_copies_.push_back(c);
            }
        }
    }

    device_copies_.resize(host_copies_.size());
    device_copies_.upload(host_copies_);
}

}
