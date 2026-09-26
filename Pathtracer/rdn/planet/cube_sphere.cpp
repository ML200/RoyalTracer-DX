#include "cube_sphere.h"
#include <algorithm>
#include <cmath>

namespace planet {

namespace {
constexpr DVec3 FACE_A[6] = {
    {  1,  0,  0 }, { -1,  0,  0 },
    {  0,  1,  0 }, {  0, -1,  0 },
    {  0,  0,  1 }, {  0,  0, -1 },
};
constexpr DVec3 FACE_U[6] = {
    {  0,  0, -1 }, {  0,  0,  1 },
    {  1,  0,  0 }, {  1,  0,  0 },
    {  1,  0,  0 }, { -1,  0,  0 },
};
constexpr DVec3 FACE_V[6] = {
    {  0,  1,  0 }, {  0,  1,  0 },
    {  0,  0, -1 }, {  0,  0,  1 },
    {  0,  1,  0 }, {  0,  1,  0 },
};

struct EdgeLink { uint8_t face; uint8_t edge; bool reverse; };
constexpr EdgeLink EDGE_LINK[6][4] = {
     { { 4, 1, false }, { 5, 0, false }, { 3, 1, true  }, { 2, 1, false } },
     { { 5, 1, false }, { 4, 0, false }, { 3, 0, false }, { 2, 0, true  } },
     { { 1, 3, true  }, { 0, 3, false }, { 4, 3, false }, { 5, 3, true  } },
     { { 1, 2, false }, { 0, 2, true  }, { 5, 2, true  }, { 4, 2, false } },
     { { 1, 1, false }, { 0, 0, false }, { 3, 3, false }, { 2, 2, false } },
     { { 0, 1, false }, { 1, 0, false }, { 3, 2, true  }, { 2, 3, true  } },
};
}

uint64_t pack_node_id(const QuadNode& n) {
    const uint32_t lo = (uint32_t)(n.face & 0x7u)
                      | ((uint32_t)(n.lod & 0x1Fu) << 3)
                      | ((n.x & 0xFFFFFFu) << 8);
    const uint32_t hi =  (n.y & 0xFFFFFFu);
    return (uint64_t)lo | ((uint64_t)hi << 32);
}

QuadNode unpack_node_id(uint64_t id) {
    const uint32_t lo = (uint32_t)(id & 0xFFFFFFFFull);
    const uint32_t hi = (uint32_t)(id >> 32);
    QuadNode n;
    n.face = (uint8_t)( lo        & 0x7u);
    n.lod  = (uint8_t)((lo >> 3 ) & 0x1Fu);
    n.x    =          ((lo >> 8 ) & 0xFFFFFFu);
    n.y    =           (hi        & 0xFFFFFFu);
    return n;
}

QuadNode child_node(const QuadNode& n, int quadrant) {
    QuadNode c;
    c.face = n.face;
    c.lod  = (uint8_t )(n.lod + 1);
    c.x    = (uint32_t)(n.x * 2 + (quadrant & 1));
    c.y    = (uint32_t)(n.y * 2 + ((quadrant >> 1) & 1));
    return c;
}

QuadNode parent_node(const QuadNode& n) {
    QuadNode p;
    p.face = n.face;
    p.lod  = n.lod ? (uint8_t)(n.lod - 1) : 0;
    p.x    = (uint32_t)(n.x >> 1);
    p.y    = (uint32_t)(n.y >> 1);
    return p;
}

// Crosses cube-face seams.
QuadNode neighbor_node(const QuadNode& n, QuadEdge edge) {
    const uint32_t N = 1u << n.lod;

    switch (edge) {
        case EDGE_NEG_S: if (n.x > 0)     return { n.face, n.lod, (uint32_t)(n.x - 1), n.y }; break;
        case EDGE_POS_S: if (n.x + 1 < N) return { n.face, n.lod, (uint32_t)(n.x + 1), n.y }; break;
        case EDGE_NEG_T: if (n.y > 0)     return { n.face, n.lod, n.x, (uint32_t)(n.y - 1) }; break;
        case EDGE_POS_T: if (n.y + 1 < N) return { n.face, n.lod, n.x, (uint32_t)(n.y + 1) }; break;
    }

    const EdgeLink lk     = EDGE_LINK[n.face][edge];
    const bool     s_edge = (edge == EDGE_NEG_S || edge == EDGE_POS_S);
    uint32_t       along  = s_edge ? n.y : n.x;
    if (lk.reverse) along = N - 1 - along;

    QuadNode r;
    r.face = lk.face;
    r.lod  = n.lod;
    switch (lk.edge) {
        case EDGE_NEG_S: r.x = 0;                 r.y = (uint32_t)along; break;
        case EDGE_POS_S: r.x = (uint32_t)(N - 1); r.y = (uint32_t)along; break;
        case EDGE_NEG_T: r.y = 0;                 r.x = (uint32_t)along; break;
        case EDGE_POS_T: r.y = (uint32_t)(N - 1); r.x = (uint32_t)along; break;
    }
    return r;
}

DVec3 cube_to_sphere_dir(uint8_t face, double s, double t) {
    const DVec3 cube = FACE_A[face] + FACE_U[face] * s + FACE_V[face] * t;
    return normalize(cube);
}

NodeGeometry compute_node_geometry(const QuadNode& n, const PlanetGeometry& planet) {
    const double cells = double(1u << n.lod);
    const double s0 = -1.0 + 2.0 *  double(n.x)        / cells;
    const double s1 = -1.0 + 2.0 * (double(n.x) + 1.0) / cells;
    const double t0 = -1.0 + 2.0 *  double(n.y)        / cells;
    const double t1 = -1.0 + 2.0 * (double(n.y) + 1.0) / cells;

    NodeGeometry g;
    g.center_dir   = cube_to_sphere_dir(n.face, 0.5 * (s0 + s1), 0.5 * (t0 + t1));
    g.center_world = planet.center + g.center_dir * planet.radius;

    const double cs[4] = { s0, s1, s0, s1 };
    const double ct[4] = { t0, t0, t1, t1 };
    DVec3 corner_dir[4];
    for (int i = 0; i < 4; ++i) {
        corner_dir[i] = cube_to_sphere_dir(n.face, cs[i], ct[i]);
        g.corners[i]  = planet.center + corner_dir[i] * planet.radius;
    }

    double max_r = 0.0;
    for (int i = 0; i < 4; ++i)
        max_r = std::max(max_r, length(g.corners[i] - g.center_world));
    g.bounding_radius = max_r * 1.0001;

    const int ring[5] = { 0, 1, 3, 2, 0 };
    double max_e = 0.0;
    for (int i = 0; i < 4; ++i)
        max_e = std::max(max_e, length(g.corners[ring[i]] - g.corners[ring[i + 1]]));
    g.edge_length = max_e;

    double max_a = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double c = std::clamp(dot(g.center_dir, corner_dir[i]), -1.0, 1.0);
        max_a = std::max(max_a, std::acos(c));
    }
    g.angular_radius = max_a;
    return g;
}

}
