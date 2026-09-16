#pragma once

#include <cstdint>
#include "coordinate_system.h"

namespace planet {

constexpr uint8_t  MAX_LOD      = 24;
constexpr uint8_t  CUBE_FACES   = 6;
constexpr uint64_t INVALID_NODE = 0xFFFFFFFFFFFFFFFFull;

struct QuadNode {
    uint8_t  face = 0;
    uint8_t  lod  = 0;
    uint32_t x    = 0;
    uint32_t y    = 0;
};

uint64_t pack_node_id  (const QuadNode& n);
QuadNode unpack_node_id(uint64_t id);

// Returns the requested child in face-local quadtree coordinates.
QuadNode child_node (const QuadNode& n, int quadrant);
QuadNode parent_node(const QuadNode& n);

enum QuadEdge : uint8_t {
    EDGE_NEG_S = 0, EDGE_POS_S = 1, EDGE_NEG_T = 2, EDGE_POS_T = 3,
};

QuadNode neighbor_node(const QuadNode& n, QuadEdge edge);

struct PlanetGeometry {
    DVec3  center { 0.0, 0.0, 0.0 };
    double radius = 1.0;
};

struct NodeGeometry {
    DVec3  center_world{};
    DVec3  center_dir{};
    DVec3  corners[4]{};
    double bounding_radius = 0.0;
    double edge_length     = 0.0;
    double angular_radius  = 0.0;
};

DVec3 cube_to_sphere_dir(uint8_t face, double s, double t);

// Computes world bounds and angular extent for one quadtree node.
NodeGeometry compute_node_geometry(const QuadNode& n, const PlanetGeometry& planet);

}
