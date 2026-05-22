#pragma once
//====================================
//PLANET - CUBE SPHERE
//====================================
//Cube-sphere addressing: 6 cube faces, each the root of a quadtree. A quadtree
//node is (face, lod, x, y); node_id packs all four into a uint32. Provides the
//cube->sphere mapping and per-node geometry (corners, bounding sphere, angular
//size) used by the visibility / LOD pass.

#include <cstdint>
#include "coordinate_system.h"

namespace planet {

//====================================
//LIMITS / NODE ID
//====================================
//node_id layout (uint64): low word  face[0:3) | lod[3:8) | x[8:32)
//                         high word y[0:24)
//x and y get 24 bits each, so a cube face can be subdivided up to MAX_LOD
//times. Widened from a packed uint32 (12-bit x/y, MAX_LOD 12) so the quadtree
//can descend to sub-metre leaves near the camera under the triangle budget.
constexpr uint8_t  MAX_LOD      = 24;
constexpr uint8_t  CUBE_FACES   = 6;
constexpr uint64_t INVALID_NODE = 0xFFFFFFFFFFFFFFFFull;  // face bits = 7 -> never a real node

//====================================
//QUAD NODE
//====================================
struct QuadNode {
    uint8_t  face = 0;   // 0..5
    uint8_t  lod  = 0;   // 0..MAX_LOD
    uint32_t x    = 0;   // 0..(2^lod - 1)
    uint32_t y    = 0;
};

uint64_t pack_node_id  (const QuadNode& n);
QuadNode unpack_node_id(uint64_t id);

//children in quadrant order: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)
QuadNode child_node (const QuadNode& n, int quadrant);
QuadNode parent_node(const QuadNode& n);

//====================================
//EDGES / NEIGHBOURS
//====================================
//A quadtree node has four edges. Edge index: 0 = -s, 1 = +s, 2 = -t, 3 = +t
//(s is the face's x/U axis, t is the y/V axis).
enum QuadEdge : uint8_t {
    EDGE_NEG_S = 0, EDGE_POS_S = 1, EDGE_NEG_T = 2, EDGE_POS_T = 3,
};

//Same-LOD quadtree node directly across one edge of 'n'. Resolves cube-face
//crossings: the result may sit on another face, with the coordinate flip that
//seam implies. The cube-sphere is closed, so every edge has exactly one
//neighbour and this is always valid. Needed by the restricted-quadtree balance
//pass and by seam stitching.
QuadNode neighbor_node(const QuadNode& n, QuadEdge edge);

//====================================
//PLANET GEOMETRY
//====================================
struct PlanetGeometry {
    DVec3  center { 0.0, 0.0, 0.0 };  // FP64 world-space planet centre
    double radius = 1.0;              // metres, surface radius (no displacement)
};

//====================================
//NODE GEOMETRY
//====================================
//Everything the visibility / LOD pass needs about one quadtree node, in FP64
//world space. Surface-only - displacement arrives in Phase 3 and will inflate
//bounding_radius by the max relief.
struct NodeGeometry {
    DVec3  center_world{};        // node centre, on the sphere
    DVec3  center_dir{};          // unit direction from planet centre to centre
    DVec3  corners[4]{};          // node corners, on the sphere
    double bounding_radius = 0.0; // sphere at center_world that covers the node
    double edge_length     = 0.0; // longest world-space edge chord
    double angular_radius  = 0.0; // half-angle the node subtends at the planet centre
};

//map a cube-face coordinate (s,t in [-1,1]) to a unit sphere direction
DVec3 cube_to_sphere_dir(uint8_t face, double s, double t);

//full per-node geometry for a planet
NodeGeometry compute_node_geometry(const QuadNode& n, const PlanetGeometry& planet);

} // namespace planet
