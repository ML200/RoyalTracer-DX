#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>

namespace mc {

using BlockId = uint16_t;
constexpr BlockId AIR_ID = 0;

using Voxel = uint32_t; // Block id, flags, emissive count or water height.
constexpr Voxel VOX_ANY = 1u << 16;
constexpr Voxel VOX_ALL = 1u << 17;
constexpr Voxel VOX_OCC = 1u << 18;
constexpr int   VOX_EMIT_SHIFT = 19;
constexpr Voxel VOX_EMIT_MAX   = 8191u;
inline BlockId  voxel_id(Voxel v)       { return (BlockId)(v & 0xFFFFu); }
inline uint32_t voxel_emissive(Voxel v, bool water = false) { return water ? 0u : (uint32_t)(v >> VOX_EMIT_SHIFT); }
// Water top in blocks above the voxel floor.
inline uint32_t voxel_water_height(Voxel v, int level) {
    const uint32_t h = (uint32_t)(v >> VOX_EMIT_SHIFT);
    return level == 0 || h == 0u ? (1u << level) : h;
}
inline Voxel    make_voxel(BlockId id, bool any, bool all, bool occ, uint32_t emissive = 0u) {
    Voxel v = (Voxel)id;
    if (any) v |= VOX_ANY;
    if (all) v |= VOX_ALL;
    if (occ) v |= VOX_OCC;
    if (emissive) v |= (Voxel)(emissive > VOX_EMIT_MAX ? VOX_EMIT_MAX : emissive) << VOX_EMIT_SHIFT;
    return v;
}

enum Face : uint8_t { FACE_DOWN = 0, FACE_UP = 1, FACE_NORTH = 2, FACE_SOUTH = 3, FACE_WEST = 4, FACE_EAST = 5, FACE_NONE = 0xFF };
constexpr int FACE_DIR[6][3] = {
    { 0, -1,  0 },
    { 0,  1,  0 },
    { 0,  0, -1 },
    { 0,  0,  1 },
    {-1,  0,  0 },
    { 1,  0,  0 },
};

struct FaceProjection { int na; int ua; int va; float su; float sv; };
constexpr FaceProjection FACE_PROJECTION[6] = {
    { 1, 0, 2,  1.0f, -1.0f },
    { 1, 0, 2,  1.0f,  1.0f },
    { 2, 0, 1, -1.0f, -1.0f },
    { 2, 0, 1,  1.0f, -1.0f },
    { 0, 2, 1,  1.0f, -1.0f },
    { 0, 2, 1, -1.0f, -1.0f },
};
inline const char* face_name(int f) {
    static const char* n[6] = { "down", "up", "north", "south", "west", "east" };
    return (f >= 0 && f < 6) ? n[f] : "none";
}
inline int face_from_name(const char* s) {
    for (int i = 0; i < 6; ++i) {
        const char* n = face_name(i);
        int k = 0;
        while (n[k] && s[k] && n[k] == s[k]) ++k;
        if (!n[k] && !s[k]) return i;
    }
    if (s[0] == 'b' && s[1] == 'o') return FACE_DOWN;
    if (s[0] == 't' && s[1] == 'o') return FACE_UP;
    return -1;
}

constexpr int SECTION_SIZE   = 16;
constexpr int SECTION_VOXELS = SECTION_SIZE * SECTION_SIZE * SECTION_SIZE;
constexpr int CHUNK_SIZE     = 32;
constexpr int CHUNK_SECTIONS = CHUNK_SIZE / SECTION_SIZE;
constexpr int CHUNK_SHIFT    = 5;
constexpr int MAX_LOD_LEVELS = 12;

inline uint32_t section_index(int x, int y, int z) {
    return (uint32_t)((y << 8) | (z << 4) | x);
}

inline int32_t floor_shift(int32_t v, int s) { return v >> s; }

struct NodeKey {
    uint8_t level = 0;
    int32_t x = 0, y = 0, z = 0;
    bool operator==(const NodeKey& o) const { return level == o.level && x == o.x && y == o.y && z == o.z; }
    bool operator!=(const NodeKey& o) const { return !(*this == o); }
};
constexpr int32_t NODE_AXIS_OFFSET = 1 << 19;
inline uint64_t pack_node(const NodeKey& k) {
    return ((uint64_t)(k.level & 0xFu) << 60)
         | (((uint64_t)(uint32_t)(k.x + NODE_AXIS_OFFSET) & 0xFFFFFull) << 40)
         | (((uint64_t)(uint32_t)(k.y + NODE_AXIS_OFFSET) & 0xFFFFFull) << 20)
         |  ((uint64_t)(uint32_t)(k.z + NODE_AXIS_OFFSET) & 0xFFFFFull);
}
inline NodeKey unpack_node(uint64_t p) {
    NodeKey k;
    k.level = (uint8_t)((p >> 60) & 0xFu);
    k.x = (int32_t)((p >> 40) & 0xFFFFFull) - NODE_AXIS_OFFSET;
    k.y = (int32_t)((p >> 20) & 0xFFFFFull) - NODE_AXIS_OFFSET;
    k.z = (int32_t)( p        & 0xFFFFFull) - NODE_AXIS_OFFSET;
    return k;
}
constexpr uint64_t INVALID_NODE_KEY = ~0ull;

inline NodeKey child_key(const NodeKey& k, int i) {
    return NodeKey{ (uint8_t)(k.level - 1), k.x * 2 + (i & 1), k.y * 2 + ((i >> 1) & 1), k.z * 2 + ((i >> 2) & 1) };
}
inline NodeKey parent_key(const NodeKey& k) {
    return NodeKey{ (uint8_t)(k.level + 1), floor_shift(k.x, 1), floor_shift(k.y, 1), floor_shift(k.z, 1) };
}
inline bool is_ancestor(const NodeKey& a, const NodeKey& b) {
    if (a.level <= b.level) return false;
    const int d = a.level - b.level;
    return floor_shift(b.x, d) == a.x && floor_shift(b.y, d) == a.y && floor_shift(b.z, d) == a.z;
}
inline int64_t node_size_blocks(int level) { return (int64_t)CHUNK_SIZE << level; }
inline void node_origin_blocks(const NodeKey& k, int64_t out[3]) {
    const int64_t s = node_size_blocks(k.level);
    out[0] = (int64_t)k.x * s; out[1] = (int64_t)k.y * s; out[2] = (int64_t)k.z * s;
}

inline uint64_t pack_xz(int32_t x, int32_t z) { return ((uint64_t)(uint32_t)x << 32) | (uint64_t)(uint32_t)z; }
inline int32_t  unpack_x(uint64_t k) { return (int32_t)(uint32_t)(k >> 32); }
inline int32_t  unpack_z(uint64_t k) { return (int32_t)(uint32_t)(k & 0xFFFFFFFFull); }

inline uint64_t hash_u64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
struct U64Hash { size_t operator()(uint64_t v) const noexcept { return (size_t)hash_u64(v); } };

struct Vec3f {
    float x = 0, y = 0, z = 0;
    Vec3f() = default;
    Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3f operator+(const Vec3f& o) const { return { x + o.x, y + o.y, z + o.z }; }
    Vec3f operator-(const Vec3f& o) const { return { x - o.x, y - o.y, z - o.z }; }
    Vec3f operator*(float s)        const { return { x * s, y * s, z * s }; }
};
inline Vec3f cross(const Vec3f& a, const Vec3f& b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}
inline float dot(const Vec3f& a, const Vec3f& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3f normalize(const Vec3f& v) {
    const float l2 = dot(v, v);
    if (l2 <= 0.0f) return {};
    const float inv = 1.0f / std::sqrt(l2);
    return v * inv;
}

inline uint32_t pack_normal_oct16(const Vec3f& n) {
    const float inv = 1.0f / (std::fabs(n.x) + std::fabs(n.y) + std::fabs(n.z) + 1e-20f);
    float ox = n.x * inv;
    float oy = n.y * inv;
    if (n.z < 0.0f) {
        const float x = (1.0f - std::fabs(oy)) * (ox >= 0.0f ? 1.0f : -1.0f);
        const float y = (1.0f - std::fabs(ox)) * (oy >= 0.0f ? 1.0f : -1.0f);
        ox = x; oy = y;
    }
    auto q = [](float v) -> uint32_t {
        v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
        return (uint32_t)(uint16_t)(int16_t)(int)(v * 32767.0f);
    };
    return q(ox) | (q(oy) << 16);
}

inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return (uint16_t)(sign | half);
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    uint32_t half = ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
    return (uint16_t)(sign | half);
}

}
