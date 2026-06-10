#include "rock_scatter.h"
#include "../../include/procedural_terrain.h"   // pt_fbm: seat rocks on the mesh's procedural micro-relief

#include <cmath>
#include <unordered_map>

namespace planet {

namespace {

//====================================
//Deterministic hashing + value noise
//====================================
inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
inline uint32_t hash3i(int x, int y, int z, uint32_t seed) {
    uint32_t h = seed;
    h = hash_u32(h ^ ((uint32_t)x * 0x27d4eb2du));
    h = hash_u32(h ^ ((uint32_t)y * 0x165667b1u));
    h = hash_u32(h ^ ((uint32_t)z * 0x9e3779b9u));
    return h;
}
inline float u01(uint32_t h) { return (h & 0x00FFFFFFu) * (1.0f / 16777216.0f); }

//Smooth 3D value noise in [0,1].
inline float vnoise(double x, double y, double z, uint32_t seed) {
    const int xi = (int)std::floor(x), yi = (int)std::floor(y), zi = (int)std::floor(z);
    const double fx = x - xi, fy = y - yi, fz = z - zi;
    auto sm = [](double t) { return t * t * (3.0 - 2.0 * t); };
    const double sx = sm(fx), sy = sm(fy), sz = sm(fz);
    auto cval = [&](int dx, int dy, int dz) {
        return (double)u01(hash3i(xi + dx, yi + dy, zi + dz, seed));
    };
    const double c00 = cval(0,0,0) + sx * (cval(1,0,0) - cval(0,0,0));
    const double c10 = cval(0,1,0) + sx * (cval(1,1,0) - cval(0,1,0));
    const double c01 = cval(0,0,1) + sx * (cval(1,0,1) - cval(0,0,1));
    const double c11 = cval(0,1,1) + sx * (cval(1,1,1) - cval(0,1,1));
    const double c0  = c00 + sy * (c10 - c00);
    const double c1  = c01 + sy * (c11 - c01);
    return (float)(c0 + sz * (c1 - c0));
}

//====================================
//Icosphere
//====================================
struct IcoBuilder {
    std::vector<DVec3>                    pos;   // unit directions
    std::unordered_map<uint64_t, uint32_t> mid;

    uint32_t add(const DVec3& p) { pos.push_back(normalize(p)); return (uint32_t)pos.size() - 1; }
    uint32_t midpoint(uint32_t a, uint32_t b) {
        const uint64_t key = (a < b) ? ((uint64_t)a << 32 | b) : ((uint64_t)b << 32 | a);
        auto it = mid.find(key);
        if (it != mid.end()) return it->second;
        const uint32_t i = add(pos[a] + pos[b]);
        mid.emplace(key, i);
        return i;
    }
};

void build_icosphere(int subdiv, std::vector<DVec3>& outDir, std::vector<uint32_t>& outIdx) {
    const double t = (1.0 + std::sqrt(5.0)) * 0.5;
    IcoBuilder b;
    const DVec3 base[12] = {
        {-1, t, 0}, { 1, t, 0}, {-1,-t, 0}, { 1,-t, 0},
        { 0,-1, t}, { 0, 1, t}, { 0,-1,-t}, { 0, 1,-t},
        { t, 0,-1}, { t, 0, 1}, {-t, 0,-1}, {-t, 0, 1}
    };
    for (const auto& p : base) b.add(p);

    static const int faces[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
        {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
        {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}
    };
    std::vector<uint32_t> idx;
    idx.reserve(60);
    for (const auto& f : faces) { idx.push_back(f[0]); idx.push_back(f[1]); idx.push_back(f[2]); }

    for (int s = 0; s < subdiv; ++s) {
        std::vector<uint32_t> ni;
        ni.reserve(idx.size() * 4);
        for (size_t i = 0; i < idx.size(); i += 3) {
            const uint32_t a = idx[i], bb = idx[i+1], c = idx[i+2];
            const uint32_t ab = b.midpoint(a, bb), bc = b.midpoint(bb, c), ca = b.midpoint(c, a);
            const uint32_t tri[12] = { a,ab,ca,  bb,bc,ab,  c,ca,bc,  ab,bc,ca };
            ni.insert(ni.end(), tri, tri + 12);
        }
        idx.swap(ni);
    }
    outDir = std::move(b.pos);
    outIdx = std::move(idx);
}

} // namespace

//====================================
//Variant generation
//====================================
std::vector<RockMesh> generate_rock_variants(int count, int subdiv, uint32_t seed) {
    if (count  < 1) count  = 1;
    if (subdiv < 0) subdiv = 0;

    std::vector<DVec3>    dir;
    std::vector<uint32_t> idx;
    build_icosphere(subdiv, dir, idx);

    std::vector<RockMesh> out(count);
    for (int v = 0; v < count; ++v) {
        const uint32_t vseed = hash_u32(seed + 0x100u * (uint32_t)(v + 1));
        RockMesh& m = out[v];
        m.indices  = idx;
        m.vertices.resize(dir.size());

        //per-variant anisotropic squash so boulders aren't round
        const DVec3 squash{ 0.75 + 0.50 * (double)u01(hash_u32(vseed + 1)),
                            0.55 + 0.40 * (double)u01(hash_u32(vseed + 2)),
                            0.75 + 0.50 * (double)u01(hash_u32(vseed + 3)) };

        std::vector<DVec3> wpos(dir.size());
        for (size_t i = 0; i < dir.size(); ++i) {
            const DVec3 d = dir[i];
            //multi-octave value-noise radius -> lumpy ~[0.6, 1.4]
            double n = 0.0, amp = 0.5, frq = 1.6;
            for (int o = 0; o < 4; ++o) {
                n += amp * ((double)vnoise(d.x * frq + 13.1, d.y * frq + 5.7, d.z * frq + 9.3, vseed) - 0.5);
                amp *= 0.5; frq *= 2.07;
            }
            double r = 1.0 + n;
            if (r < 0.45) r = 0.45;
            const DVec3 sq{ d.x * squash.x, d.y * squash.y, d.z * squash.z };
            wpos[i] = normalize(sq) * r;
        }

        //smooth normals from accumulated face normals of the displaced mesh
        std::vector<DVec3> nrm(dir.size(), DVec3{ 0,0,0 });
        for (size_t i = 0; i < idx.size(); i += 3) {
            const uint32_t a = idx[i], bb = idx[i+1], c = idx[i+2];
            const DVec3 fn = cross(wpos[bb] - wpos[a], wpos[c] - wpos[a]);
            nrm[a] = nrm[a] + fn; nrm[bb] = nrm[bb] + fn; nrm[c] = nrm[c] + fn;
        }
        for (size_t i = 0; i < dir.size(); ++i) {
            const DVec3 p  = wpos[i];
            const DVec3 nn = normalize(nrm[i]);
            const DVec3 pn = normalize(p);
            const double yy = pn.y < -1.0 ? -1.0 : (pn.y > 1.0 ? 1.0 : pn.y);
            m.vertices[i].position = Vec3f{ (float)p.x,  (float)p.y,  (float)p.z  };
            m.vertices[i].normal   = Vec3f{ (float)nn.x, (float)nn.y, (float)nn.z };
            m.vertices[i].u = (float)(0.5 + std::atan2(p.z, p.x) * 0.15915494309); // 1/2pi
            m.vertices[i].v = (float)(0.5 - std::asin(yy)        * 0.31830988618); // 1/pi
        }
    }
    return out;
}

//====================================
//Camera-following scatter
//====================================
void RockScatter::configure(const RockScatterConfig& cfg, int variant_count) {
    m_cfg          = cfg;
    m_variantCount = variant_count < 1 ? 1 : variant_count;
    m_have         = false;
    m_live.clear();
}

bool RockScatter::update(const DVec3& camera_world, const IRockHeight& height) {
    const DVec3 off  = camera_world - m_cfg.planet.center;
    const DVec3 gdir = normalize(off);
    if (length_sq(gdir) <= 0.0) return false;

    if (m_have) {
        double cosang = dot(gdir, m_lastGroundDir);
        if (cosang >  1.0) cosang =  1.0;
        if (cosang < -1.0) cosang = -1.0;
        const double moved = std::acos(cosang) * m_cfg.planet.radius;  // surface arc length
        if (moved < m_cfg.retrigger_move_m) return false;
    }
    rebuild(gdir, height);
    m_lastGroundDir = gdir;
    m_have          = true;
    return true;
}

void RockScatter::rebuild(const DVec3& gdir, const IRockHeight& height) {
    m_live.clear();
    const PlanetGeometry& pl = m_cfg.planet;

    //tangent frame at the camera ground point
    const DVec3 upAxis = (std::fabs(gdir.y) < 0.95) ? DVec3{ 0,1,0 } : DVec3{ 1,0,0 };
    const DVec3 tA     = normalize(cross(upAxis, gdir));
    const DVec3 tB     = cross(gdir, tA);
    const DVec3 ground = pl.center + gdir * pl.radius;

    const int    N  = (int)std::ceil(m_cfg.region_radius_m / m_cfg.cell_size_m);
    const double cs = m_cfg.cell_size_m;
    const double r2 = m_cfg.region_radius_m * m_cfg.region_radius_m;

    uint32_t slot = 0;
    for (int j = -N; j <= N && slot < m_cfg.max_rocks; ++j) {
        for (int i = -N; i <= N && slot < m_cfg.max_rocks; ++i) {
            const double lx = i * cs, ly = j * cs;
            if (lx * lx + ly * ly > r2) continue;

            //exact surface point of this cell, then a WORLD-anchored cell index
            //so the same ground always grows the same rock (no swimming).
            const DVec3 tanPos = ground + tA * lx + tB * ly;
            const DVec3 cdir   = normalize(tanPos - pl.center);
            const DVec3 surf   = pl.center + cdir * pl.radius;
            const int cx = (int)std::lround(surf.x / cs);
            const int cy = (int)std::lround(surf.y / cs);
            const int cz = (int)std::lround(surf.z / cs);

            const uint32_t h = hash3i(cx, cy, cz, m_cfg.seed);
            if (u01(h) > m_cfg.coverage) continue;

            //jitter within the cell, snap to terrain height, orient to surface
            const float jx = u01(hash_u32(h + 1u)) - 0.5f;
            const float jy = u01(hash_u32(h + 2u)) - 0.5f;
            const DVec3 jpos = surf + tA * ((double)jx * cs) + tB * ((double)jy * cs);
            const DVec3 d    = normalize(jpos - pl.center);
            const float hm   = height.sample_height_m(d);
            //Seat on the SAME surface the tessellator builds: baked heightmap +
            //the procedural micro-relief (pt_fbm). Sampling pt here in FP64 at a
            //fine footprint keeps rocks from floating/sinking on the bumps the
            //mesh has but the heightmap alone doesn't.
            DVec3        ptGrad;
            const double hpt = pt_fbm<double>(
                DVec3{ d.x * pl.radius, d.y * pl.radius, d.z * pl.radius }, 0.5, ptGrad);
            const DVec3 anchor = pl.center + d * (pl.radius + (double)hm + hpt);

            const float    scale = m_cfg.min_scale_m
                                 + (m_cfg.max_scale_m - m_cfg.min_scale_m) * u01(hash_u32(h + 3u));
            const float    yaw   = u01(hash_u32(h + 4u)) * 6.2831853f;
            const uint32_t var   = (m_variantCount > 1)
                                 ? (hash_u32(h + 5u) % (uint32_t)m_variantCount) : 0u;

            //local +Y -> surface normal d, yaw about d, uniform scale
            const DVec3  up   = d;
            const DVec3  seed = (std::fabs(up.y) < 0.95) ? DVec3{ 0,1,0 } : DVec3{ 1,0,0 };
            const DVec3  rt0  = normalize(cross(up, seed));
            const DVec3  fw0  = cross(up, rt0);
            const double cyw  = std::cos((double)yaw), syw = std::sin((double)yaw);
            const DVec3  rt   = rt0 * cyw + fw0 * syw;
            const DVec3  fw   = fw0 * cyw - rt0 * syw;

            RockInstance inst;
            inst.anchor_world = anchor;
            inst.variant      = var;
            inst.stable_id    = slot++;
            const float sc = scale;
            //row-major 3x3, column j = image of local axis j (x->rt, y->up, z->fw)
            inst.rot_scale[0] = (float)rt.x * sc; inst.rot_scale[1] = (float)up.x * sc; inst.rot_scale[2] = (float)fw.x * sc;
            inst.rot_scale[3] = (float)rt.y * sc; inst.rot_scale[4] = (float)up.y * sc; inst.rot_scale[5] = (float)fw.y * sc;
            inst.rot_scale[6] = (float)rt.z * sc; inst.rot_scale[7] = (float)up.z * sc; inst.rot_scale[8] = (float)fw.z * sc;
            m_live.push_back(inst);
        }
    }
}

} // namespace planet
