//====================================
//PLANET - CUBEMAP HEIGHTMAP
//====================================

#include "heightmap_cubemap.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

//stb_image: header-only PNG decoder. The renderer's ThirdPartyImpl.cpp
//already defines STB_IMAGE_IMPLEMENTATION in one TU, so we only need the
//declarations here.
#include "../../src/Util/stb_image.h"

namespace planet {

//====================================
//Heightmap exaggeration — multiplies every cell of m_faces on load, AFTER the
//calibration shift. Both the CPU mesh path (HeightmapCubemap::sample feeding
//the tessellator) and the GPU shader path (downsample_face_km feeding the
//R32F upload, then HLSL TerrainHeight) read m_faces directly, so this scalar
//exaggerates both in lockstep with no shader changes. 1.0f = unmodified bake.
//Set to e.g. 5.0f or 10.0f to inspect near-camera detail that is invisible at
//realistic Earth-scale relief (Earth: ~5 km on a 6371 km radius = 0.08%
//geometric variation, which reads as flat from anywhere above mountain
//height). Change this constant and rebuild; the value is baked into m_faces
//at load and into the GPU R32F texture at upload.
constexpr float kHeightExaggeration = 1.0f;

namespace {

constexpr double kPi = 3.14159265358979323846;

//Minimal manifest parse - we only need the int value of "resolution". The
//manifest is small + we control its format so a hand-rolled find is enough
//to avoid pulling nlohmann into the planet module just for this.
bool find_int_field(const std::string& text, const std::string& key, uint32_t& out) {
    const std::string needle = std::string("\"") + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')) ++pos;
    int64_t v = 0;
    bool   any = false;
    while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        v = v * 10 + (text[pos] - '0');
        ++pos;
        any = true;
    }
    if (!any) return false;
    if (v < 0 || v > 0x7FFFFFFF) return false;
    out = static_cast<uint32_t>(v);
    return true;
}

//Equiangular cubed-sphere inverse - mirror of planetbaker's
//`sphere_to_face_uv` (tools/planetbaker/src/core/cubed_sphere.h). u, v are
//returned in [-1, +1]; the same face order as FACE_A in cube_sphere.cpp
//(+X -X +Y -Y +Z -Z).
void sphere_to_equiangular(const DVec3& p,
                           uint8_t& face, double& u, double& v) {
    const double ax = std::fabs(p.x);
    const double ay = std::fabs(p.y);
    const double az = std::fabs(p.z);
    double ut = 0.0, vt = 0.0;
    if (ax >= ay && ax >= az) {
        if (p.x > 0.0) { face = 0; ut = -p.z / ax; vt = -p.y / ax; }
        else           { face = 1; ut =  p.z / ax; vt = -p.y / ax; }
    } else if (ay >= ax && ay >= az) {
        if (p.y > 0.0) { face = 2; ut =  p.x / ay; vt =  p.z / ay; }
        else           { face = 3; ut =  p.x / ay; vt = -p.z / ay; }
    } else {
        if (p.z > 0.0) { face = 4; ut =  p.x / az; vt = -p.y / az; }
        else           { face = 5; ut = -p.x / az; vt = -p.y / az; }
    }
    u = std::atan(ut) * (4.0 / kPi);
    v = std::atan(vt) * (4.0 / kPi);
}

//Bilinear sample on a single face plane. fi, fj are pixel-space fractional
//coordinates in [-0.5, N-0.5]; samples outside the face are clamped.
float bilinear_face(const std::vector<float>& face_data,
                    uint32_t resolution,
                    double fi, double fj) {
    const int n  = static_cast<int>(resolution);
    auto clampi = [n](int x) { return x < 0 ? 0 : (x > n - 1 ? n - 1 : x); };

    int   i0 = static_cast<int>(std::floor(fi));
    int   j0 = static_cast<int>(std::floor(fj));
    int   i1 = i0 + 1;
    int   j1 = j0 + 1;
    double ti = fi - i0;
    double tj = fj - j0;
    i0 = clampi(i0); i1 = clampi(i1);
    j0 = clampi(j0); j1 = clampi(j1);

    const float v00 = face_data[static_cast<size_t>(j0) * n + i0];
    const float v10 = face_data[static_cast<size_t>(j0) * n + i1];
    const float v01 = face_data[static_cast<size_t>(j1) * n + i0];
    const float v11 = face_data[static_cast<size_t>(j1) * n + i1];

    const double a  = (1.0 - ti) * v00 + ti * v10;
    const double b  = (1.0 - ti) * v01 + ti * v11;
    return static_cast<float>((1.0 - tj) * a + tj * b);
}

} // namespace

bool HeightmapCubemap::load(const std::filesystem::path& terrain_dir) {
    m_resolution = 0;
    m_faces.clear();

    //----- resolution lookup -----
    //Resolution is derived from the on-disk size of elevation_face0.r32:
    //a square float32 face is res^2 * 4 bytes, so res = sqrt(filesize / 4).
    //We deliberately ignore the manifest's "resolution" field even when
    //present - the v8 manifest carries one resolution PER LAYER (elevation
    //at 8192, cloud_offset at 256, etc), and a naive substring search will
    //hit cloud_offset first (alphabetical) and silently bind a tiny grid
    //to the heightmap pipeline.
    uint32_t res = 0;
    {
        const auto probe_path = terrain_dir / "elevation_face0.r32";
        std::ifstream ifs(probe_path, std::ios::binary | std::ios::ate);
        if (!ifs) {
            std::fprintf(stderr,
                "[planet] heightmap: %s not present - cannot determine "
                "resolution\n",
                probe_path.string().c_str());
            return false;
        }
        const std::streamoff bytes = ifs.tellg();
        if (bytes <= 0 || bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
            std::fprintf(stderr,
                "[planet] heightmap: %s size %lld is not a multiple of 4 bytes\n",
                probe_path.string().c_str(), static_cast<long long>(bytes));
            return false;
        }
        const std::uint64_t floats = static_cast<std::uint64_t>(bytes) / sizeof(float);
        //Integer-sqrt with bounds check.
        std::uint64_t r = 1;
        while (r * r < floats && r <= 32768u) ++r;
        if (r * r != floats) {
            std::fprintf(stderr,
                "[planet] heightmap: %s size %lld bytes -> %llu floats is not a square\n",
                probe_path.string().c_str(),
                static_cast<long long>(bytes),
                static_cast<unsigned long long>(floats));
            return false;
        }
        res = static_cast<uint32_t>(r);
        std::fprintf(stdout,
            "[planet] heightmap: resolution %u inferred from %s file size\n",
            res, probe_path.string().c_str());
    }

    if (res > 32768u) {
        std::fprintf(stderr, "[planet] heightmap: refusing resolution %u "
                             "(over 32768 cap; check elevation_face0.r32 size)\n", res);
        return false;
    }

    //----- 6 face files -----
    const size_t expected_bytes = static_cast<size_t>(res) * res * sizeof(float);
    m_faces.assign(6, std::vector<float>{});
    for (int f = 0; f < 6; ++f) {
        char fname[64];
        std::snprintf(fname, sizeof(fname), "elevation_face%d.r32", f);
        const auto fp = terrain_dir / fname;
        std::ifstream ifs(fp, std::ios::binary);
        if (!ifs) {
            std::fprintf(stderr, "[planet] heightmap: cannot open %s\n", fp.string().c_str());
            m_faces.clear();
            return false;
        }
        m_faces[f].resize(static_cast<size_t>(res) * res);
        ifs.read(reinterpret_cast<char*>(m_faces[f].data()),
                 static_cast<std::streamsize>(expected_bytes));
        if (static_cast<size_t>(ifs.gcount()) != expected_bytes) {
            std::fprintf(stderr, "[planet] heightmap: %s short read "
                                 "(got %lld, want %zu)\n",
                         fp.string().c_str(),
                         static_cast<long long>(ifs.gcount()),
                         expected_bytes);
            m_faces.clear();
            return false;
        }
    }

    m_resolution = res;
    std::fprintf(stdout, "[planet] heightmap: loaded %u^2 x 6 faces from %s\n",
                 res, terrain_dir.string().c_str());

    //(No global-min calibration shift. An earlier version subtracted the
    //heightmap's global minimum from every cell so the deepest point sat
    //exactly on the analytic surface, but once impact craters joined the
    //bake the global min became dominated by crater-basin floors - a
    //single deep basin could pull the WHOLE planet up by 6-10 km. The
    //atmosphere / cloud / sun-shadow "clamp to ATMOS_BOTTOM_RADIUS"
    //comments in Clouds_v8.hlsli still describe a 2%-transmittance error
    //where the mesh dips below Rb, which is a much smaller defect than
    //inflating every continent by basin depth. So we feed raw signed
    //elevation through to both the CPU mesh path and the GPU upload.)

    //Optional vertical exaggeration. Scales every cell uniformly so peaks
    //and valleys grow / shrink in lockstep, signed values preserved. Useful
    //for inspecting whether "flat" renders are caused by missing displacement
    //or just by realistic Earth-scale relief being invisible at the camera's
    //altitude.
    if (kHeightExaggeration != 1.0f) {
        for (int f = 0; f < 6; ++f) {
            for (float& v : m_faces[f]) {
                v *= kHeightExaggeration;
            }
        }
        std::fprintf(stdout,
            "[planet] heightmap: exaggerated by x%.3f (every cell scaled)\n",
            static_cast<double>(kHeightExaggeration));
    }

    //Diagnostic: per-face min/max. Negative values are expected (crater
    //basin floors, plus the natural signed-elevation output of the bake).
    //The metres column is what the renderer feeds the mesh as the radial
    //displacement above planet.radius.
    for (int f = 0; f < 6; ++f) {
        float vmin = +std::numeric_limits<float>::infinity();
        float vmax = -std::numeric_limits<float>::infinity();
        bool  any_nan = false;
        for (float v : m_faces[f]) {
            if (std::isnan(v)) { any_nan = true; continue; }
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
        }
        std::fprintf(stdout,
            "[planet] heightmap: face %d range [%.4f, %.4f] km "
            "([%.1f, %.1f] m on planet)%s\n",
            f, vmin, vmax,
            static_cast<double>(vmin) * 1000.0,
            static_cast<double>(vmax) * 1000.0,
            any_nan ? "  WARN: contains NaN" : "");
    }

    //Probe samples at the 6 face centres. sample() goes through the same
    //sphere_to_equiangular + bilinear path the tessellator uses, so a 0.0 m
    //here means the mesh is being built without displacement regardless of
    //whether the disk data looks healthy above.
    const DVec3 probes[6] = {
        { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0},
        { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1},
    };
    for (int p = 0; p < 6; ++p) {
        const float h_m = sample(probes[p], 0);
        std::fprintf(stdout,
            "[planet] heightmap: probe dir=%+g,%+g,%+g -> sample = %.2f m\n",
            probes[p].x, probes[p].y, probes[p].z,
            static_cast<double>(h_m));
    }

    //====================================
    //COMPANION LAYERS (baker v8)
    //====================================
    //Best-effort: any missing layer is silently dropped. The renderer's
    //null-SRV fallback path keeps the legacy look intact in that case.

    auto load_png_rgb = [&](const char* fname_pattern,
                            std::vector<std::vector<std::uint8_t>>& dst_faces,
                            uint32_t& dst_resolution,
                            const char* label) {
        dst_faces.assign(6, std::vector<std::uint8_t>{});
        int common_w = 0, common_h = 0;
        for (int f = 0; f < 6; ++f) {
            char fname[64];
            std::snprintf(fname, sizeof(fname), fname_pattern, f);
            const auto fp = terrain_dir / fname;
            int w = 0, h = 0, n = 0;
            stbi_uc* data = stbi_load(fp.string().c_str(), &w, &h, &n, 4);
            if (!data) {
                if (f == 0) {
                    //First face missing -> treat the whole layer as not
                    //present, no warning (it's just an older bake).
                    dst_faces.clear();
                    return false;
                }
                std::fprintf(stderr,
                    "[planet] %s: face %d failed to load (%s)\n",
                    label, f, stbi_failure_reason() ? stbi_failure_reason() : "?");
                dst_faces.clear();
                return false;
            }
            if (w != h || w <= 0) {
                std::fprintf(stderr,
                    "[planet] %s: face %d not square (%d x %d) - skipping layer\n",
                    label, f, w, h);
                stbi_image_free(data);
                dst_faces.clear();
                return false;
            }
            if (f == 0) {
                common_w = w; common_h = h;
            } else if (w != common_w || h != common_h) {
                std::fprintf(stderr,
                    "[planet] %s: face %d size %dx%d != face 0 %dx%d - skipping layer\n",
                    label, f, w, h, common_w, common_h);
                stbi_image_free(data);
                dst_faces.clear();
                return false;
            }
            dst_faces[f].assign(data, data + (static_cast<size_t>(w) * h * 4));
            stbi_image_free(data);
        }
        dst_resolution = static_cast<uint32_t>(common_w);
        std::fprintf(stdout,
            "[planet] %s: loaded %u^2 x 6 faces (RGBA8) from %s\n",
            label, dst_resolution, terrain_dir.string().c_str());
        return true;
    };

    load_png_rgb("surface_color_face%d.png",
                 m_surface_color_faces, m_surface_color_resolution,
                 "surface_color");
    load_png_rgb("normal_face%d.png",
                 m_normal_faces, m_normal_resolution,
                 "normal");

    //Cloud offset: raw float32 km, fixed 256x256 in the v8 bake (manifest
    //carries the resolution; we read it from the file size for resilience).
    {
        m_cloud_offset_faces.assign(6, std::vector<float>{});
        bool ok = true;
        uint32_t cloud_res = 0;
        for (int f = 0; f < 6 && ok; ++f) {
            char fname[64];
            std::snprintf(fname, sizeof(fname), "cloud_offset_face%d.r32", f);
            const auto fp = terrain_dir / fname;
            std::ifstream ifs(fp, std::ios::binary | std::ios::ate);
            if (!ifs) { ok = false; break; }
            const std::streamsize bytes = ifs.tellg();
            if (bytes <= 0 || bytes % sizeof(float) != 0) { ok = false; break; }
            const size_t total = static_cast<size_t>(bytes) / sizeof(float);
            //Square: side length = sqrt(total).
            uint32_t side = 0;
            for (uint32_t s = 1; s <= 8192; ++s) {
                if (static_cast<size_t>(s) * s == total) { side = s; break; }
            }
            if (side == 0) { ok = false; break; }
            if (f == 0) cloud_res = side;
            else if (side != cloud_res) { ok = false; break; }
            ifs.seekg(0);
            m_cloud_offset_faces[f].resize(total);
            ifs.read(reinterpret_cast<char*>(m_cloud_offset_faces[f].data()),
                     static_cast<std::streamsize>(bytes));
            if (ifs.gcount() != bytes) { ok = false; break; }
        }
        if (ok) {
            m_cloud_offset_resolution = cloud_res;
            std::fprintf(stdout,
                "[planet] cloud_offset: loaded %u^2 x 6 faces (float32 km) from %s\n",
                cloud_res, terrain_dir.string().c_str());
        } else {
            m_cloud_offset_faces.clear();
        }
    }

    return true;
}

float HeightmapCubemap::sample(const DVec3& dir, uint8_t /*lod*/) const {
    if (m_resolution == 0) return 0.0f;

    uint8_t face = 0;
    double  u = 0.0, v = 0.0;
    sphere_to_equiangular(dir, face, u, v);

    //(u, v) in [-1, +1] -> pixel-space fractional coords. Cell centres at
    //(i + 0.5, j + 0.5), so u = -1 -> fi = -0.5, u = +1 -> fi = N - 0.5.
    const double n  = static_cast<double>(m_resolution);
    const double fi = (u + 1.0) * 0.5 * n - 0.5;
    const double fj = (v + 1.0) * 0.5 * n - 0.5;

    const float km = bilinear_face(m_faces[face], m_resolution, fi, fj);
    return km * 1000.0f;   // km -> m, IHeightmapSource contract
}

namespace {

//Box-average downsample for an RGBA8 face. src and dst must both be
//resolution^2 * 4 bytes; src_res must be a positive integer multiple of
//dst_res. The output A channel is forced to 255.
void downsample_rgba8(const std::uint8_t* src, uint32_t src_res,
                       std::uint8_t* dst, uint32_t dst_res) {
    const uint32_t step = src_res / dst_res;
    const uint32_t samples = step * step;
    for (uint32_t j = 0; j < dst_res; ++j) {
        for (uint32_t i = 0; i < dst_res; ++i) {
            uint32_t r = 0, g = 0, b = 0;
            for (uint32_t dj = 0; dj < step; ++dj) {
                const uint32_t sj = j * step + dj;
                const size_t   row = static_cast<size_t>(sj) * src_res * 4;
                for (uint32_t di = 0; di < step; ++di) {
                    const size_t off = row + (i * step + di) * 4;
                    r += src[off + 0];
                    g += src[off + 1];
                    b += src[off + 2];
                }
            }
            const size_t doff = (static_cast<size_t>(j) * dst_res + i) * 4;
            dst[doff + 0] = static_cast<std::uint8_t>(r / samples);
            dst[doff + 1] = static_cast<std::uint8_t>(g / samples);
            dst[doff + 2] = static_cast<std::uint8_t>(b / samples);
            dst[doff + 3] = 255u;
        }
    }
}

} // namespace

bool HeightmapCubemap::surface_color_face(uint8_t face_idx,
                                          uint32_t dst_resolution,
                                          std::uint8_t* out) const {
    if (face_idx >= 6 || !out || dst_resolution == 0) return false;
    if (m_surface_color_resolution == 0) return false;
    if (m_surface_color_resolution % dst_resolution != 0) return false;

    const std::uint8_t* src = m_surface_color_faces[face_idx].data();
    if (dst_resolution == m_surface_color_resolution) {
        std::memcpy(out, src,
                    static_cast<size_t>(dst_resolution) * dst_resolution * 4);
        return true;
    }
    downsample_rgba8(src, m_surface_color_resolution, out, dst_resolution);
    return true;
}

bool HeightmapCubemap::normal_face(uint8_t face_idx,
                                   uint32_t dst_resolution,
                                   std::uint8_t* out) const {
    if (face_idx >= 6 || !out || dst_resolution == 0) return false;
    if (m_normal_resolution == 0) return false;
    if (m_normal_resolution % dst_resolution != 0) return false;

    const std::uint8_t* src = m_normal_faces[face_idx].data();
    if (dst_resolution == m_normal_resolution) {
        std::memcpy(out, src,
                    static_cast<size_t>(dst_resolution) * dst_resolution * 4);
        return true;
    }
    //Box-averaging tangent-space normals is technically wrong (it doesn't
    //preserve unit length), but for "far-detail" downsampling the bias is
    //negligible - the renormalise in the shader brings everything back to
    //the unit sphere. Cheaper than full quaternion / vector averaging.
    downsample_rgba8(src, m_normal_resolution, out, dst_resolution);
    return true;
}

bool HeightmapCubemap::cloud_offset_face(uint8_t face_idx, float* out) const {
    if (face_idx >= 6 || !out) return false;
    if (m_cloud_offset_resolution == 0) return false;
    const auto& face = m_cloud_offset_faces[face_idx];
    std::memcpy(out, face.data(), face.size() * sizeof(float));
    return true;
}

bool HeightmapCubemap::downsample_face_km(uint8_t face_idx,
                                          uint32_t dst_resolution,
                                          float* out) const {
    if (face_idx >= 6 || !out || dst_resolution == 0) return false;
    if (m_resolution == 0) return false;
    if (m_resolution % dst_resolution != 0) return false;        // require integer factor

    const uint32_t src = m_resolution;
    const uint32_t dst = dst_resolution;
    const uint32_t step = src / dst;
    const float    invSamples = 1.0f / static_cast<float>(step * step);
    const auto&    face_data = m_faces[face_idx];

    for (uint32_t j = 0; j < dst; ++j) {
        for (uint32_t i = 0; i < dst; ++i) {
            //box average over the step x step source pixels covering this
            //destination texel.
            double acc = 0.0;
            for (uint32_t dj = 0; dj < step; ++dj) {
                const uint32_t sj = j * step + dj;
                const size_t   row = static_cast<size_t>(sj) * src;
                for (uint32_t di = 0; di < step; ++di) {
                    acc += face_data[row + i * step + di];
                }
            }
            out[static_cast<size_t>(j) * dst + i] =
                static_cast<float>(acc) * invSamples;
        }
    }
    return true;
}

} // namespace planet
