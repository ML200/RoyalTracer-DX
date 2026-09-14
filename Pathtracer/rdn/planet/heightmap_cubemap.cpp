#include "heightmap_cubemap.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include "../../src/Util/stb_image.h"

namespace planet {

constexpr float kHeightExaggeration = 1.0f;

namespace {

constexpr double kPi = 3.14159265358979323846;

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

}

bool HeightmapCubemap::load(const std::filesystem::path& terrain_dir) {
    m_resolution = 0;
    m_faces.clear();

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

    return true;
}

float HeightmapCubemap::sample(const DVec3& dir, uint8_t) const {
    if (m_resolution == 0) return 0.0f;

    uint8_t face = 0;
    double  u = 0.0, v = 0.0;
    sphere_to_equiangular(dir, face, u, v);

    const double n  = static_cast<double>(m_resolution);
    const double fi = (u + 1.0) * 0.5 * n - 0.5;
    const double fj = (v + 1.0) * 0.5 * n - 0.5;

    const float km = bilinear_face(m_faces[face], m_resolution, fi, fj);
    return km * 1000.0f;
}

namespace {

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

}

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
    downsample_rgba8(src, m_normal_resolution, out, dst_resolution);
    return true;
}

bool HeightmapCubemap::downsample_face_km(uint8_t face_idx,
                                          uint32_t dst_resolution,
                                          float* out) const {
    if (face_idx >= 6 || !out || dst_resolution == 0) return false;
    if (m_resolution == 0) return false;
    if (m_resolution % dst_resolution != 0) return false;

    const uint32_t src = m_resolution;
    const uint32_t dst = dst_resolution;
    const uint32_t step = src / dst;
    const float    invSamples = 1.0f / static_cast<float>(step * step);
    const auto&    face_data = m_faces[face_idx];

    for (uint32_t j = 0; j < dst; ++j) {
        for (uint32_t i = 0; i < dst; ++i) {
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

}
