#include "mc_bake.h"
#include <algorithm>
#include <cmath>

namespace mc {

namespace {

inline float image_coord(float a, float s, int size) {
    const float t = s > 0.0f ? a : 1.0f - a;
    return t * (float)size;
}

struct Vert { float x, y; float u, v; float depth; };

inline float edge(const Vert& a, const Vert& b, float px, float py) {
    return (b.x - a.x) * (py - a.y) - (b.y - a.y) * (px - a.x);
}

inline const uint8_t* texel(const BakeTexture& t, float u, float v) {
    int tx = (int)std::floor(u * (float)t.width), ty = (int)std::floor(v * (float)t.height);
    tx = ((tx % t.width) + t.width) % t.width;
    ty = ((ty % t.height) + t.height) % t.height;
    return t.rgba + (size_t)ty * t.pitch + (size_t)tx * 4;
}

void raster_triangle(const Vert& v0, const Vert& v1, const Vert& v2, const BakeTexture& tex, BakedFace& face, std::vector<float>& zbuf) {
    const float area = edge(v0, v1, v2.x, v2.y);
    if (std::fabs(area) < 1e-8f) return;
    const float inv = 1.0f / area;
    const int size = face.size;
    const int x0 = std::max(0, (int)std::floor(std::min({ v0.x, v1.x, v2.x })));
    const int x1 = std::min(size - 1, (int)std::ceil(std::max({ v0.x, v1.x, v2.x })));
    const int y0 = std::max(0, (int)std::floor(std::min({ v0.y, v1.y, v2.y })));
    const int y1 = std::min(size - 1, (int)std::ceil(std::max({ v0.y, v1.y, v2.y })));
    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x) {
        const float px = (float)x + 0.5f, py = (float)y + 0.5f;
        float w0 = edge(v1, v2, px, py) * inv;
        float w1 = edge(v2, v0, px, py) * inv;
        float w2 = edge(v0, v1, px, py) * inv;
        if (w0 < -1e-5f || w1 < -1e-5f || w2 < -1e-5f) continue;
        const float depth = w0 * v0.depth + w1 * v1.depth + w2 * v2.depth;
        float& z = zbuf[(size_t)y * size + x];
        if (depth >= z) continue;
        const float u = w0 * v0.u + w1 * v1.u + w2 * v2.u;
        const float v = w0 * v0.v + w1 * v1.v + w2 * v2.v;
        const uint8_t* t = texel(tex, u, v);
        if (t[3] < 128) continue;
        uint8_t* dst = &face.rgba[((size_t)y * size + x) * 4];
        dst[0] = t[0]; dst[1] = t[1]; dst[2] = t[2]; dst[3] = 255;
        z = depth;
    }
}

}

void bake_block_faces(const std::vector<RawQuad>& quads, int size, const BakeTextureLookup& lookup, BakedFace out[6]) {
    size = std::max(1, size);
    std::vector<float> zbuf[6];
    for (int f = 0; f < 6; ++f) {
        out[f] = BakedFace{};
        out[f].size = size;
        out[f].rgba.assign((size_t)size * size * 4, 0);
        zbuf[f].assign((size_t)size * size, 2.0f);
    }
    for (const RawQuad& q : quads) {
        const BakeTexture tex = lookup(q);
        if (!tex.rgba || tex.width <= 0 || tex.height <= 0) continue;
        for (int f = 0; f < 6; ++f) {
            const FaceProjection& pr = FACE_PROJECTION[f];
            const int dir = FACE_DIR[f][pr.na];
            Vert v[4];
            for (int k = 0; k < 4; ++k) {
                const float p[3] = { q.pos[k].x / 16.0f, q.pos[k].y / 16.0f, q.pos[k].z / 16.0f };
                v[k].x = image_coord(p[pr.ua], pr.su, size);
                v[k].y = image_coord(p[pr.va], pr.sv, size);
                v[k].u = q.uv[k][0] / 16.0f;
                v[k].v = q.uv[k][1] / 16.0f;
                v[k].depth = dir > 0 ? 1.0f - p[pr.na] : p[pr.na];
            }
            raster_triangle(v[0], v[1], v[2], tex, out[f], zbuf[f]);
            raster_triangle(v[0], v[2], v[3], tex, out[f], zbuf[f]);
        }
    }
    for (int f = 0; f < 6; ++f) {
        size_t drawn = 0;
        float minDepth = 1.0f;
        for (size_t i = 0; i < (size_t)size * size; ++i) {
            if (out[f].rgba[i * 4 + 3] == 0) continue;
            ++drawn;
            minDepth = std::min(minDepth, zbuf[f][i]);
        }
        out[f].coverage = (float)drawn / (float)((size_t)size * size);
        out[f].minDepth = drawn ? std::max(0.0f, minDepth) : 1.0f;
    }
}

double fill_holes(std::vector<uint8_t>& rgba, int w, int h) {
    if (w <= 0 || h <= 0 || rgba.size() < (size_t)w * h * 4) return 0.0;
    const size_t n = (size_t)w * h;
    std::vector<uint8_t> opaque(n);
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) { opaque[i] = rgba[i * 4 + 3] >= 128 ? 1 : 0; count += opaque[i]; }
    const double coverage = (double)count / (double)n;
    if (count == 0) return coverage;
    std::vector<uint8_t> next(n);
    while (count < n) {
        next = opaque;
        bool grew = false;
        for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const size_t i = (size_t)y * w + x;
            if (opaque[i]) continue;
            const int nx[4] = { (x + w - 1) % w, (x + 1) % w, x, x };
            const int ny[4] = { y, y, (y + h - 1) % h, (y + 1) % h };
            int sum[3] = { 0, 0, 0 }, cnt = 0;
            for (int k = 0; k < 4; ++k) {
                const size_t j = (size_t)ny[k] * w + nx[k];
                if (!opaque[j]) continue;
                sum[0] += rgba[j * 4]; sum[1] += rgba[j * 4 + 1]; sum[2] += rgba[j * 4 + 2]; ++cnt;
            }
            if (!cnt) continue;
            rgba[i * 4] = (uint8_t)(sum[0] / cnt); rgba[i * 4 + 1] = (uint8_t)(sum[1] / cnt); rgba[i * 4 + 2] = (uint8_t)(sum[2] / cnt);
            next[i] = 1; ++count; grew = true;
        }
        opaque.swap(next);
        if (!grew) break;
    }
    for (size_t i = 0; i < n; ++i) rgba[i * 4 + 3] = 255;
    return coverage;
}

}
