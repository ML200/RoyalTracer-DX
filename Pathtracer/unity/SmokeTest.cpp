// Drives RoyalTracerUnity.dll without Unity: a ground plane, a metal box and an alpha-tested panel under a sun,
// 64 frames with a quality switch, a moved box and, at frame 44, the panel released and a new box streamed in;
// then checks the colour/depth outputs and writes a preview (smoke_preview.bmp) next to the exe.
#define NOMINMAX
#include <windows.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include "RoyalTracerApi.h"

static void __stdcall LogLine(int32_t level, const char* message) {
    if (level > RT_LOG_INFO || std::getenv("RT_SMOKE_VERBOSE"))
        std::printf("%s %s\n", level == RT_LOG_ERROR ? "[error]" : level == RT_LOG_WARNING ? "[warn]" : "[info]",
                    message);
}

struct MeshData {
    std::vector<float> p, n, uv;
    std::vector<uint32_t> idx;
    std::vector<int32_t> mat;
    void Quad(const float o[3], const float u[3], const float v[3], const float nrm[3], int32_t material) {
        const uint32_t base = (uint32_t)p.size() / 3;
        const float corners[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (auto& c : corners) {
            for (int k = 0; k < 3; ++k) {
                p.push_back(o[k] + u[k] * c[0] + v[k] * c[1]);
                n.push_back(nrm[k]);
            }
            uv.push_back(c[0]);
            uv.push_back(c[1]);
        }
        // Counter-clockwise seen from the normal side.
        idx.insert(idx.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        mat.insert(mat.end(), {material, material});
    }
    void Box(float cx, float cy, float cz, float h, int32_t material) {
        const float axes[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (int a = 0; a < 3; ++a)
            for (int s = -1; s <= 1; s += 2) {
                const float* nA = axes[a];
                const float* uA = axes[(a + 1) % 3];
                const float* vA = axes[(a + 2) % 3];
                float nrm[3], u[3], v[3], o[3];
                for (int k = 0; k < 3; ++k) {
                    nrm[k] = nA[k] * s;
                    u[k] = uA[k] * 2 * h;
                    v[k] = vA[k] * 2 * h;
                }
                if (s < 0)
                    std::swap(u, v);
                const float c[3] = {cx, cy, cz};
                for (int k = 0; k < 3; ++k)
                    o[k] = c[k] + nrm[k] * h - u[k] * 0.5f - v[k] * 0.5f;
                Quad(o, u, v, nrm, material);
            }
    }
    int32_t Add() {
        RT_MeshDesc d{p.data(), n.data(), uv.data(), (uint32_t)(p.size() / 3), idx.data(), (uint32_t)idx.size(),
                      mat.data()};
        return RT_CreateMesh(&d);
    }
};

int main() {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir(exe);
    dir.resize(dir.find_last_of(L"\\/"));
    const std::wstring data = dir + L"\\RoyalTracerData", cache = dir + L"\\RoyalTracerCache";

    setvbuf(stdout, nullptr, _IONBF, 0);
    RT_InitDesc init{data.c_str(), cache.c_str(), LogLine};
    const int32_t initResult = RT_Init(&init);
    RT_Capabilities caps{};
    RT_GetCapabilities(&caps);
    std::printf("adapter=\"%s\" vram=%lluMB sm=%x dxr=%u serReorders=%d dlssRR=%d reason=\"%s\"\n", caps.adapterName,
                (unsigned long long)(caps.dedicatedVideoMemory >> 20), caps.shaderModel, caps.raytracingTier,
                caps.serReorders, caps.dlssRayReconstruction, caps.reason);
    if (initResult != 1) {
        std::printf("FAIL init: %s\n", RT_GetLastError());
        return 1;
    }

    // Checker with transparent holes for the alpha-tested panel.
    const uint32_t T = 64;
    std::vector<uint8_t> checker(T * T * 4);
    for (uint32_t y = 0; y < T; ++y)
        for (uint32_t x = 0; x < T; ++x) {
            const bool on = ((x / 8) + (y / 8)) & 1;
            uint8_t* px = &checker[(y * T + x) * 4];
            px[0] = on ? 230 : 40;
            px[1] = on ? 120 : 40;
            px[2] = on ? 40 : 200;
            px[3] = on ? 255 : 0;
        }
    RT_TextureDesc tex{T, T, RT_TEXTURE_ALBEDO, checker.data()};
    const int32_t checkerTex = RT_CreateTexture(&tex);

    RT_MaterialDesc ground{{0.45f, 0.45f, 0.42f, 1}, {0, 0, 0}, 0.8f, 0, 0, 0, 1.5f, 1, -1, -1, -1, {1, 1}};
    RT_MaterialDesc metal{{0.95f, 0.64f, 0.54f, 1}, {0, 0, 0}, 0.25f, 1, 0, 0, 1.5f, 1, -1, -1, -1, {1, 1}};
    RT_MaterialDesc panel{{1, 1, 1, 1}, {0, 0, 0}, 0.6f, 0, 0, 0, 1.5f, 0.5f, checkerTex, -1, -1, {1, 1}};
    const int32_t mGround = RT_CreateMaterial(&ground), mMetal = RT_CreateMaterial(&metal),
                  mPanel = RT_CreateMaterial(&panel);

    MeshData plane, box, card, late;
    const float o[3] = {-100, 0, -100}, u[3] = {0, 0, 200}, v[3] = {200, 0, 0}, up[3] = {0, 1, 0};
    plane.Quad(o, u, v, up, mGround);
    box.Box(0, 1, 0, 1, mMetal);
    const float co[3] = {-2.5f, 0, 0.5f}, cu[3] = {2, 0, 0}, cv[3] = {0, 2, 0}, cn[3] = {0, 0, 1};
    card.Quad(co, cu, cv, cn, mPanel);
    const float identity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    const float moved[12] = {1, 0, 0, 2.5f, 0, 1, 0, 0, 0, 0, 1, -1};
    const int32_t cardMesh = card.Add();
    const int32_t planeInstance = RT_CreateInstance(plane.Add(), identity);
    const int32_t boxInstance = RT_CreateInstance(box.Add(), identity);
    const int32_t cardInstance = RT_CreateInstance(cardMesh, identity);
    late.Box(-1.5f, 0.75f, 2.0f, 0.75f, mMetal);
    // An emissive panel: glows when hit and joins the light tree as a streamed light slot.
    RT_MaterialDesc glowDesc{{0.1f, 0.1f, 0.1f, 1}, {20, 18, 14}, 0.5f, 0, 0, 0, 1.5f, 1, -1, -1, -1, {1, 1}};
    const int32_t mGlow = RT_CreateMaterial(&glowDesc);
    MeshData glow;
    const float go[3] = {0.6f, 1.7f, 3}, gu[3] = {0.8f, 0, 0}, gv[3] = {0, 0.8f, 0}, gn[3] = {0, 0, 1};
    glow.Quad(go, gu, gv, gn, mGlow);
    const int32_t glowInstance = RT_CreateInstance(glow.Add(), identity);
    if (glowInstance < 0) {
        std::printf("FAIL glow panel: %s\n", RT_GetLastError());
        return 1;
    }
    // A light's proxy: lights the scene, but camera and shadow rays pass through it (the host draws its lights).
    MeshData proxy;
    const float po[3] = {-0.3f, 0.2f, 4}, pu[3] = {0.6f, 0, 0}, pv[3] = {0, 0.6f, 0};
    proxy.Quad(po, pu, pv, gn, mGlow);
    const int32_t proxyInstance = RT_CreateInstance(proxy.Add(), identity);
    if (proxyInstance < 0) {
        std::printf("FAIL light proxy: %s\n", RT_GetLastError());
        return 1;
    }
    RT_SetInstanceFlags(proxyInstance, RT_INSTANCE_EMITTER);
    // Particles: a ring of emitters circling above the scene, rewritten every frame from frame 10.
    const uint32_t particleCount = 512;
    const int32_t particles = RT_CreateEmitterSet(particleCount);
    if (particles < 0) {
        std::printf("FAIL emitter set: %s\n", RT_GetLastError());
        return 1;
    }
    std::vector<RT_Emitter> emitters(particleCount);
    double emitterMs = 0;
    // A deforming quad behind the scene, its vertices rewritten every frame from frame 30 (refits).
    MeshData flag;
    const float fo[3] = {3, 0.5f, -4}, fu[3] = {1.5f, 0, 0}, fv[3] = {0, 1.5f, 0}, fn[3] = {0, 0, 1};
    flag.Quad(fo, fu, fv, fn, mPanel);
    RT_MeshDesc flagDesc{flag.p.data(), flag.n.data(), flag.uv.data(), 4, flag.idx.data(), 6, flag.mat.data(),
                         RT_MESH_DEFORMING};
    const int32_t flagMesh = RT_CreateMesh(&flagDesc);
    if (flagMesh < 0 || RT_CreateInstance(flagMesh, identity) < 0) {
        std::printf("FAIL deforming mesh: %s\n", RT_GetLastError());
        return 1;
    }
    if (checkerTex < 0 || mPanel < 0 || planeInstance < 0 || boxInstance < 0 || cardInstance < 0) {
        std::printf("FAIL scene: %s\n", RT_GetLastError());
        return 1;
    }

    const uint32_t W = 640, H = 360;
    RT_FrameDesc frame{};
    const float eye[3] = {0, 2.5f, 8}, target[3] = {0, 1, 0}, camUp[3] = {0, 1, 0};
    memcpy(frame.eye, eye, sizeof eye);
    memcpy(frame.target, target, sizeof target);
    memcpy(frame.up, camUp, sizeof camUp);
    frame.fovYDegrees = 50;
    frame.width = W;
    frame.height = H;
    frame.deltaTime = 1.0f / 60.0f;
    const float sun[3] = {0.4f, 0.6f, 0.3f};
    const float len = std::sqrt(sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]);
    for (int k = 0; k < 3; ++k)
        frame.sunDirection[k] = sun[k] / len;
    frame.sunIntensity = 6.5f;
    frame.skyIntensity = 1;
    frame.turbidity = 2;
    frame.skyGroundY = -1;

    std::vector<float> rgba(W * H * 4), depth(W * H);
    RT_FrameTicket ticket{};

    // The pixel of the panel's centre, which the late box covers from frame 44.
    auto normalize = [](float v[3]) {
        const float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (int k = 0; k < 3; ++k)
            v[k] /= l;
    };
    auto cross = [](const float a[3], const float b[3], float o[3]) {
        o[0] = a[1] * b[2] - a[2] * b[1];
        o[1] = a[2] * b[0] - a[0] * b[2];
        o[2] = a[0] * b[1] - a[1] * b[0];
    };
    float fwd[3] = {target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]}, right[3], camY[3];
    normalize(fwd);
    cross(fwd, camUp, right);
    normalize(right);
    cross(right, fwd, camY);
    const float tanHalf = std::tan(0.5f * frame.fovYDegrees * 3.14159265f / 180.0f);
    auto pixelOf = [&](float x, float y, float z) {
        const float d[3] = {x - eye[0], y - eye[1], z - eye[2]};
        const float vx = d[0] * right[0] + d[1] * right[1] + d[2] * right[2];
        const float vy = d[0] * camY[0] + d[1] * camY[1] + d[2] * camY[2];
        const float vz = d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2];
        const uint32_t px = (uint32_t)((vx / (vz * tanHalf * W / H) * 0.5f + 0.5f) * W);
        const uint32_t py = (uint32_t)((0.5f - vy / (vz * tanHalf) * 0.5f) * H);
        return py * W + px;
    };
    const uint32_t panelPixel = pixelOf(-1.5f, 1.0f, 0.5f), glowPixel = pixelOf(1.0f, 2.1f, 3.0f);
    float panelDepth = 0;
    for (int i = 0; i < 64; ++i) {
        if (i == 16 || i == 24) // a DLSS-RR input resolution change and back
            RT_SetQuality(i == 16 ? RT_QUALITY_PERFORMANCE : RT_QUALITY_DLAA);
        if (i == 32) // moves the box half way through, like a host transform update
            RT_SetInstanceTransforms(1, &boxInstance, moved);
        if (i >= 30) {
            std::vector<float> moving = flag.p;
            for (size_t k = 0; k < moving.size(); k += 3)
                moving[k + 2] += 0.3f * std::sin(0.2f * i + moving[k + 1]);
            if (RT_UpdateMeshVertices(flagMesh, moving.data(), flag.n.data()) != 1) {
                std::printf("FAIL vertex update: %s\n", RT_GetLastError());
                return 1;
            }
        }
        if (i >= 10) {
            const uint32_t alive = i % 7 == 0 ? particleCount / 3 : particleCount; // sets shrink and grow
            for (uint32_t k = 0; k < alive; ++k) {
                const float a = 6.2831853f * k / particleCount + 0.1f * i;
                emitters[k] = {{3.0f * std::cos(a), 2.6f + 0.3f * std::sin(5.0f * a), 3.0f * std::sin(a) - 1.0f},
                               0.03f, {40.0f, 18.0f, 4.0f}, 0};
            }
            LARGE_INTEGER t0, t1, f;
            QueryPerformanceCounter(&t0);
            const int32_t updated = RT_UpdateEmitterSet(particles, emitters.data(), alive);
            QueryPerformanceCounter(&t1);
            QueryPerformanceFrequency(&f);
            emitterMs += 1000.0 * (double)(t1.QuadPart - t0.QuadPart) / (double)f.QuadPart;
            if (updated != 1) {
                std::printf("FAIL emitter update: %s\n", RT_GetLastError());
                return 1;
            }
        }
        if (i == 50 || i == 56) { // emission changes rebuild the panel's light records
            RT_MaterialDesc dimmed = glowDesc;
            for (float& e : dimmed.emission)
                e *= i == 50 ? 0.25f : 1.0f;
            if (RT_UpdateMaterial(mGlow, &dimmed) != 1) {
                std::printf("FAIL material update: %s\n", RT_GetLastError());
                return 1;
            }
        }
        if (i == 44) { // the panel leaves and a box in front of it arrives, without a renderer rebuild
            RT_ReleaseInstance(cardInstance);
            RT_ReleaseMesh(cardMesh);
            if (RT_CreateInstance(late.Add(), identity) < 0) {
                std::printf("FAIL late box: %s\n", RT_GetLastError());
                return 1;
            }
        }
        if (RT_SubmitFrame(&frame, &ticket) != 1 || RT_RenderTicketForTest(ticket.index, rgba.data(), depth.data()) != 1) {
            std::printf("FAIL frame %d: %s\n", i, RT_GetLastError());
            return 1;
        }
        if (i == 43)
            panelDepth = depth[panelPixel];
    }
    const float lateDepth = depth[panelPixel];
    const float proxyDepth = depth[pixelOf(0.0f, 0.5f, 4.0f)]; // the ground behind the proxy (~5.4 m), not the proxy (~4.3 m)
    // The emitter nearest the camera in the last frame: its pixel shows what lies behind it, at least 1 m further.
    const RT_Emitter& front = emitters[particleCount / 4];
    const float frontDistance = std::sqrt((front.position[0] - eye[0]) * (front.position[0] - eye[0]) +
                                          (front.position[1] - eye[1]) * (front.position[1] - eye[1]) +
                                          (front.position[2] - eye[2]) * (front.position[2] - eye[2]));
    const float particleDepth = depth[pixelOf(front.position[0], front.position[1], front.position[2])];
    RT_ReleaseEmitterSet(particles);
    std::printf("stream: %s\n", RT_GetStats());

    // Checks: finite colour, a plausible exposure, sky has no depth, the box is hit at its distance.
    size_t nonFinite = 0;
    double luminance = 0;
    for (uint32_t i = 0; i < W * H; ++i) {
        const float* c = &rgba[i * 4];
        if (!std::isfinite(c[0]) || !std::isfinite(c[1]) || !std::isfinite(c[2]) || !std::isfinite(c[3]))
            ++nonFinite;
        else
            luminance += (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / std::exp2(c[3]);
    }
    luminance /= W * H;
    const float skyDepth = depth[(H / 20) * W + W / 2];
    const float boxDepth = depth[(H / 2) * W + W / 2 + W / 8]; // the moved box, right of centre
    const float exposure = std::exp2(rgba[((H / 2) * W + W / 2) * 4 + 3]);
    const float* g = &rgba[glowPixel * 4];
    const float glowRadiance = (0.2126f * g[0] + 0.7152f * g[1] + 0.0722f * g[2]) / std::exp2(g[3]);
    std::printf("nonFinite=%zu meanRadiance=%.4f exposure=%.4f skyDepth=%.2f boxDepth=%.2f panelDepth=%.2f "
                "lateBoxDepth=%.2f glowRadiance=%.2f proxyDepth=%.2f particleDepth=%.2f (emitter at %.2f) "
                "emitterUpdate=%.3fms\n",
                nonFinite, luminance, exposure, skyDepth, boxDepth, panelDepth, lateDepth, glowRadiance, proxyDepth,
                particleDepth, frontDistance, emitterMs / 54.0);

    // Preview: exposed and gamma encoded.
    std::vector<uint8_t> bmp(W * H * 3);
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x)
            for (int c = 0; c < 3; ++c) {
                float v = rgba[(y * W + x) * 4 + c];
                v = v / (1 + v);
                v = std::pow(std::fmin(std::fmax(v, 0.0f), 1.0f), 1 / 2.2f);
                bmp[((H - 1 - y) * W + x) * 3 + (2 - c)] = (uint8_t)(v * 255 + 0.5f);
            }
    BITMAPFILEHEADER fh{0x4D42, (DWORD)(sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + bmp.size()), 0, 0,
                        sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER)};
    BITMAPINFOHEADER ih{sizeof(BITMAPINFOHEADER), (LONG)W, (LONG)H, 1, 24, BI_RGB};
    if (FILE* f = _wfopen((dir + L"\\smoke_preview.bmp").c_str(), L"wb")) {
        fwrite(&fh, sizeof fh, 1, f);
        fwrite(&ih, sizeof ih, 1, f);
        fwrite(bmp.data(), 1, bmp.size(), f);
        fclose(f);
    }

    // Panel centre: the panel (or the ground through its holes) at >= 7.6 m, then the late box's face at ~5.5 m.
    const bool pass = nonFinite == 0 && luminance > 0 && exposure > 0 && skyDepth == 0.0f && boxDepth > 5 &&
                      boxDepth < 9 && panelDepth > 7.0f && lateDepth > 4.5f && lateDepth < 6.5f &&
                      glowRadiance > 4.0f && // emission 18 seen directly, under DLSS's emitter-spike clamp; lit surfaces < 1
                      proxyDepth > 4.8f && (particleDepth == 0.0f || particleDepth > frontDistance + 1.0f);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    RT_Shutdown();
    return pass ? 0 : 1;
}
