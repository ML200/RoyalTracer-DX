#include "gpu/preview_field.h"
#include "gpu/cuda_check.h"
#include "core/cubed_sphere.h"

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cassert>

namespace pb {

//====================================
//Test pattern kernel: continuous noise computed from 3D direction so the
//six face textures agree on shared edges.
//====================================

__device__ inline float pattern_at(float x, float y, float z, float freq, float warp) {
    float a = sinf(freq * x) * cosf(freq * y) * sinf(freq * z);
    float b = sinf(freq * 1.7f * (x + warp * y));
    float c = cosf(freq * 1.3f * (y + warp * z));
    return 0.5f + 0.4f * a + 0.1f * b * c;
}

__global__ void fill_pattern_kernel(cudaSurfaceObject_t surf, int n, int face,
                                    float freq, float warp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    float u = ((i + 0.5f) / n) * 2.0f - 1.0f;
    float v = ((j + 0.5f) / n) * 2.0f - 1.0f;
    Vec3f p = face_uv_to_sphere(face, u, v);
    float val = pattern_at(p.x, p.y, p.z, freq, warp);

    surf2Dwrite(val, surf, i * static_cast<int>(sizeof(float)), j);
}

//====================================
//Field copy kernels. One per source element type. Each thread reads one
//interior cell of the source face (skipping the field's halo) and writes
//one float to the GL_R32F surface.
//====================================

__global__ void copy_f1_kernel(cudaSurfaceObject_t surf,
                               const float* __restrict__ src,
                               int n, int stride, int halo, int face) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int idx = face * stride * stride + (j + halo) * stride + (i + halo);
    surf2Dwrite(src[idx], surf, i * static_cast<int>(sizeof(float)), j);
}

__global__ void copy_f2_channel_kernel(cudaSurfaceObject_t surf,
                                       const float2* __restrict__ src,
                                       int n, int stride, int halo, int face,
                                       int channel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int    idx = face * stride * stride + (j + halo) * stride + (i + halo);
    float2 v   = src[idx];
    float  s   = (channel == 0) ? v.x : v.y;
    surf2Dwrite(s, surf, i * static_cast<int>(sizeof(float)), j);
}

__global__ void copy_f4_channel_kernel(cudaSurfaceObject_t surf,
                                       const float4* __restrict__ src,
                                       int n, int stride, int halo, int face,
                                       int channel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int    idx = face * stride * stride + (j + halo) * stride + (i + halo);
    float4 v   = src[idx];
    float  s   = 0.0f;
    switch (channel) {
        case 0:  s = v.x; break;
        case 1:  s = v.y; break;
        case 2:  s = v.z; break;
        default: s = v.w; break;
    }
    surf2Dwrite(s, surf, i * static_cast<int>(sizeof(float)), j);
}

__global__ void copy_u8_kernel(cudaSurfaceObject_t surf,
                               const std::uint8_t* __restrict__ src,
                               int n, int stride, int halo, int face,
                               float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int idx = face * stride * stride + (j + halo) * stride + (i + halo);
    float v = static_cast<float>(src[idx]) * scale;
    surf2Dwrite(v, surf, i * static_cast<int>(sizeof(float)), j);
}

//====================================
//PreviewField implementation
//====================================

PreviewField::PreviewField(int face_resolution) : n_(face_resolution) {
    for (int f = 0; f < 6; ++f) {
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, n_, n_, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        gl_textures_[f] = tex;

        cudaGraphicsResource_t res = nullptr;
        CUDA_CHECK(cudaGraphicsGLRegisterImage(
            &res, tex, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsSurfaceLoadStore));
        resources_[f] = res;
    }
}

PreviewField::~PreviewField() {
    for (int f = 0; f < 6; ++f) {
        if (resources_[f]) {
            cudaGraphicsUnregisterResource(static_cast<cudaGraphicsResource_t>(resources_[f]));
            resources_[f] = nullptr;
        }
        if (gl_textures_[f]) {
            glDeleteTextures(1, &gl_textures_[f]);
            gl_textures_[f] = 0;
        }
    }
}

//====================================
//Map all 6 interop resources, hand each one's surface to a per-face callback,
//then unmap. The callback must do nothing async-unfriendly other than launch
//a kernel; cudaDeviceSynchronize is called once after unmap.
//====================================

namespace {

template <typename Fn>
void with_face_surfaces(void* (&resources)[6], Fn&& fn) {
    cudaGraphicsResource_t res[6];
    for (int f = 0; f < 6; ++f) {
        res[f] = static_cast<cudaGraphicsResource_t>(resources[f]);
    }
    CUDA_CHECK(cudaGraphicsMapResources(6, res));

    for (int f = 0; f < 6; ++f) {
        cudaArray_t arr = nullptr;
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&arr, res[f], 0, 0));

        cudaResourceDesc desc{};
        desc.resType = cudaResourceTypeArray;
        desc.res.array.array = arr;

        cudaSurfaceObject_t surf = 0;
        CUDA_CHECK(cudaCreateSurfaceObject(&surf, &desc));

        fn(f, surf);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDestroySurfaceObject(surf));
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(6, res));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}

void PreviewField::fill_test_pattern(float freq, float warp) {
    with_face_surfaces(resources_, [&](int face, cudaSurfaceObject_t surf) {
        dim3 block(16, 16);
        dim3 grid((n_ + block.x - 1) / block.x, (n_ + block.y - 1) / block.y);
        fill_pattern_kernel<<<grid, block>>>(surf, n_, face, freq, warp);
    });
}

void PreviewField::fill_from(const Field<float>& src) {
    assert(src.grid().n == n_ && "PreviewField resolution must match field");
    const int n      = n_;
    const int halo   = CubedSphereGrid::HALO;
    const int stride = src.grid().stride();
    const float* d   = src.data();

    with_face_surfaces(resources_, [&](int face, cudaSurfaceObject_t surf) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        copy_f1_kernel<<<grid, block>>>(surf, d, n, stride, halo, face);
    });
}

void PreviewField::fill_from(const Field<float2>& src, int channel) {
    assert(src.grid().n == n_ && "PreviewField resolution must match field");
    const int n      = n_;
    const int halo   = CubedSphereGrid::HALO;
    const int stride = src.grid().stride();
    const float2* d  = src.data();
    const int     c  = channel & 0x1;

    with_face_surfaces(resources_, [&](int face, cudaSurfaceObject_t surf) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        copy_f2_channel_kernel<<<grid, block>>>(surf, d, n, stride, halo, face, c);
    });
}

void PreviewField::fill_from(const Field<float4>& src, int channel) {
    assert(src.grid().n == n_ && "PreviewField resolution must match field");
    const int n      = n_;
    const int halo   = CubedSphereGrid::HALO;
    const int stride = src.grid().stride();
    const float4* d  = src.data();
    const int     c  = channel & 0x3;

    with_face_surfaces(resources_, [&](int face, cudaSurfaceObject_t surf) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        copy_f4_channel_kernel<<<grid, block>>>(surf, d, n, stride, halo, face, c);
    });
}

void PreviewField::fill_from(const Field<std::uint8_t>& src, float scale) {
    assert(src.grid().n == n_ && "PreviewField resolution must match field");
    const int n      = n_;
    const int halo   = CubedSphereGrid::HALO;
    const int stride = src.grid().stride();
    const std::uint8_t* d = src.data();

    with_face_surfaces(resources_, [&](int face, cudaSurfaceObject_t surf) {
        dim3 block(16, 16);
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        copy_u8_kernel<<<grid, block>>>(surf, d, n, stride, halo, face, scale);
    });
}

}
