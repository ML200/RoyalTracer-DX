#include "gpu/preview_field.h"
#include "gpu/cuda_check.h"
#include "core/cubed_sphere.h"

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace pb {

__device__ inline float pattern_at(float x, float y, float z, float freq, float warp) {
    float a = sinf(freq * x) * cosf(freq * y) * sinf(freq * z);
    float b = sinf(freq * 1.7f * (x + warp * y));
    float c = cosf(freq * 1.3f * (y + warp * z));
    return 0.5f + 0.4f * a + 0.1f * b * c;
}

__global__ void fill_face_kernel(cudaSurfaceObject_t surf, int n, int face,
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

void PreviewField::fill_test_pattern(float freq, float warp) {
    cudaGraphicsResource_t res[6];
    for (int f = 0; f < 6; ++f) {
        res[f] = static_cast<cudaGraphicsResource_t>(resources_[f]);
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

        dim3 block(16, 16);
        dim3 grid((n_ + block.x - 1) / block.x, (n_ + block.y - 1) / block.y);
        fill_face_kernel<<<grid, block>>>(surf, n_, f, freq, warp);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDestroySurfaceObject(surf));
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(6, res));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
