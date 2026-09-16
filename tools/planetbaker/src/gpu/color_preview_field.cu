#include "gpu/color_preview_field.h"
#include "gpu/cuda_check.h"
#include "core/cubed_sphere.h"

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cassert>

namespace pb {

//====================================
//RGBA copy kernel. Writes float4 per interior cell as one surf2Dwrite into
//the GL_RGBA32F surface. Pixel pitch is sizeof(float4) = 16 bytes.
//====================================
__global__ void copy_color_rgba_kernel(cudaSurfaceObject_t surf,
                                        const float4* __restrict__ src,
                                        int n, int stride, int halo, int face) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    int idx = face * stride * stride + (j + halo) * stride + (i + halo);
    surf2Dwrite(src[idx], surf, i * static_cast<int>(sizeof(float4)), j);
}

ColorPreviewField::ColorPreviewField(int face_resolution) : n_(face_resolution) {
    for (int f = 0; f < 6; ++f) {
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, n_, n_, 0, GL_RGBA, GL_FLOAT, nullptr);
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

ColorPreviewField::~ColorPreviewField() {
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

void ColorPreviewField::fill_from(const Field<float4>& src) {
    assert(src.grid().n == n_ && "ColorPreviewField resolution must match field");
    const int n      = n_;
    const int halo   = CubedSphereGrid::HALO;
    const int stride = src.grid().stride();
    const float4* d  = src.data();

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
        dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        copy_color_rgba_kernel<<<grid, block>>>(surf, d, n, stride, halo, f);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaDestroySurfaceObject(surf));
    }

    CUDA_CHECK(cudaGraphicsUnmapResources(6, res));
    CUDA_CHECK(cudaDeviceSynchronize());
}

}
