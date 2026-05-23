#pragma once

#include <cstdint>

namespace pb {

//====================================
//PreviewField owns six GL_R32F textures, one per cubed-sphere face,
//each registered with CUDA for write via cudaGraphicsGLRegisterImage.
//Kernels fill the textures via surface objects; the viewer samples
//them as plain GL textures in the fragment shader.
//====================================

class PreviewField {
public:
    explicit PreviewField(int face_resolution);
    ~PreviewField();

    PreviewField(const PreviewField&) = delete;
    PreviewField& operator=(const PreviewField&) = delete;

    void fill_test_pattern(float frequency, float warp);

    uint32_t gl_texture(int face) const { return gl_textures_[face]; }
    int      resolution() const         { return n_; }

private:
    int      n_                = 0;
    uint32_t gl_textures_[6]   = {};
    void*    resources_[6]     = {};
};

}
