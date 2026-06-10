#pragma once

#include <cstdint>

#include "core/field.h"

namespace pb {

//====================================
//ColorPreviewField mirrors PreviewField but holds six GL_RGBA32F textures
//instead of GL_R32F. CUDA-OpenGL interop is identical otherwise. Used by
//the viewer to display Field<float4> surface_color as a real RGB tint
//rather than scalarised-then-viridised on a single channel.
//====================================

class ColorPreviewField {
public:
    explicit ColorPreviewField(int face_resolution);
    ~ColorPreviewField();

    ColorPreviewField(const ColorPreviewField&)            = delete;
    ColorPreviewField& operator=(const ColorPreviewField&) = delete;

    //Copy the interior cells of a Field<float4> (R,G,B,A per cell) into the
    //per-face GL textures. The grid resolution must match the constructor.
    void fill_from(const Field<float4>& src);

    uint32_t gl_texture(int face) const { return gl_textures_[face]; }
    int      resolution() const         { return n_; }

private:
    int      n_                = 0;
    uint32_t gl_textures_[6]   = {};
    void*    resources_[6]     = {};
};

}
