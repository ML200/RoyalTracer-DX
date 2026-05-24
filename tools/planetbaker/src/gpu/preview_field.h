#pragma once

#include <cstdint>

#include "core/field.h"

namespace pb {

//====================================
//PreviewField owns six GL_R32F textures, one per cubed-sphere face, each
//registered with CUDA for surface-load-store. The viewer samples them as
//plain GL textures in its fragment shader; CUDA kernels fill them either
//from a synthetic test pattern or by scalarizing a Field<T> from
//PlanetState. Texture resolution N must match the Field's interior
//resolution; fill_from asserts.
//====================================

class PreviewField {
public:
    explicit PreviewField(int face_resolution);
    ~PreviewField();

    PreviewField(const PreviewField&)            = delete;
    PreviewField& operator=(const PreviewField&) = delete;

    //Synthetic continuous-on-sphere pattern, used to debug face seams.
    void fill_test_pattern(float frequency, float warp);

    //Copy interior cells of a Field<T> into the GL textures, scalarizing as
    //needed. For vector fields, channel picks which component to display
    //(0..channels-1). For uint8 fields, scale multiplies the value before
    //writing to GL_R32F.
    void fill_from(const Field<float>&         src);
    void fill_from(const Field<float2>&        src, int channel);
    void fill_from(const Field<float4>&        src, int channel);
    void fill_from(const Field<std::uint8_t>&  src, float scale);

    uint32_t gl_texture(int face) const { return gl_textures_[face]; }
    int      resolution() const         { return n_; }

private:
    int      n_                = 0;
    uint32_t gl_textures_[6]   = {};
    void*    resources_[6]     = {};
};

}
