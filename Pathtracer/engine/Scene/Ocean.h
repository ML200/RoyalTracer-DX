#pragma once
#include "../../rdn/Renderer.h"
#include "../../rdn/ocean/OceanCommon.h"

class Ocean {
  public:
    using Params = ocean::Params;

    // Call from SceneDefinition::Init, before the scene buffers are built.
    void Init(const Params& params, Renderer& renderer);

    // Mean water line, world units; valid before the bake.
    static float SurfaceLevel(const Params& params) { return (float)ocean::PredictSurfaceLevel(params); }
    float SurfaceLevel() const { return SurfaceLevel(m_params); }

  private:
    Params m_params;
};
