#pragma once
#include "SceneManager.h"
#include "../../rdn/Renderer.h"
#include "../../rdn/ocean/OceanCommon.h"

// Scene-facing handle for the ocean, in the same shape as EmissiveCubes: fill in a Params and call
// Init from a SceneDefinition. Everything else - the spectral simulation, the level of detail, the
// acceleration structures and the shading - is driven from inside the renderer.
//
//   Ocean::Params sea;
//   sea.windSpeed = 12.0f;   // m/s at 10 m: Beaufort 6
//   sea.fetch     = 200000.0f;
//   m_ocean.Init(sea, r);
//
// The sea state is fully described by wind speed and fetch; everything else has a physical default.
class Ocean {
  public:
    using Params = ocean::Params;

    // Must be called from SceneDefinition::Init, before the renderer builds its scene buffers.
    void Init(const Params& params, Renderer& renderer);

    // Applies a new sea state. Changing wind, fetch or direction re-bakes the spectrum, which
    // takes a few milliseconds on the CPU; the rest take effect on the next frame.
    void SetParams(const Params& params, Renderer& renderer);

    const Params& GetParams() const { return m_params; }

    // Mean water line the ocean will sit at, in world units. Known before the wave field is baked,
    // so a scene can place a camera or a hull against it inside its own Init.
    static float SurfaceLevel(const Params& params) { return (float)ocean::PredictSurfaceLevel(params); }
    float SurfaceLevel() const { return SurfaceLevel(m_params); }

    // Significant wave height for this sea state: the average of the highest third of the waves,
    // which is what a mariner would call the sea height.
    static float WaveHeight(const Params& params) { return (float)ocean::PredictSignificantWaveHeight(params); }

  private:
    Params m_params;
};
