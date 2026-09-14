# Planet integration notes

The planet module uses the renderer's device and scene origin. CPU code keeps
world positions in `DVec3`; GPU instance translations are the FP32 cast of the
anchor relative to the current scene origin.

## Queues and lifetime

Copy uploads complete before BLAS builds, and the graphics queue waits for the
compute fence before ray dispatch. Resources remain allocated until every
in-flight TLAS that references them has retired.

## Acceleration structures

`TlasBuilder` assembles one camera-relative TLAS from scene instances, resident
terrain chunks, and registered external instances. A chunk built during a frame
becomes eligible for the next TLAS build.

## Shading

Terrain hit data is reconstructed from the ray-hit position. The shader's
heightmap expression, planet radius, and displacement parameters must match
`HeightmapProcedural` and the CPU tessellator. Atmosphere remains a screen-space
pass and does not contribute TLAS instances.

## Verification

Build the `Pathtracer` target and run `PlanetTests`. Check that camera motion
keeps resident geometry visible while chunks stream and that queue fences cover
all copy, BLAS, and TLAS work.
