#include "Includes_v8.hlsli"

// Escaped rays are shaded by the trace raygen (PtShadeMiss in PtVertex_v8.hlsli); this miss
// shader only exists so the pipeline has one, and is never invoked.
[shader("miss")]
void Miss(inout TracePayload payload)
{
}
