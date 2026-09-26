#include "Includes_v8.hlsli"

// Never invoked; the pipeline just needs a miss shader.
[shader("miss")]
void Miss(inout TracePayload payload)
{
}
