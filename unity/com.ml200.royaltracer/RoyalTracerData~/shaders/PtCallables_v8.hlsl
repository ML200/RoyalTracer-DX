#include "Includes_v8.hlsli"
#include "OceanVolume.hlsli"
#include "PtVertex_v8.hlsli"

// Path-tracer callables: rare or heavy work kept out of Pass_pt_trace_v8 and its material shaders, so
// those stay small in code and registers. Table order: PT_CALLABLE_LIGHT_PDF, PT_CALLABLE_SSS,
// PT_CALLABLE_WATER_SEGMENT, PT_CALLABLE_NEE_WATER. Other light samples run in Pass_pt_nee_v8.

// Emitter MIS: the density with which the previous vertex's light sample picks this emitter.
[shader("callable")]
void PtLightPdf(inout PtLightPdfIO p)
{
    PtLightPdfEval(p);
}

// Subsurface random walk from the entry point to the exit.
[shader("callable")]
void PtSss(inout PtSssIO p)
{
    PtSssWalk(p);
}

// In-scatter and attenuation of a path segment inside the sea.
[shader("callable")]
void PtWaterSegment(inout PtWaterSegmentIO p)
{
    PtWaterSegmentShade(p, DispatchRaysIndex().xy);
}

// Light-tree and sun sample on the sea surface or in the water (in-water sun, sea highlight, water
// reach); it may precede the deferred vertex, so it is not queued.
[shader("callable")]
void PtNeeWater(inout PtNeeIO p)
{
    PtNeeShade(p, DispatchRaysIndex().xy);
}
