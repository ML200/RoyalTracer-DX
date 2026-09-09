#ifndef DLSS_GUIDE_LAYOUT
#define DLSS_GUIDE_LAYOUT
// Shared CPU/HLSL reverse-Z camera contract. Clouds and terrain extend beyond 10 km.
// Reverse-Z R32F retains distant depths without the cancellation of forward-Z projection.
#define DLSS_GUIDE_DEPTH_NEAR 0.01f
#define DLSS_GUIDE_DEPTH_FAR 100000000.0f
// The optional diagnostic hit-distance texture is R16F; its storage range is independent.
#define DLSS_SPEC_HIT_MAX 65504.0f
#endif
