#pragma once

// rs_flags bits, shared by the host and the shaders.
#define RS_FLAG_NO_MESH_LIGHTS 0x80u
#define RS_FLAG_CLAMP_EMITTERS 0x100u
#define RS_FLAG_FORCE_DIFFUSE 0x20000u
// A host composites the image and applies its own fog and aerial perspective.
#define RS_FLAG_HOST_COMPOSITE 0x100000u
