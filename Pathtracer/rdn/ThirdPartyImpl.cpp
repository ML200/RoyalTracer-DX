//====================================
//THIRD-PARTY HEADER-ONLY IMPL
//====================================
//compiles header-only libs exactly once for the project

#include "stdafx.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../src/Util/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../src/Util/stb_image_resize2.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../lib/tiny_obj_loader.h"

#define TINYGLTF3_IMPLEMENTATION
#include "../lib/tiny_gltf_v3.h"

//TinyEXR: reuse stb_image's zlib decoder (already in this TU via
//STB_IMAGE_IMPLEMENTATION above) instead of pulling in miniz. The defines
//MUST match the ones used wherever else tinyexr.h is included (currently
//Renderer_Pipeline.cpp) so the extern declarations agree.
#define TINYEXR_USE_MINIZ    0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "../lib/tinyexr/tinyexr.h"
