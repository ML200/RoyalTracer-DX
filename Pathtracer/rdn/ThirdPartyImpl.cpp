#include "stdafx.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../src/Util/stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../src/Util/stb_image_resize2.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "../lib/tiny_obj_loader.h"

#define TINYGLTF3_IMPLEMENTATION
#include "../lib/tiny_gltf_v3.h"

// Must match Renderer_Pipeline.cpp.
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_IMPLEMENTATION
#include "../lib/tinyexr/tinyexr.h"
