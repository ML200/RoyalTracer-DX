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
