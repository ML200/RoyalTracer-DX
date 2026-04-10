// ═══════════════════════════════════════════════════════════════════
// ThirdPartyImpl.cpp — Compile header-only library implementations
//                      exactly ONCE across the entire project.
//
// If ObjLoader.h currently defines STB_IMAGE_IMPLEMENTATION,
// TINYOBJLOADER_IMPLEMENTATION, etc., REMOVE those #defines from
// ObjLoader.h. They must only exist here.
// ═══════════════════════════════════════════════════════════════════

#include "stdafx.h"

// stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "../src/Util/stb_image.h"

// stb_image_resize
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "../src/Util/stb_image_resize2.h"

// tinyobjloader
#define TINYOBJLOADER_IMPLEMENTATION
#include "../lib/tiny_obj_loader.h"

// tg3 (glTF loader) — adjust the header name if yours differs
#define TINYGLTF3_IMPLEMENTATION
#include "../lib/tiny_gltf_v3.h"
