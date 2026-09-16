//Single translation unit that brings in the stb_image_write implementation.
//Keeps the macro out of any .cu file (NVCC compiles the implementation header
//cleanly, but isolating it here is the safer single-TU pattern stb expects).
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
