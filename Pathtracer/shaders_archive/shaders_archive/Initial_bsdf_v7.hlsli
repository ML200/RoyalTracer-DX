// NOT USED RIGHT NOW
static const uint B_dir_init    = 12;   // float3
static const uint B_x2_init     = 12;   // float3
static const uint B_n2_init     = 12;   // float3
static const uint B_obj_init    =  4;   // uint
static const uint B_mat_init    =  4;   // uint
static const uint B_pdfB_init   =  4;   // float  (pdf_bsdf)
static const uint B_pdfN_init   =  4;   // float  (pdf_nee)

static const uint P_dir_init    = 0;
static const uint P_x2_init     = P_dir_init  + B_dir_init;
static const uint P_n2_init     = P_x2_init   + B_x2_init;
static const uint P_obj_init    = P_n2_init   + B_n2_init;
static const uint P_mat_init    = P_obj_init  + B_obj_init;
static const uint P_pdfB_init   = P_mat_init  + B_mat_init;
static const uint P_pdfN_init   = P_pdfB_init + B_pdfB_init;

#define PIXEL_COUNT (DispatchRaysDimensions().x * DispatchRaysDimensions().y)

struct InitialBSDFRay {
    float3 direction;
    float3 x2;
    float3 n2;
    uint   objID;
    uint   matID;
    float  pdf_bsdf;
    float  pdf_nee;
};

// direction
float3 load_dir_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_dir_init * PIXEL_COUNT + pixelIdx * B_dir_init;
    return asfloat(buffer.Load3(addr));
}
void store_dir_init(float3 v, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_dir_init * PIXEL_COUNT + pixelIdx * B_dir_init;
    buffer.Store3(addr, asuint(v));
}

// x2
float3 load_x2_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_x2_init * PIXEL_COUNT + pixelIdx * B_x2_init;
    return asfloat(buffer.Load3(addr));
}
void store_x2_init(float3 v, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_x2_init * PIXEL_COUNT + pixelIdx * B_x2_init;
    buffer.Store3(addr, asuint(v));
}

// n2
float3 load_n2_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_n2_init * PIXEL_COUNT + pixelIdx * B_n2_init;
    return asfloat(buffer.Load3(addr));
}
void store_n2_init(float3 v, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_n2_init * PIXEL_COUNT + pixelIdx * B_n2_init;
    buffer.Store3(addr, asuint(v));
}

// objID
uint load_objID_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_obj_init * PIXEL_COUNT + pixelIdx * B_obj_init;
    return buffer.Load(addr);
}
void store_objID_init(uint id, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_obj_init * PIXEL_COUNT + pixelIdx * B_obj_init;
    buffer.Store(addr, id);
}

// matID
uint load_matID_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_mat_init * PIXEL_COUNT + pixelIdx * B_mat_init;
    return buffer.Load(addr);
}
void store_matID_init(uint id, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_mat_init * PIXEL_COUNT + pixelIdx * B_mat_init;
    buffer.Store(addr, id);
}

// pdfs
float load_pdfB_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_pdfB_init * PIXEL_COUNT + pixelIdx * B_pdfB_init;
    return asfloat(buffer.Load(addr));
}
void store_pdfB_init(float pdf, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_pdfB_init * PIXEL_COUNT + pixelIdx * B_pdfB_init;
    buffer.Store(addr, asuint(pdf));
}
float load_pdfN_init(RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_pdfN_init * PIXEL_COUNT + pixelIdx * B_pdfN_init;
    return asfloat(buffer.Load(addr));
}
void store_pdfN_init(float pdf, RWByteAddressBuffer buffer, uint pixelIdx)
{
    uint addr = P_pdfN_init * PIXEL_COUNT + pixelIdx * B_pdfN_init;
    buffer.Store(addr, asuint(pdf));
}

// complete load
InitialBSDFRay loadInitialBSDFRay(RWByteAddressBuffer buffer, uint pixelIdx)
{
    InitialBSDFRay r;
    r.direction  = load_dir_init (buffer, pixelIdx);
    r.x2         = load_x2_init  (buffer, pixelIdx);
    r.n2         = load_n2_init  (buffer, pixelIdx);
    r.objID      = load_objID_init(buffer, pixelIdx);
    r.matID      = load_matID_init(buffer, pixelIdx);
    r.pdf_bsdf   = load_pdfB_init(buffer, pixelIdx);
    r.pdf_nee    = load_pdfN_init(buffer, pixelIdx);
    return r;
}
