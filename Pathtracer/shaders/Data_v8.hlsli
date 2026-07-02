//====================================
//VERTEX AND ATTRIBUTES
//====================================
struct STriVertex {
    float3 vertex;
    uint   packedNormal;
    half2  texCoord;
};

struct Attributes {
    float2 bary;
};

//====================================
//INSTANCE PROPERTIES
//====================================
//224B record (was 416B with six float4x4). Transforms are affine float3x4
//(the dropped 4th ROW was 0,0,0,1): mul(M, float4(p,1)) computes the same
//4-term dot per component as the float4x4 path, so transforms stay
//bit-identical while each fetch shrinks 64B -> 48B. Hot fields (current
//transforms + index/material bases) pack into the first 176B; the ONE prev
//matrix any shader reads (prevObjectToWorld, Camera_Ray reprojection) sits
//cold at the tail. prevObjectToWorldInverse/Normal were never read by any
//pass and now exist only on the CPU working struct.
//MUST mirror InstanceProperties in Common.h field-for-field.
struct InstanceProperties
{
    float3x4 objectToWorld;
    float3x4 objectToWorldInverse;
    float3x4 objectToWorldNormal;
    uint  indexBase;
    uint  vertexBase;
    uint  materialBase;
    uint  triToLightBase;
    uint  opaqueTriCount;
    uint  _pad[3];
    float3x4 prevObjectToWorld;
};

//====================================
//LIGHT TRIANGLE
//====================================
struct LightTriangle {
    float3 x;
    float cdf;
    float3 y;
    uint instanceID;
    float3 z;
    float weight;
    float3 emission;
    uint triCount;
    float total_weight;
    float3 pad0;
};

//====================================
//PACKED MATERIAL
//====================================
//48B AoS, one material per cache line, layout in src/Components/Vertex.h.
//Field count MUST equal MaterialPack::kMatPackedU32 (Vertex.h) and the SRV
//StructureByteStride in Renderer_Pipeline.cpp.
struct MatPacked {
    uint Kd_rgb;
    uint w_Ni;
    uint PrPmPsPc;
    uint Tf_rgb;
    uint Pcr_Aniso_Rot_AlphaTh;
    uint texIDs_01;
    uint texIDs_2;
    uint uv_albedo;
    uint uv_normal;
    uint uv_rma;
    uint sss_albedo;    //RGB9E5 single-scattering albedo (inside colour)
    uint sss_radius_g;  //half scatter distance | half phase g << 16
};

//====================================
//LIGHT TREE NODES
//====================================
//64B, 4x Load4 per node. Build-stat fields (primCount, sumPower, sumPowerSq,
//itemFirst, itemCount) were never read by any shader and only inflated
//descent register pressure, so they were removed. C++ struct in LightTree.h
//mirrors this layout.
struct LightTLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float sinTheta_o;   //precomputed at build, saves one sqrt per importance call

    uint   firstChild;
    uint   childCount;
    uint   blasIndex;
    uint   _pad;                         //keeps struct stride 16B aligned
};

//64B, same motivation as the TLAS variant
struct LightBLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float sinTheta_o;   //precomputed at build, saves one sqrt per importance call

    uint   firstChild;
    uint   childCount;
    uint   triFirst;
    uint   triCount;
};

struct BlasRangeGpu {
    uint nodeOffset;
    uint nodeCount;
    uint triIndexOffset;
    uint triIndexCount;
    float4x4 worldToLocal;
};


struct LT_Sample { uint id; float pdf; };
struct LT_Path_Sample { float3 dir; float pdf; uint tri;};
