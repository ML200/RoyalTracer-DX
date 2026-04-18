struct STriVertex {
    float3 vertex;
    uint   packedNormal;
    half2  texCoord;
};

struct Attributes {
    float2 bary;
};

struct InstanceProperties
{
    float4x4 objectToWorld;
    float4x4 objectToWorldInverse;
    float4x4 prevObjectToWorld;
    float4x4 prevObjectToWorldInverse;
    float4x4 objectToWorldNormal;
    float4x4 prevObjectToWorldNormal;
    uint  indexBase;
    uint  vertexBase;
    uint  materialBase;
    uint triToLightBase;
    uint opaqueTriCount;
    uint _pad[3];
};

struct LightTriangle {
    float3 x;
    float cdf;       // 16 bytes
    float3 y;
    uint instanceID; // 16 bytes
    float3 z;
    float weight;       // 16 bytes
    float3 emission;
    uint triCount;   // 16 bytes
    float total_weight;
    float3 pad0;       // 16 bytes
};

// Material is stored as a compressed AoS struct (40 B / material, was 112 B).
// Single struct keeps one material's data in one cache line — path-tracing
// threads hit random materials, so SoA planes would force 3 separate cache
// line fetches per material. See Material_Decoder_v8.hlsli for accessors
// and src/Components/Vertex.h (MaterialSoA) for the CPU packing.
//
// Ke dropped (CPU-only; emission travels via the LightTriangle buffer).
//
//  Offset   Field                     Encoding
//  ───────  ────────────────────────  ────────────────────────────────
//   0       Kd_rgb                    RGB9E5 (32 bit)
//   4       w_Ni                      half Kd.w | half Ni
//   8       PrPmPsPc                  4× uint8 in [0,1], LSB = Pr
//  12       Tf_rgb                    RGB9E5
//  16       Pcr_Aniso_Rot_AlphaTh     u8 Pcr | i8 aniso | u8 anisoRot | u8 alphaTh
//  20       texIDs_01                 i16 albedoTexID | i16 normalTexID
//  24       texIDs_2                  i16 rmaTexID    | i16 pad
//  28       uv_albedo                 half2
//  32       uv_normal                 half2
//  36       uv_rma                    half2
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
};

struct LightTLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float cosTheta_e;

    uint   firstChild;
    uint   childCount;
    uint   blasIndex;
    uint   primCount;

    float  sumPower;
    float  sumPowerSq;

    uint   itemFirst;
    uint   itemCount;
};

struct LightBLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float cosTheta_e;

    uint   firstChild;
    uint   childCount;
    uint   triFirst;
    uint   triCount;

    uint   primCount;     uint _pad0;
    float  sumPower;      float sumPowerSq;
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