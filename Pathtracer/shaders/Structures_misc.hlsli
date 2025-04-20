struct STriVertex {
    float3 vertex;
    float4 normal;
};

cbuffer Colors : register(b0) {
    float3 A;
    float3 B;
    float3 C;
}

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

struct Material
{
    float4 Kd;
    float3 Ks; float Ni;
    float3 Ke; float pad0;
    float4 Pr_Pm_Ps_Pc;
    float LUT[16];
};

// Ray payloads for closest hit and shadow ray
struct [[raypayload]] HitInfo {
    float3 hitPosition : read(caller)
                         : write(anyhit,closesthit,miss);
    uint materialID : read(caller)
                         : write(anyhit,closesthit,miss);
    float3 hitNormal : read(caller)
                         : write(anyhit,closesthit,miss);
    float area: read(caller)
                         : write(anyhit,closesthit,miss);
    uint objID: read(caller)
                         : write(anyhit,closesthit,miss);
};

struct [[raypayload]] ShadowHitInfo {
    bool isHit: read(caller)
                         : write(anyhit,closesthit,miss);
};