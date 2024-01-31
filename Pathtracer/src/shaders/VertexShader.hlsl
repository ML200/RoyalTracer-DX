cbuffer ModelViewProjectionCB : register(b0)
{
    matrix MVP;
};

struct VertexPosColor
{
    float3 Position : POSITION;
    float3 Color    : COLOR;
};

struct VertexShaderOutput
{
    float4 Color    : COLOR;
    float4 Position : SV_Position;
};

VertexShaderOutput main(VertexPosColor IN)
{
    VertexShaderOutput OUT;

    // Transform the vertex position to clip space
    OUT.Position = mul(MVP, float4(IN.Position, 1.0f));

    // Hardcoded light position
    float3 lightPosition = float3(0.0, 10.0, 0.0); // Example position

    // Calculate vector from vertex to light
    float3 lightDir = lightPosition - IN.Position;

    // Normalize the light direction
    lightDir = normalize(lightDir);

    // Simple diffuse lighting calculation
    float diff = max(dot(lightDir, float3(0, 1, 0)), 0.0); // Assuming vertex normal pointing up

    // Hardcoded light color
    float3 lightColor = float3(1.0, 1.0, 1.0); // White light

    // Calculate final color with simple diffuse component
    float3 finalColor = diff * lightColor;

    // Apply the light effect to the vertex color
    OUT.Color = float4(IN.Color * finalColor, 1.0);

    return OUT;
}
