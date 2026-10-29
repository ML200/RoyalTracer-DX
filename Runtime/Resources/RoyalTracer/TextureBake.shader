// Bakes HDRP material textures into the tracer's layouts (MaterialConverter.Bake). Pipeline independent: used only
// through Graphics.Blit.
Shader "Hidden/RoyalTracer/TextureBake"
{
    Properties { _MainTex ("Source", 2D) = "white" {} }

    CGINCLUDE
    #include "UnityCG.cginc"
    sampler2D _MainTex;
    float4 _ParamsA, _ParamsB;

    struct Varyings { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
    Varyings Vert(appdata_img v)
    {
        Varyings o;
        o.position = UnityObjectToClipPos(v.vertex);
        o.uv = v.texcoord;
        return o;
    }
    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // 0: albedo = base map * base colour (_ParamsA, linear); the target encodes sRGB.
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return tex2D(_MainTex, i.uv) * _ParamsA; }
            ENDCG
        }
        // 1: HDRP mask map (metallic, AO, detail, smoothness) to (AO, roughness, metallic) with HDRP's remaps:
        // _ParamsA = metallic min/max, smoothness min/max; _ParamsB.xy = AO min/max.
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target
            {
                float4 mask = tex2D(_MainTex, i.uv);
                float metallic = lerp(_ParamsA.x, _ParamsA.y, mask.r);
                float smoothness = lerp(_ParamsA.z, _ParamsA.w, mask.a);
                float ao = lerp(_ParamsB.x, _ParamsB.y, mask.g);
                return float4(ao, 1 - smoothness, metallic, 1);
            }
            ENDCG
        }
        // 2: Unity normal map (RG or AG packed) scaled by _ParamsA.x, as n * 0.5 + 0.5.
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target
            {
                float4 p = tex2D(_MainTex, i.uv);
                p.x *= p.w;
                float3 n;
                n.xy = (p.xy * 2 - 1) * _ParamsA.x;
                n.z = sqrt(1 - saturate(dot(n.xy, n.xy)));
                return float4(normalize(n) * 0.5 + 0.5, 1);
            }
            ENDCG
        }
        // 3: preview of a tracer output (pre-exposed radiance), Reinhard tone mapped; for captures.
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target
            {
                float3 c = max(tex2D(_MainTex, i.uv).rgb, 0);
                return float4(c / (1 + c), 1);
            }
            ENDCG
        }
    }
}
