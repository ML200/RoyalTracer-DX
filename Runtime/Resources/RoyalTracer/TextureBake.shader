// Bakes HDRP material textures into the tracer's layouts (MaterialConverter.Bake). Pipeline independent: used only
// through Graphics.Blit. Passes 4-6 read slice _Slice of _ArrayTex (texture banks such as the Ship Palette's)
// instead of _MainTex.
Shader "Hidden/RoyalTracer/TextureBake"
{
    Properties { _MainTex ("Source", 2D) = "white" {} }

    CGINCLUDE
    #include "UnityCG.cginc"
    #pragma require 2darray
    sampler2D _MainTex;
    UNITY_DECLARE_TEX2DARRAY(_ArrayTex);
    float _Slice;
    float4 _ParamsA, _ParamsB;

    struct Varyings { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };
    Varyings Vert(appdata_img v)
    {
        Varyings o;
        o.position = UnityObjectToClipPos(v.vertex);
        o.uv = v.texcoord;
        return o;
    }

    float4 SampleArray(float2 uv) { return UNITY_SAMPLE_TEX2DARRAY(_ArrayTex, float3(uv, _Slice)); }

    // Albedo = base map * base colour (_ParamsA, linear); the target encodes sRGB.
    float4 Albedo(float4 base) { return base * _ParamsA; }

    // HDRP mask map (metallic, AO, detail, smoothness) to (AO, roughness, metallic) with HDRP's remaps:
    // _ParamsA = metallic min/max, smoothness min/max; _ParamsB.xy = AO min/max.
    float4 Rma(float4 mask)
    {
        float metallic = lerp(_ParamsA.x, _ParamsA.y, mask.r);
        float smoothness = lerp(_ParamsA.z, _ParamsA.w, mask.a);
        float ao = lerp(_ParamsB.x, _ParamsB.y, mask.g);
        return float4(ao, 1 - smoothness, metallic, 1);
    }

    // Unity normal map (RG or AG packed) scaled by _ParamsA.x, as n * 0.5 + 0.5.
    float4 Normal(float4 p)
    {
        p.x *= p.w;
        float3 n;
        n.xy = (p.xy * 2 - 1) * _ParamsA.x;
        n.z = sqrt(1 - saturate(dot(n.xy, n.xy)));
        return float4(normalize(n) * 0.5 + 0.5, 1);
    }
    ENDCG

    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        // 0: albedo
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return Albedo(tex2D(_MainTex, i.uv)); }
            ENDCG
        }
        // 1: mask map to RMA
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return Rma(tex2D(_MainTex, i.uv)); }
            ENDCG
        }
        // 2: normal map
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return Normal(tex2D(_MainTex, i.uv)); }
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
        // 4: albedo from an array slice
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return Albedo(SampleArray(i.uv)); }
            ENDCG
        }
        // 5: mask map to RMA from an array slice
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return Rma(SampleArray(i.uv)); }
            ENDCG
        }
        // 6: normal map from an array slice
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            float4 Frag(Varyings i) : SV_Target { return Normal(SampleArray(i.uv)); }
            ENDCG
        }
    }
}
