// HDRP's sky for RoyalTracer (SkyTracker): the sky cubemap HDRP lights the scene with, unwrapped into the tracer's
// latitude-longitude map (RT_SetSky). Row 0 is straight up; u runs along atan2(-z, x) in Unity's frame, which is
// atan2(z, x) in the tracer's right-handed one. Radiance as HDRP stores it, not exposed.
Shader "Hidden/RoyalTracer/SkyUnwrap"
{
    HLSLINCLUDE

    #pragma vertex Vert
    #pragma target 4.5
    #pragma only_renderers d3d11

    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipeline/RenderPass/CustomPass/CustomPassCommon.hlsl"

    float4 _RtSkySize; // width, height, 1 / width, 1 / height

    float4 Unwrap(Varyings varyings) : SV_Target
    {
        // The render target's own pixels, so its rows run as the tracer reads them whatever the platform flips.
        const float2 uv = varyings.positionCS.xy * _RtSkySize.zw;
        const float phi = (uv.x - 0.5) * TWO_PI, theta = uv.y * PI;
        const float3 tracer = float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
        return float4(SampleSkyTexture(float3(tracer.x, tracer.y, -tracer.z), 0, 0).rgb, 1);
    }

    ENDHLSL

    SubShader
    {
        Tags{ "RenderPipeline" = "HDRenderPipeline" }
        Pass
        {
            Name "Unwrap"
            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment Unwrap
            ENDHLSL
        }
    }
    Fallback Off
}
