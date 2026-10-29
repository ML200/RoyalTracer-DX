// Writes RoyalTracer's image into HDRP's camera colour after opaque lighting (RoyalTracerSystem.RenderCamera).
// A pixel is replaced only where the tracer's unjittered primary hit is the surface HDRP rasterised there; sky,
// and surfaces the tracer does not know, keep HDRP's lighting.
Shader "Hidden/RoyalTracer/Composite"
{
    HLSLINCLUDE

    #pragma vertex Vert
    #pragma target 4.5
    #pragma only_renderers d3d11

    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipeline/RenderPass/CustomPass/CustomPassCommon.hlsl"

    TEXTURE2D(_RtColor); // pre-exposed tracer radiance; alpha = log2 of that exposure
    TEXTURE2D(_RtDepth); // view depth (m) of the tracer's primary hit, 0 = miss
    float _RtInvUnitScale;
    float _RtShowMask;

    float4 FullScreenPass(Varyings varyings) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(varyings);
        const uint2 px = uint2(varyings.positionCS.xy);
        const float depth = LoadCameraDepth(px);
        PositionInputs posInput = GetPositionInput(varyings.positionCS.xy, _ScreenSize.zw, depth, UNITY_MATRIX_I_VP, UNITY_MATRIX_V);

        // HDRP keeps its buffers bottom row first; the tracer writes the top row first.
        const uint2 rtPx = uint2(px.x, (uint)_ScreenSize.y - 1 - px.y);
        const float rtDepth = LOAD_TEXTURE2D(_RtDepth, rtPx).r;
        if (_RtShowMask > 1.5) // raw tracer image wherever it hit something
        {
            if (rtDepth <= 0)
                discard;
            const float4 raw = LOAD_TEXTURE2D(_RtColor, rtPx);
            return float4(raw.rgb * exp2(-raw.a) * _RtInvUnitScale * GetCurrentExposureMultiplier(), 1);
        }
        const bool sameSurface = depth != UNITY_RAW_FAR_CLIP_VALUE && rtDepth > 0 &&
                                 abs(posInput.linearDepth - rtDepth) <= max(0.02, 0.005 * posInput.linearDepth);
        if (!sameSurface)
            discard;
        const float4 rt = LOAD_TEXTURE2D(_RtColor, rtPx);
        const float3 radiance = rt.rgb * exp2(-rt.a) * _RtInvUnitScale;
        if (_RtShowMask > 0.5)
            return float4(radiance * GetCurrentExposureMultiplier() * float3(0.35, 1, 0.35), 1);
        return float4(radiance * GetCurrentExposureMultiplier(), 1);
    }

    ENDHLSL

    SubShader
    {
        Tags{ "RenderPipeline" = "HDRenderPipeline" }
        Pass
        {
            Name "Composite"
            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment FullScreenPass
            ENDHLSL
        }
    }
    Fallback Off
}
