// Writes RoyalTracer's image into HDRP's camera colour before transparents (RoyalTracerSystem.RenderCamera), with
// HDRP's fog, as HDRP's own water lighting does at that point. A pixel is replaced only where the tracer's unjittered
// primary hit is the surface HDRP rasterised there; sky, and surfaces the tracer does not know, keep HDRP's lighting.
// The tracer's ocean stands in for HDRP's water, which HDRP does not draw while path tracing: it shows wherever HDRP
// sees past it, and its depth goes into HDRP's buffer for the transparents that follow.
Shader "Hidden/RoyalTracer/Composite"
{
    HLSLINCLUDE

    #pragma vertex Vert
    #pragma target 4.5
    #pragma only_renderers d3d11

    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipeline/RenderPass/CustomPass/CustomPassCommon.hlsl"
    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/Lighting/AtmosphericScattering/AtmosphericScattering.hlsl"

    TEXTURE2D(_RtColor); // pre-exposed tracer radiance; alpha = log2 of that exposure
    TEXTURE2D(_RtDepth); // view depth (m) of the tracer's primary hit, 0 = miss, negative on its ocean
    float _RtInvUnitScale;
    float _RtShowMask;

    struct CompositeOutput
    {
        float4 color : SV_Target;
        float depth : SV_Depth;
    };

    CompositeOutput FullScreenPass(Varyings varyings)
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(varyings);
        const uint2 px = uint2(varyings.positionCS.xy);
        const float depth = LoadCameraDepth(px);
        PositionInputs posInput = GetPositionInput(varyings.positionCS.xy, _ScreenSize.zw, depth, UNITY_MATRIX_I_VP, UNITY_MATRIX_V);

        // HDRP keeps its buffers bottom row first; the tracer writes the top row first.
        const uint2 rtPx = uint2(px.x, (uint)_ScreenSize.y - 1 - px.y);
        const float rtRaw = LOAD_TEXTURE2D(_RtDepth, rtPx).r;
        const float rtDepth = abs(rtRaw);
        CompositeOutput o;
        o.depth = depth;
        if (_RtShowMask > 1.5) // raw tracer image wherever it hit something
        {
            if (rtDepth <= 0)
                discard;
            const float4 raw = LOAD_TEXTURE2D(_RtColor, rtPx);
            o.color = float4(raw.rgb * exp2(-raw.a) * _RtInvUnitScale * GetCurrentExposureMultiplier(), 1);
            return o;
        }
        const float tolerance = max(0.02, 0.005 * posInput.linearDepth);
        const bool sameSurface = depth != UNITY_RAW_FAR_CLIP_VALUE && rtDepth > 0 &&
                                 abs(posInput.linearDepth - rtDepth) <= tolerance;
        const bool water = rtRaw < 0 && (depth == UNITY_RAW_FAR_CLIP_VALUE || posInput.linearDepth > rtDepth - tolerance);
        if (!sameSurface && !water)
            discard;
        if (water) // LinearEyeDepth inverted
            o.depth = (1.0 / rtDepth - _ZBufferParams.w) / _ZBufferParams.z;
        const float4 rt = LOAD_TEXTURE2D(_RtColor, rtPx);
        float3 exposed = rt.rgb * exp2(-rt.a) * _RtInvUnitScale * GetCurrentExposureMultiplier();
        // HDRP's auto exposure averages this image into its history: one non-finite pixel would ruin every frame after.
        if (any(!isfinite(exposed)))
            discard;
        // HDRP's fog over the surface written here (the opaque fog before this pass saw HDRP's pixels, not these).
        const PositionInputs surface = GetPositionInput(varyings.positionCS.xy, _ScreenSize.zw, o.depth, UNITY_MATRIX_I_VP, UNITY_MATRIX_V);
        float3 fogColor, fogOpacity;
        EvaluateAtmosphericScattering(surface, GetWorldSpaceNormalizeViewDir(surface.positionWS), fogColor, fogOpacity);
        exposed = exposed * (1 - fogOpacity) + fogColor;
        o.color = float4(_RtShowMask > 0.5 ? exposed * float3(0.35, 1, 0.35) : exposed, 1);
        return o;
    }

    // HDRP's exposure multiplier, read back for emitters given as screen values (unlit particles).
    float4 ExposurePass(Varyings varyings) : SV_Target
    {
        return GetCurrentExposureMultiplier();
    }

    ENDHLSL

    SubShader
    {
        Tags{ "RenderPipeline" = "HDRenderPipeline" }
        Pass
        {
            Name "Composite"
            ZWrite On
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment FullScreenPass
            ENDHLSL
        }
        Pass
        {
            Name "Exposure"
            ZWrite Off
            ZTest Always
            Blend Off
            Cull Off

            HLSLPROGRAM
                #pragma fragment ExposurePass
            ENDHLSL
        }
    }
    Fallback Off
}
