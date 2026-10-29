using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace RoyalTracer
{
    // HDRP's sky lights the tracer: each frame the sky cubemap HDRP lights the scene with (without the sun disk, which
    // the tracer adds itself) is unwrapped into a latitude-longitude map shared with the tracer, in place of the
    // tracer's own atmosphere for everything that sees the sky or is lit by it.
    sealed class SkyTracker
    {
        const int Width = 1024, Height = 512;
        const int FormatRGBA16F = 10; // DXGI_FORMAT_R16G16B16A16_FLOAT

        Material material;
        RenderTexture target;
        Texture2D shared;
        int id = -1;
        bool failed;

        // Before the frame's render event: the sky of this frame for the tracer, or its own atmosphere.
        public void Record(CommandBuffer cmd, bool use)
        {
            if (!use || !Ensure())
            {
                Native.RT_SetSky(-1, 0f);
                return;
            }
            CoreUtils.SetRenderTarget(cmd, target, ClearFlag.None);
            CoreUtils.DrawFullScreen(cmd, material);
            cmd.CopyTexture(target, shared);
            Native.RT_SetSky(id, RoyalTracerSystem.UnitScale);
        }

        bool Ensure()
        {
            if (id >= 0)
                return true;
            if (failed)
                return false;
            var shader = Resources.Load<Shader>("RoyalTracer/SkyUnwrap");
            var texture = IntPtr.Zero;
            id = shader ? Native.RT_CreateHostTexture(Width, Height, FormatRGBA16F, out texture) : -1;
            if (id < 0)
            {
                failed = true;
                Debug.LogWarning("[RoyalTracer] The tracer keeps its own sky: " + (shader ? Native.LastError : "the sky shader is missing"));
                return false;
            }
            shared = Texture2D.CreateExternalTexture(Width, Height, TextureFormat.RGBAHalf, false, true, texture);
            target = new RenderTexture(Width, Height, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear) { name = "RoyalTracer sky" };
            material = CoreUtils.CreateEngineMaterial(shader);
            material.SetVector("_RtSkySize", new Vector4(Width, Height, 1f / Width, 1f / Height));
            return true;
        }

        public void Dispose()
        {
            if (id >= 0)
            {
                Native.RT_SetSky(-1, 0f);
                Native.RT_ReleaseHostTexture(id);
            }
            id = -1;
            CoreUtils.Destroy(shared);
            CoreUtils.Destroy(target);
            CoreUtils.Destroy(material);
            shared = null;
            target = null;
            material = null;
            failed = false;
        }
    }
}
