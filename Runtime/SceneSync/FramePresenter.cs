using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // DLSS frame generation (player builds): the plugin presents the game's finished frames itself, through a child
    // window over the game's window, on its own device where frame generation runs. Each path traced frame hands it
    // two images, both top row first: HDRP's post-processed image before the UI (from a pass after post-processing),
    // and the finished frame with the UI (captured at the end of the frame). Frame generation interpolates the scene
    // and keeps the UI where the two differ; depth and motion come from the tracer's own frame.
    sealed class FramePresenter
    {
        const int FormatRGBA8Srgb = 29; // DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        static readonly int HudSourceId = Shader.PropertyToID("_RtHudSource");

        readonly Material composite;
        readonly CommandBuffer cmd = new() { name = "RoyalTracer present" };
        readonly HudlessPass hudlessPass;
        RenderTexture capture, hudlessTarget;
        Texture2D finished, hudless;
        int finishedId = -1, hudlessId = -1, savedVSync;
        int frames = -1, vsync = -1; // as last handed to the plugin
        bool active, hudlessThisFrame;
        string failure;

        public bool Active => active;
        public string Failure => failure;

        public FramePresenter(Material composite)
        {
            this.composite = composite;
            hudlessPass = new HudlessPass(this) { name = "RoyalTracer frame before UI" };
        }

        // Main thread, each frame: frame generation at this mode (Off: none), the window's size followed.
        public void Follow(FrameGenerationMode mode)
        {
            bool wanted = mode != FrameGenerationMode.Off;
            if (wanted && !active && failure == null)
            {
                if (HDROutputSettings.main != null && HDROutputSettings.main.active)
                    failure = "Frame generation does not run with HDR output yet.";
                else if (Native.RT_PresenterEnable(1) == 1)
                {
                    active = true;
                    savedVSync = QualitySettings.vSyncCount;
                    frames = vsync = -1;
                    CustomPassVolume.RegisterGlobalCustomPass(CustomPassInjectionPoint.AfterPostProcess, hudlessPass);
                    Debug.Log("[RoyalTracer] Frame generation on");
                }
                else
                    failure = Native.LastError;
                if (failure != null)
                    Debug.LogWarning("[RoyalTracer] Frame generation unavailable: " + failure);
            }
            else if (!wanted && active)
                Dispose();
            if (!active)
                return;
            // The plugin presents, at the game's vsync choice; Unity's own presents go to a hidden window and must not wait.
            if (QualitySettings.vSyncCount != 0)
            {
                savedVSync = QualitySettings.vSyncCount;
                QualitySettings.vSyncCount = 0;
            }
            int wantFrames = (int)mode, wantVSync = savedVSync > 0 ? 1 : 0;
            if (wantFrames != frames || wantVSync != vsync)
            {
                Native.RT_PresenterOptions(wantFrames, wantVSync);
                frames = wantFrames;
                vsync = wantVSync;
            }
            Native.RT_PresenterUpdate();
        }

        // After HDRP's post-processing of the game camera: its image before the UI, top row first.
        void RecordHudless(CustomPassContext ctx)
        {
            var camera = ctx.hdCamera.camera;
            if (!active || camera.cameraType != CameraType.Game || camera.targetTexture || !Ensure())
                return;
            ctx.cmd.SetGlobalTexture(HudSourceId, ctx.cameraColorBuffer); // a render graph target: no texture of its own
            CoreUtils.SetRenderTarget(ctx.cmd, hudlessTarget, ClearFlag.None);
            CoreUtils.DrawFullScreen(ctx.cmd, composite, ctx.propertyBlock, shaderPassId: 2);
            ctx.cmd.CopyTexture(hudlessTarget, hudless);
            hudlessThisFrame = true;
        }

        // End of a path traced frame: the finished image to the plugin, presented with the ticket's frame.
        public void Present(ulong ticket, IntPtr renderEvent)
        {
            if (!active || !Ensure())
                return;
            // The capture holds the screen top row first, as the window's swap chain does.
            ScreenCapture.CaptureScreenshotIntoRenderTexture(capture);
            cmd.Clear();
            cmd.CopyTexture(capture, finished);
            Native.RT_PresenterSource(finishedId, hudlessThisFrame ? hudlessId : -1);
            cmd.IssuePluginEventAndData(renderEvent, Native.EventPresent, (IntPtr)(long)ticket);
            Graphics.ExecuteCommandBuffer(cmd);
            hudlessThisFrame = false;
        }

        // Textures at the screen's size.
        bool Ensure()
        {
            int w = Screen.width, h = Screen.height;
            if (finished && finished.width == w && finished.height == h)
                return true;
            ReleaseTextures();
            var hudlessPtr = IntPtr.Zero;
            finishedId = Native.RT_CreateHostTexture((uint)w, (uint)h, FormatRGBA8Srgb, out var finishedPtr);
            hudlessId = finishedId < 0 ? -1 : Native.RT_CreateHostTexture((uint)w, (uint)h, FormatRGBA8Srgb, out hudlessPtr);
            if (hudlessId < 0)
            {
                failure = Native.LastError;
                Debug.LogWarning("[RoyalTracer] Frame generation unavailable: " + failure);
                Dispose();
                return false;
            }
            finished = Texture2D.CreateExternalTexture(w, h, TextureFormat.RGBA32, false, false, finishedPtr);
            hudless = Texture2D.CreateExternalTexture(w, h, TextureFormat.RGBA32, false, false, hudlessPtr);
            capture = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { name = "RoyalTracer frame" };
            hudlessTarget = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB) { name = "RoyalTracer frame before UI" };
            return true;
        }

        void ReleaseTextures()
        {
            Native.RT_PresenterSource(-1, -1);
            if (finishedId >= 0)
                Native.RT_ReleaseHostTexture(finishedId);
            if (hudlessId >= 0)
                Native.RT_ReleaseHostTexture(hudlessId);
            finishedId = hudlessId = -1;
            CoreUtils.Destroy(finished);
            CoreUtils.Destroy(hudless);
            CoreUtils.Destroy(capture);
            CoreUtils.Destroy(hudlessTarget);
            finished = hudless = null;
            capture = hudlessTarget = null;
            hudlessThisFrame = false;
        }

        public void Dispose()
        {
            if (active)
            {
                CustomPassVolume.UnregisterGlobalCustomPass(hudlessPass);
                Native.RT_PresenterEnable(0);
                QualitySettings.vSyncCount = savedVSync;
                Debug.Log("[RoyalTracer] Frame generation off");
            }
            active = false;
            ReleaseTextures();
        }

        sealed class HudlessPass : CustomPass
        {
            readonly FramePresenter presenter;
            public HudlessPass(FramePresenter presenter) => this.presenter = presenter;
            protected override void Execute(CustomPassContext ctx) => presenter.RecordHudless(ctx);
        }
    }
}
