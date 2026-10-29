using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // What a game's settings screen needs: whether path tracing can run on this machine and why not, the player's
    // choice, and whether it is running. In play mode it runs while Available && UserEnabled and a camera exists;
    // otherwise HDRP renders as usual.
    public static class PathTracing
    {
        const string EnabledKey = "RoyalTracer.Enabled", QualityKey = "RoyalTracer.Quality",
            FrameGenerationKey = "RoyalTracer.FrameGeneration";

        // Any non-empty value makes this machine report unsupported, to test a game's fallback.
        public const string ForceUnsupportedVariable = "ROYALTRACER_UNSUPPORTED";

        static bool probed;
        static string reason = "", adapter = "";

        // The first query checks the GPU, which creates the renderer's device (well under a second).
        public static bool Available { get { Probe(); return reason.Length == 0; } }

        // Player-readable; empty when Available.
        public static string UnavailableReason { get { Probe(); return reason; } }

        public static string AdapterName { get { Probe(); return adapter; } }

        // Saved in PlayerPrefs; RoyalTracerSettings.enabledByDefault until the player chooses.
        public static bool UserEnabled =>
            PlayerPrefs.HasKey(EnabledKey) ? PlayerPrefs.GetInt(EnabledKey) != 0 : RoyalTracerSettings.Instance.enabledByDefault;

        public static void SetUserEnabled(bool enabled)
        {
            PlayerPrefs.SetInt(EnabledKey, enabled ? 1 : 0);
            PlayerPrefs.Save();
            RoyalTracerSystem.FollowUserChoice();
        }

        // Saved in PlayerPrefs; RoyalTracerSettings.quality until set.
        public static PathTracingQuality Quality
        {
            get => PlayerPrefs.HasKey(QualityKey) ? (PathTracingQuality)PlayerPrefs.GetInt(QualityKey) : RoyalTracerSettings.Instance.quality;
            set
            {
                PlayerPrefs.SetInt(QualityKey, (int)value);
                PlayerPrefs.Save();
                RoyalTracerSystem.ApplyQuality();
            }
        }

        public static bool Active => RoyalTracerSystem.Enabled;

        // DLSS frame generation while path tracing, in player builds on GPUs that support it (FrameGenerationMax).
        // Saved in PlayerPrefs; off until the player chooses. Above FrameGenerationMax it runs at the most there is.
        public static FrameGenerationMode FrameGeneration
        {
            get => (FrameGenerationMode)Mathf.Clamp(PlayerPrefs.GetInt(FrameGenerationKey, 0), 0, (int)FrameGenerationMode.X4);
            set
            {
                PlayerPrefs.SetInt(FrameGenerationKey, (int)value);
                PlayerPrefs.Save();
            }
        }

        // The most this machine can show: Off where frame generation cannot run (the editor, path tracing
        // unavailable, a GPU without it).
        public static FrameGenerationMode FrameGenerationMax => RoyalTracerSystem.FrameGenerationMax;

        // Frame generation runs now.
        public static bool FrameGenerationActive => RoyalTracerSystem.FrameGenerationActive;

        // Why frame generation does not run although wanted (the editor, an unsupported GPU, HDR output); empty
        // otherwise.
        public static string FrameGenerationIssue => RoyalTracerSystem.FrameGenerationIssue;

        // One line for logs: presents and frame generation's state.
        public static string FrameGenerationStats => Native.PresenterStats;

        // Raised on the main thread when path tracing starts or stops.
        public static event Action<bool> ActiveChanged;

        internal static void RaiseActiveChanged(bool active) => ActiveChanged?.Invoke(active);

        // A frame failure stops path tracing for the session; the reason then shows like an unsupported GPU's.
        internal static void MarkUnavailable(string why)
        {
            probed = true;
            reason = why;
        }

        static void Probe()
        {
            if (probed)
                return;
            probed = true;
            reason = HostReason() ?? RoyalTracerSystem.Initialize(out adapter) ?? "";
            if (reason.Length > 0)
                Debug.Log("[RoyalTracer] Path tracing unavailable: " + reason);
        }

        static string HostReason()
        {
#if UNITY_SERVER
            return "Dedicated server builds do not render.";
#else
            if (Application.platform is not (RuntimePlatform.WindowsPlayer or RuntimePlatform.WindowsEditor) || !Environment.Is64BitProcess)
                return "Requires 64-bit Windows.";
            var forced = Environment.GetEnvironmentVariable(ForceUnsupportedVariable);
            if (!string.IsNullOrEmpty(forced))
                return $"Turned off by {ForceUnsupportedVariable}={forced}.";
            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D12)
                return $"Requires Direct3D 12 (the game is running on {SystemInfo.graphicsDeviceType}).";
            if (GraphicsSettings.currentRenderPipeline is not HDRenderPipelineAsset)
                return "Requires the High Definition Render Pipeline.";
            int needMB = RoyalTracerSettings.Instance.minVideoMemoryMB, haveMB = SystemInfo.graphicsMemorySize;
            if (haveMB < needMB)
                return $"Requires {needMB / 1024f:0.#} GB of video memory ({SystemInfo.graphicsDeviceName} has {haveMB / 1024f:0.#} GB).";
            return null;
#endif
        }
    }
}
