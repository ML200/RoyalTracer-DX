using UnityEngine;

namespace RoyalTracer
{
    // DLSS Ray Reconstruction input resolution; values match RT_Quality.
    public enum PathTracingQuality
    {
        DLAA = 0,
        Quality = 1,
        Balanced = 2,
        Performance = 3,
        UltraPerformance = 4,
    }

    // DLSS frame generation while path tracing: frames shown per rendered frame (values: frames generated).
    public enum FrameGenerationMode
    {
        Off = 0,
        X2 = 1,
        X3 = 2,
        X4 = 3,
    }

    // Project-wide defaults, edited under Project Settings > RoyalTracer. The asset lives at
    // Assets/Resources/RoyalTracerSettings.asset; without it the defaults below apply.
    public sealed class RoyalTracerSettings : ScriptableObject
    {
        public const string ResourceName = "RoyalTracerSettings";

        [Tooltip("Path tracing starts on supported GPUs until the player turns it off (PathTracing.SetUserEnabled).")]
        public bool enabledByDefault = true;

        [Tooltip("Default DLSS Ray Reconstruction input resolution (PathTracing.Quality).")]
        public PathTracingQuality quality = PathTracingQuality.DLAA;

        [Tooltip("GPUs with less dedicated video memory are treated as unsupported. 8 GB cards report slightly under 8192 MB.")]
        public int minVideoMemoryMB = 7680;

        [Tooltip("Development builds and the editor: this key turns path tracing on and off for the session. None disables it.")]
        public KeyCode toggleKey = KeyCode.F9;

        [Tooltip("Light the path traced scene with HDRP's sky (its sky reflection, with clouds). Off: RoyalTracer's own physical atmosphere, driven by HDRP's sun.")]
        public bool useHdrpSky = true;

        [Tooltip("While path tracing, HDRP draws LOD groups at their full-detail level, the one RoyalTracer traces, so both agree on every outline. Off: HDRP keeps switching levels, and its own pixels can show along the edges of objects at a lower level.")]
        public bool fullDetailLods = true;

        [Tooltip("Forward the renderer's informational log lines to the Unity console.")]
        public bool verboseLog;

        static RoyalTracerSettings instance;

        public static RoyalTracerSettings Instance
        {
            get
            {
                if (!instance)
                {
                    instance = Resources.Load<RoyalTracerSettings>(ResourceName);
                    if (!instance)
                    {
                        instance = CreateInstance<RoyalTracerSettings>();
                        instance.hideFlags = HideFlags.HideAndDontSave;
                    }
                }
                return instance;
            }
        }

        public static void Reload() => instance = null; // after the asset was created or edited
    }
}
