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
