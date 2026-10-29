using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // Path traced opaque surfaces for HDRP game cameras. Enabling hands the loaded scenes' supported renderers to
    // RoyalTracer (its own D3D12 device), and a custom pass after opaque lighting writes its image into the camera
    // colour. HDRP still draws sky, water, fog, transparents, VFX, post-processing and UI over it.
    // In play mode a driver keeps it running while PathTracing.Available && PathTracing.UserEnabled.
    public static class RoyalTracerSystem
    {
        // HDRP light values in the tracer's tuned range: a 100000 lux sun becomes 5.
        public const float UnitScale = 5f / 100000f;

        public static bool Enabled { get; private set; }
        public static bool ShowCompositeMask; // tints the pixels the tracer replaced
        public static bool ShowRawTracer;     // the tracer's image wherever it hit geometry, ignoring HDRP's depth
        public static float RescanInterval = 0.5f;

        static readonly int ColorId = Shader.PropertyToID("_RtColor"), DepthId = Shader.PropertyToID("_RtDepth"),
            InvUnitScaleId = Shader.PropertyToID("_RtInvUnitScale"), MaskId = Shader.PropertyToID("_RtShowMask");

        static bool initialized, disableRequested;
        static bool? sessionChoice; // Enable/Disable and the toggle key override the saved choice until it changes
        static Native.LogFn logFn;
        static IntPtr renderEvent;
        static SceneTracker tracker;
        static LightTracker lights;
        static ParticleTracker particles;
        static VfxTracker effects;
        static Pass pass;
        static Material composite;
        static RenderTexture exposureTarget;
        static float exposure; // HDRP's exposure multiplier, a frame or two old; 0 until read back
        static float nextScan;
        static readonly OutputTextures outputs = new();
        public static Texture LastColor { get; private set; } // the last composited tracer output, for captures

        // What the tracer's scene holds, for logs and tools.
        public static string Stats => initialized ? Native.Stats + (tracker != null ? $"; {tracker.Count} renderers, {tracker.Pending} waiting for mesh data, {lights.Count} lights, {particles.Count} particle systems, trails and lines ({particles.Emitters} emitters), {effects.Count} exporting VFX ({effects.Particles} particles, {effects.Emitters} emitters)" : "") : "not running";
        static readonly Dictionary<HDAdditionalCameraData, HDAdditionalCameraData.AntialiasingMode> savedAntialiasing = new();

        static bool Wanted => sessionChoice ?? PathTracing.UserEnabled;

#if !UNITY_SERVER
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void InstallDriver()
        {
            if (Application.platform is not (RuntimePlatform.WindowsPlayer or RuntimePlatform.WindowsEditor))
                return;
            sessionChoice = null;
            var go = new GameObject("RoyalTracer") { hideFlags = HideFlags.HideAndDontSave };
            UnityEngine.Object.DontDestroyOnLoad(go);
            go.AddComponent<Driver>();
        }
#endif

        // Starts path tracing for this session regardless of the saved choice (captures, tools).
        public static bool Enable(Camera camera = null)
        {
            sessionChoice = true;
            return Start(camera, quiet: false);
        }

        public static void Disable()
        {
            sessionChoice = false;
            Stop();
        }

        internal static void FollowUserChoice()
        {
            sessionChoice = null;
            nextScan = 0f;
            if (!Wanted)
                Stop();
        }

        internal static void ApplyQuality()
        {
            if (initialized)
                Native.RT_SetQuality((int)PathTracing.Quality);
        }

        static bool Start(Camera camera, bool quiet)
        {
            if (Enabled)
                return true;
            if (!PathTracing.Available)
            {
                if (!quiet)
                    Debug.LogWarning("[RoyalTracer] Path tracing unavailable: " + PathTracing.UnavailableReason);
                return false;
            }
            if (!initialized && Initialize(out _) is string error) // after a Shutdown
            {
                PathTracing.MarkUnavailable(error);
                return false;
            }
            camera = camera ? camera : Camera.main;
            if (!camera)
            {
                if (!quiet)
                    Debug.LogWarning("[RoyalTracer] No camera to path trace");
                return false;
            }
            tracker.Scan(camera, wait: true);
            lights.Scan();
            particles.Scan();
            effects.Scan();
            CustomPassVolume.RegisterGlobalCustomPass(CustomPassInjectionPoint.AfterOpaqueColor, pass);
            ApplyQuality();
            Enabled = true;
            disableRequested = false;
            nextScan = Time.realtimeSinceStartup + RescanInterval;
            Debug.Log("[RoyalTracer] Path tracing on");
            PathTracing.RaiseActiveChanged(true);
            return true;
        }

        static void Stop()
        {
            if (!Enabled)
                return;
            CustomPassVolume.UnregisterGlobalCustomPass(pass);
            foreach (var (data, mode) in savedAntialiasing)
                if (data)
                    data.antialiasing = mode;
            savedAntialiasing.Clear();
            outputs.Clear();
            Enabled = false;
            Debug.Log("[RoyalTracer] Path tracing off");
            PathTracing.RaiseActiveChanged(false);
            Shutdown(); // frees the renderer and its scene; the next start builds both again
        }

        // Once per frame (Driver.Update): toggle key, deferred stop, following the player's choice, and keeping the
        // tracer's scene in step with Unity's.
        public static void Tick()
        {
            outputs.Tick();
            if (disableRequested)
            {
                disableRequested = false;
                Stop();
            }
            var key = RoyalTracerSettings.Instance.toggleKey;
            bool toggle = false;
            if (key != KeyCode.None && (Application.isEditor || Debug.isDebugBuild))
                try { toggle = Input.GetKeyDown(key); } catch (InvalidOperationException) { } // new Input System only
            if (toggle)
            {
                sessionChoice = !Enabled;
                nextScan = 0f;
                if (!Enabled && !PathTracing.Available)
                    Debug.LogWarning("[RoyalTracer] Path tracing unavailable: " + PathTracing.UnavailableReason);
            }
            if (Time.realtimeSinceStartup < nextScan)
                return;
            nextScan = Time.realtimeSinceStartup + RescanInterval;
            if (!Wanted)
            {
                Stop();
                return;
            }
            if (!PathTracing.Available)
                return;
            var camera = Camera.main;
            if (!Enabled)
            {
                if (camera)
                    Start(camera, quiet: true);
                return;
            }
            tracker.Scan(camera); // new meshes arrive over the next frames
            lights.Scan();
            particles.Scan();
            effects.Scan();
        }

        // Creates the native device once per process. Returns null, or why path tracing cannot run.
        internal static string Initialize(out string adapter)
        {
            adapter = "";
            if (initialized)
                return null;
            try
            {
                logFn = OnLog;
                var desc = new Native.InitDesc
                {
                    dataDir = DataDir(),
                    cacheDir = Path.Combine(Application.persistentDataPath, "RoyalTracerShaderCache"),
                    log = Marshal.GetFunctionPointerForDelegate(logFn),
                };
                if (!Directory.Exists(desc.dataDir))
                    return "The RoyalTracer runtime data is missing (" + desc.dataDir + ").";
                int result = Native.RT_Init(ref desc);
                if (Native.RT_GetCapabilities(out var caps) == 1)
                {
                    adapter = caps.AdapterName;
                    Debug.Log($"[RoyalTracer] {adapter}: {caps.dedicatedVideoMemory >> 20} MB, shader model " +
                              $"{caps.shaderModel >> 4}.{caps.shaderModel & 15}, DXR {caps.raytracingTier / 10}.{caps.raytracingTier % 10}, " +
                              $"SER reorders {caps.serReorders != 0}, DLSS-RR {caps.dlssRayReconstruction != 0}");
                }
                if (result != 1)
                    return caps.Reason.Length > 0 ? caps.Reason + "." : Native.LastError;
                renderEvent = Native.RT_GetRenderEventFunc();
                composite = CoreUtils.CreateEngineMaterial(Resources.Load<Shader>("RoyalTracer/Composite"));
                tracker = new SceneTracker();
                lights = new LightTracker();
                particles = new ParticleTracker();
                effects = new VfxTracker();
                pass = new Pass { name = "RoyalTracer composite" };
                initialized = true;
#if UNITY_EDITOR
                UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += Shutdown;
#endif
                Application.quitting += Shutdown;
                return null;
            }
            catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
            {
                return "The RoyalTracer plugin binary could not be loaded (" + e.Message + ").";
            }
        }

        // Releases the renderer and its scene; the native device stays for the process.
        public static void Shutdown()
        {
            Stop();
            if (!initialized)
                return;
            Native.RT_Shutdown(); // releases the native scene with it
            tracker.Dispose();
            tracker = null;
            lights = null;
            particles.Dispose();
            particles = null;
            effects.Dispose();
            effects = null;
            CoreUtils.Destroy(composite);
            CoreUtils.Destroy(exposureTarget);
            exposure = 0f;
            initialized = false;
#if UNITY_EDITOR
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload -= Shutdown;
#endif
            Application.quitting -= Shutdown;
        }

        static string DataDir()
        {
#if UNITY_EDITOR
            var info = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(RoyalTracerSystem).Assembly);
            if (info != null)
                return Path.Combine(info.resolvedPath, "RoyalTracerData~");
#endif
            return Path.Combine(Application.dataPath, "Plugins", "x86_64", "RoyalTracerData");
        }

        [AOT.MonoPInvokeCallback(typeof(Native.LogFn))]
        static void OnLog(int level, IntPtr message)
        {
            var text = "[RoyalTracer] " + Marshal.PtrToStringUTF8(message);
            if (level == Native.LogError) Debug.LogError(text);
            else if (level == Native.LogWarning) Debug.LogWarning(text);
            else if (RoyalTracerSettings.Instance.verboseLog) Debug.Log(text);
        }

        internal static void RenderCamera(CustomPassContext ctx)
        {
            var camera = ctx.hdCamera.camera;
            if (!Enabled || disableRequested || camera.cameraType != CameraType.Game)
                return;
            // TAA jitters HDRP's depth; SMAA keeps it aligned with the tracer's unjittered depth.
            if (camera.TryGetComponent(out HDAdditionalCameraData data) &&
                data.antialiasing == HDAdditionalCameraData.AntialiasingMode.TemporalAntialiasing)
            {
                savedAntialiasing[data] = data.antialiasing;
                data.antialiasing = HDAdditionalCameraData.AntialiasingMode.SubpixelMorphologicalAntiAliasing;
            }

            tracker.PushTransforms();
            tracker.UpdateDeforming();
            lights.Update();
            particles.Update(exposure);
            var frame = Frame(camera, ctx.hdCamera.actualWidth, ctx.hdCamera.actualHeight);
            if (Native.RT_SubmitFrame(ref frame, out var ticket) != 1)
            {
                Debug.LogError("[RoyalTracer] " + Native.LastError);
                PathTracing.MarkUnavailable("Path tracing stopped after a renderer error.");
                disableRequested = true; // not while HDRP iterates its custom passes
                return;
            }
            outputs.Update(ticket);
            LastColor = outputs.Color(ticket.slot);

            var cmd = ctx.cmd;
            cmd.IssuePluginEventAndData(renderEvent, Native.EventRender, (IntPtr)(long)ticket.index);
            ctx.propertyBlock.SetTexture(ColorId, outputs.Color(ticket.slot));
            ctx.propertyBlock.SetTexture(DepthId, outputs.Depth(ticket.slot));
            ctx.propertyBlock.SetFloat(InvUnitScaleId, 1f / UnitScale);
            ctx.propertyBlock.SetFloat(MaskId, ShowRawTracer ? 2f : ShowCompositeMask ? 1f : 0f);
            CoreUtils.SetRenderTarget(cmd, ctx.cameraColorBuffer, ClearFlag.None);
            CoreUtils.DrawFullScreen(cmd, composite, ctx.propertyBlock, shaderPassId: 0);
            cmd.IssuePluginEventAndData(renderEvent, Native.EventRelease, (IntPtr)(long)ticket.index);
            effects.Record(cmd, exposure); // VFX particles of this frame's update, for the next frames

            if (!exposureTarget)
                exposureTarget = new RenderTexture(1, 1, 0, RenderTextureFormat.RFloat) { name = "RoyalTracer exposure" };
            CoreUtils.SetRenderTarget(cmd, exposureTarget, ClearFlag.None);
            CoreUtils.DrawFullScreen(cmd, composite, ctx.propertyBlock, shaderPassId: 1);
            cmd.RequestAsyncReadback(exposureTarget, r =>
            {
                if (!r.hasError && float.IsFinite(r.GetData<float>()[0]))
                    exposure = r.GetData<float>()[0];
            });
        }

        static Native.FrameDesc Frame(Camera camera, int width, int height)
        {
            var t = camera.transform;
            Vector3 eye = Space.Point(t.position), target = eye + Space.Point(t.forward), up = Space.Point(t.up);
            var frame = new Native.FrameDesc
            {
                eyeX = eye.x, eyeY = eye.y, eyeZ = eye.z,
                targetX = target.x, targetY = target.y, targetZ = target.z,
                upX = up.x, upY = up.y, upZ = up.z,
                fovYDegrees = camera.fieldOfView,
                width = (uint)width, height = (uint)height,
                deltaTime = Application.isPlaying ? Time.deltaTime : 1f / 60f,
                sunY = 1f, skyIntensity = 1f, turbidity = 2f,
            };
            var sun = (RenderPipelineManager.currentPipeline as HDRenderPipeline)?.GetMainLight();
            if (sun)
            {
                var toSun = Space.Point(-sun.transform.forward);
                frame.sunX = toSun.x; frame.sunY = toSun.y; frame.sunZ = toSun.z;
                var tint = sun.color.linear * (sun.useColorTemperature ? Mathf.CorrelatedColorTemperatureToRGB(sun.colorTemperature) : Color.white);
                frame.sunIntensity = sun.intensity * tint.grayscale * UnitScale;
                // The tracer's sky is lit by its sun: scaled with it, a dark night sky stays dark.
                frame.skyIntensity = Mathf.Clamp01(sun.intensity / 100000f);
            }
            float water = SceneTracker.WaterLevel();
            frame.skyGroundY = float.IsInfinity(water) ? -1f : water - 2f;
            return frame;
        }

        sealed class Pass : CustomPass
        {
            protected override void Execute(CustomPassContext ctx) => RenderCamera(ctx);
        }

        sealed class Driver : MonoBehaviour
        {
            void Update() => Tick();
        }

        // Unity textures over the plugin's shared outputs. Replaced ones live a few more frames, since queued GPU
        // work may still read them.
        sealed class OutputTextures
        {
            readonly Texture2D[] color = new Texture2D[2], depth = new Texture2D[2];
            readonly List<(Texture2D texture, int frame)> retired = new();
            uint generation;
            int frame;

            public Texture Color(uint slot) => color[slot];
            public Texture Depth(uint slot) => depth[slot];

            public void Update(Native.FrameTicket t)
            {
                if (color[0] && generation == t.generation)
                    return;
                Clear();
                generation = t.generation;
                color[0] = Texture2D.CreateExternalTexture((int)t.width, (int)t.height, TextureFormat.RGBAHalf, false, true, t.color0);
                color[1] = Texture2D.CreateExternalTexture((int)t.width, (int)t.height, TextureFormat.RGBAHalf, false, true, t.color1);
                depth[0] = Texture2D.CreateExternalTexture((int)t.width, (int)t.height, TextureFormat.RFloat, false, true, t.depth0);
                depth[1] = Texture2D.CreateExternalTexture((int)t.width, (int)t.height, TextureFormat.RFloat, false, true, t.depth1);
            }

            public void Clear()
            {
                for (int i = 0; i < 2; ++i)
                {
                    if (color[i]) retired.Add((color[i], frame));
                    if (depth[i]) retired.Add((depth[i], frame));
                    color[i] = depth[i] = null;
                }
            }

            public void Tick()
            {
                ++frame;
                for (int i = retired.Count - 1; i >= 0; --i)
                    if (frame - retired[i].frame > 4)
                    {
                        CoreUtils.Destroy(retired[i].texture);
                        retired.RemoveAt(i);
                    }
            }
        }
    }
}
