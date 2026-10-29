using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // Path traced opaque surfaces and ocean for HDRP game cameras. Enabling hands the loaded scenes' supported renderers
    // to RoyalTracer (its own D3D12 device), and a custom pass before transparents writes its image, fogged as HDRP
    // fogs, into the camera colour. HDRP still draws the sky, transparents, VFX, post-processing and UI.
    // In play mode a driver keeps it running while PathTracing.Available && PathTracing.UserEnabled, for the game
    // cameras. Outside play mode the editor can path trace the Scene view instead (SceneViewTracing). The tracer keeps
    // one temporal history, so it follows one of them at a time.
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
        static WaterTracker water;
        static SkyTracker sky;
        static FramePresenter presenter;
        static ulong presentTicket; // this frame's ticket for the presenter, 0 when none
        static Camera tracedCamera; // outside play mode the only camera traced; in play mode the one scanned for
        static Pass pass;
        static Material composite;
        static RenderTexture exposureTarget;
        static float exposure; // HDRP's exposure multiplier, a frame or two old; 0 until read back
        static float nextScan;
        static readonly OutputTextures outputs = new();
        public static Texture LastColor { get; private set; } // the last composited tracer output, for captures

        // What the tracer's scene holds, for logs and tools.
        public static string Stats => initialized ? Native.Stats + (tracker != null ? $"; {tracker.Count} renderers, {tracker.Pending} waiting for mesh data, {lights.Count} lights, {particles.Count} particle systems, trails and lines ({particles.Emitters} emitters), {effects.Count} exporting VFX ({effects.Particles} particles, {effects.Emitters} emitters), {water.Overlays} water overlays" : "") : "not running";
        // Cameras whose anti-aliasing and upscaling are off while path tracing: their own settings.
        static readonly Dictionary<HDAdditionalCameraData, (HDAdditionalCameraData.AntialiasingMode antialiasing, bool dynamicResolution, bool dlss)> savedCameras = new();
        // Cameras whose HDRP water is off while the tracer draws the ocean: their own frame-settings state.
        static readonly Dictionary<HDAdditionalCameraData, (bool custom, bool overridden, bool enabled)> savedWater = new();

        internal static bool FrameGenerationActive => presenter != null && presenter.Active;
        internal static string FrameGenerationIssue =>
            Application.isEditor ? "Frame generation runs in player builds only." :
            FrameGenerationMax == FrameGenerationMode.Off ? "This GPU has no DLSS frame generation." : presenter?.Failure ?? "";

        // The GPU's most, once frame generation has run on it (remembered for later sessions); until then up to 4x.
        const string FrameGenerationMaxKey = "RoyalTracer.FrameGenerationMax";
        internal static FrameGenerationMode FrameGenerationMax
        {
            get
            {
                if (Application.isEditor || !PathTracing.Available)
                    return FrameGenerationMode.Off;
                int max = Native.RT_FrameGenerationMax();
                if (max > 0 && PlayerPrefs.GetInt(FrameGenerationMaxKey, 0) != max)
                {
                    PlayerPrefs.SetInt(FrameGenerationMaxKey, max);
                    PlayerPrefs.Save();
                }
                else if (max < 0)
                    max = PlayerPrefs.GetInt(FrameGenerationMaxKey, (int)FrameGenerationMode.X4);
                return (FrameGenerationMode)Mathf.Clamp(max, 0, (int)FrameGenerationMode.X4);
            }
        }

        // The tracer draws HDRP's ocean: the water stand-in plane goes.
        internal static bool WaterTraced => water != null && water.Active;

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
            tracedCamera = camera;
            tracker.Scan(camera, wait: true);
            lights.Scan();
            particles.Scan();
            effects.Scan();
            water.Scan();
            CustomPassVolume.RegisterGlobalCustomPass(CustomPassInjectionPoint.BeforeTransparent, pass);
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
            foreach (var (data, state) in savedCameras)
                if (data)
                {
                    data.antialiasing = state.antialiasing;
                    data.allowDynamicResolution = state.dynamicResolution;
                    data.allowDeepLearningSuperSampling = state.dlss;
                }
            savedCameras.Clear();
            foreach (var (data, state) in savedWater)
                if (data)
                    SetWaterRendering(data, state.custom, state.overridden, state.enabled);
            savedWater.Clear();
            outputs.Clear();
            tracedCamera = null;
            Enabled = false;
            Debug.Log("[RoyalTracer] Path tracing off");
            PathTracing.RaiseActiveChanged(false);
            Shutdown(); // frees the renderer and its scene; the next start builds both again
        }

        // The toggle key, read through the Input System where it runs, else through the Input Manager.
        static bool KeyDown(KeyCode key)
        {
#if ROYALTRACER_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard != null)
            {
                var k = InputSystemKey(key);
                return k != UnityEngine.InputSystem.Key.None && keyboard[k].wasPressedThisFrame;
            }
#endif
            try { return Input.GetKeyDown(key); }
            catch (InvalidOperationException) { return false; } // the Input System only, and no keyboard
        }

#if ROYALTRACER_INPUT_SYSTEM
        // The Input System names its keys as KeyCode does, but for the digits.
        static UnityEngine.InputSystem.Key InputSystemKey(KeyCode key)
        {
            if (key >= KeyCode.Alpha0 && key <= KeyCode.Alpha9)
                return UnityEngine.InputSystem.Key.Digit0 + (key - KeyCode.Alpha0);
            return Enum.TryParse(key.ToString(), out UnityEngine.InputSystem.Key k) ? k : UnityEngine.InputSystem.Key.None;
        }
#endif

        // Once per frame (Driver.Update): toggle key, deferred stop, following the player's choice, and keeping the
        // tracer's scene in step with Unity's.
        public static void Tick() => Tick(null, Camera.main);

        // wanted: null follows the player's choice.
        static void Tick(bool? wanted, Camera camera)
        {
            outputs.Tick();
            presenter?.Follow(Enabled && !disableRequested
                ? (FrameGenerationMode)Mathf.Min((int)PathTracing.FrameGeneration, (int)FrameGenerationMax)
                : FrameGenerationMode.Off);
            if (disableRequested)
            {
                disableRequested = false;
                Stop();
            }
            var key = RoyalTracerSettings.Instance.toggleKey;
            bool toggle = false;
            if (key != KeyCode.None && Application.isPlaying && (Application.isEditor || Debug.isDebugBuild))
                toggle = KeyDown(key);
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
            if (!(wanted ?? Wanted))
            {
                Stop();
                return;
            }
            if (!PathTracing.Available)
                return;
            if (Enabled && camera)
                tracedCamera = camera; // the same scene from another view
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
            water.Scan();
        }

#if UNITY_EDITOR
        const string SceneViewKey = "RoyalTracer.SceneView";

        // Outside play mode, the last active Scene view is path traced while this is on (an editor preference).
        public static bool SceneViewTracing
        {
            get => UnityEditor.EditorPrefs.GetBool(SceneViewKey, false);
            set
            {
                UnityEditor.EditorPrefs.SetBool(SceneViewKey, value);
                nextScan = 0f;
                if (!value && !Application.isPlaying && sessionChoice != true)
                    Stop();
                UnityEditor.SceneView.RepaintAll();
            }
        }

        [UnityEditor.InitializeOnLoadMethod]
        static void InstallEditorDriver()
        {
            UnityEditor.EditorApplication.update += EditorTick;
            UnityEditor.EditorApplication.playModeStateChanged += state =>
            {
                // Back in edit mode the game's choice no longer applies; only the Scene view keeps it running.
                if (state != UnityEditor.PlayModeStateChange.EnteredEditMode)
                    return;
                sessionChoice = null;
                if (!SceneViewTracing)
                    Stop();
            };
        }

        static void EditorTick()
        {
            // Play mode has its driver; a tool that enabled path tracing for a camera of its own keeps it.
            if (Application.isPlaying || !SceneViewTracing || sessionChoice == true)
                return;
            var view = UnityEditor.SceneView.lastActiveSceneView;
            Tick(true, view && !view.camera.orthographic ? view.camera : null);
            if (Enabled && view)
                view.Repaint(); // the tracer converges over frames
        }
#endif

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
                water = new WaterTracker();
                sky = new SkyTracker();
                presenter = new FramePresenter(composite);
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
            presenter.Dispose(); // its window and shared texture, before the renderer goes
            presenter = null;
            water.Dispose(); // its shared textures, before the device's scene goes
            water = null;
            sky.Dispose();
            sky = null;
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
            // In play mode the game cameras; otherwise the one camera followed (the Scene view, or a tool's).
            bool traced = Application.isPlaying ? camera.cameraType == CameraType.Game : camera == tracedCamera;
            if (!Enabled || disableRequested || !traced || camera.orthographic)
                return;
            // DLSS Ray Reconstruction anti-aliases and upscales the traced image: HDRP's anti-aliasing and upscaling on
            // top would only soften it, and TAA's jitter or a lower render size would part HDRP's depth from the tracer's.
            camera.TryGetComponent(out HDAdditionalCameraData data);
            if (data && !savedCameras.ContainsKey(data))
            {
                savedCameras[data] = (data.antialiasing, data.allowDynamicResolution, data.allowDeepLearningSuperSampling);
                data.antialiasing = HDAdditionalCameraData.AntialiasingMode.None;
                data.allowDynamicResolution = false;
                data.allowDeepLearningSuperSampling = false;
            }
            // While the tracer draws the ocean, this camera draws no HDRP water (from its next frame).
            if (data)
                HideWater(data, water.Active);

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
            if (ticket.color0 == IntPtr.Zero)
            {
                // The plugin never got Unity's device: in the editor, a plugin updated while it runs loads without it.
                Debug.LogError("[RoyalTracer] The plugin has no access to Unity's D3D12 device. If the package was updated while the editor was open, restart the editor.");
                PathTracing.MarkUnavailable("Restart the editor: the path tracer's plugin was loaded without Unity's graphics device.");
                disableRequested = true;
                return;
            }
            outputs.Update(ticket);
            LastColor = outputs.Color(ticket.slot);

            var cmd = ctx.cmd;
            sky.Record(cmd, RoyalTracerSettings.Instance.useHdrpSky);
            water.Record(cmd, ticket.index); // HDRP's bands and the overlays of this frame, copied before the render event orders the tracer after them
            cmd.IssuePluginEventAndData(renderEvent, Native.EventRender, (IntPtr)(long)ticket.index);
            presentTicket = ticket.index;
            ctx.propertyBlock.SetTexture(ColorId, outputs.Color(ticket.slot));
            ctx.propertyBlock.SetTexture(DepthId, outputs.Depth(ticket.slot));
            ctx.propertyBlock.SetFloat(InvUnitScaleId, 1f / UnitScale);
            ctx.propertyBlock.SetFloat(MaskId, ShowRawTracer ? 2f : ShowCompositeMask ? 1f : 0f);
            CoreUtils.SetRenderTarget(cmd, ctx.cameraColorBuffer, ctx.cameraDepthBuffer, ClearFlag.None); // the ocean writes depth
            CoreUtils.DrawFullScreen(cmd, composite, ctx.propertyBlock, shaderPassId: 0);
            // No release event: the next frame's render event releases this ticket, saving Unity a command buffer flush.
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

        static void HideWater(HDAdditionalCameraData data, bool hide)
        {
            if (hide && !savedWater.ContainsKey(data))
            {
                savedWater[data] = (data.customRenderingSettings,
                    data.renderingPathCustomFrameSettingsOverrideMask.mask[(uint)FrameSettingsField.Water],
                    data.renderingPathCustomFrameSettings.IsEnabled(FrameSettingsField.Water));
                SetWaterRendering(data, true, true, false);
            }
            else if (!hide && savedWater.Remove(data, out var state))
                SetWaterRendering(data, state.custom, state.overridden, state.enabled);
        }

        static void SetWaterRendering(HDAdditionalCameraData data, bool custom, bool overridden, bool enabled)
        {
            data.customRenderingSettings = custom;
            var mask = data.renderingPathCustomFrameSettingsOverrideMask;
            var bits = mask.mask;
            bits[(uint)FrameSettingsField.Water] = overridden;
            mask.mask = bits;
            data.renderingPathCustomFrameSettingsOverrideMask = mask;
            var settings = data.renderingPathCustomFrameSettings;
            settings.SetEnabled(FrameSettingsField.Water, enabled);
            data.renderingPathCustomFrameSettings = settings;
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

        // End of frame, after the UI: the frame's finished image to the presenter.
        static void EndOfFrame()
        {
            if (presentTicket != 0 && presenter != null && Enabled)
                presenter.Present(presentTicket, renderEvent);
            presentTicket = 0;
        }

        sealed class Driver : MonoBehaviour
        {
            void Update() => Tick();

            System.Collections.IEnumerator Start()
            {
                var endOfFrame = new WaitForEndOfFrame();
                while (true)
                {
                    yield return endOfFrame;
                    EndOfFrame();
                }
            }
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
