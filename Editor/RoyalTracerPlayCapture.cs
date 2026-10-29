using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RoyalTracer.Editor
{
    // Unattended check in play mode, where the game's own systems run, static batching applies and path tracing starts
    // by itself:
    //   Unity.exe -batchmode -force-d3d12 -projectPath <project> -executeMethod RoyalTracer.Editor.RoyalTracerPlayCapture.Run
    //             -rtCaptureOut <folder> (-rtScene <scene asset path> | -rtTestScene <new scene asset path> [-rtNight])
    //             [-rtCamera <object name>] [-rtSeconds 10] [-rtBroadcast <method>,<method>...]
    // -rtTestScene builds the package's test scene and saves it at that path first. -rtBroadcast sends those messages to
    // the scene's objects once path tracing runs (to start effects, for example).
    // Enters play mode, waits for path tracing and then -rtSeconds for the scene to settle and stream in, writes
    // pathtraced.png, tracer_full.png and mask.png, turns path tracing off, writes raster.png and exits.
    [InitializeOnLoad]
    public static class RoyalTracerPlayCapture
    {
        const string StateKey = "RoyalTracer.PlayCapture";
        static int stage, frame;
        static double stageStart;
        static Camera camera;

        // Play mode reloads scripts; the session state carries the request across.
        static RoyalTracerPlayCapture()
        {
            if (!string.IsNullOrEmpty(SessionState.GetString(StateKey, "")))
                EditorApplication.update += Step;
        }

        public static void Run()
        {
            var scene = Arg("-rtScene");
            var testScene = Arg("-rtTestScene");
            if (testScene != null) // play mode reloads scenes from disk: the test scene is saved there first
            {
                RoyalTracerTestScene.Build(Environment.GetCommandLineArgs().Contains("-rtNight"));
                EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene(), testScene);
            }
            else if (scene != null)
                EditorSceneManager.OpenScene(scene);
            else
            {
                Debug.LogError("[RoyalTracerPlayCapture] -rtScene or -rtTestScene is required");
                EditorApplication.Exit(1);
                return;
            }
            var outDir = Arg("-rtCaptureOut") ?? Path.GetFullPath("RoyalTracerCapture");
            Directory.CreateDirectory(outDir);
            SessionState.SetString(StateKey, string.Join("|", outDir, Arg("-rtCamera") ?? "", Arg("-rtSeconds") ?? "10",
                Arg("-rtBroadcast") ?? ""));
            EditorApplication.update += Step;
            EditorApplication.EnterPlaymode();
        }

        static void Step()
        {
            if (!EditorApplication.isPlaying)
                return;
            var parts = SessionState.GetString(StateKey, "").Split('|');
            if (parts.Length < 3)
                return;
            string outDir = parts[0], cameraName = parts[1];
            float settle = float.Parse(parts[2], System.Globalization.CultureInfo.InvariantCulture);
            double now = EditorApplication.timeSinceStartup;
            if (stage == 0)
            {
                stage = 1;
                stageStart = now;
            }
            try
            {
                if (stage == 1) // path tracing starts by itself when available and enabled
                {
                    if (PathTracing.Active || !PathTracing.Available)
                    {
                        stage = 2;
                        // Lets the scene's own scripts start something (effects, a salvo) once path tracing runs.
                        if (parts.Length > 3)
                            foreach (var message in parts[3].Split(',', StringSplitOptions.RemoveEmptyEntries))
                                foreach (var root in UnityEngine.SceneManagement.SceneManager.GetActiveScene().GetRootGameObjects())
                                    root.BroadcastMessage(message, SendMessageOptions.DontRequireReceiver);
                        stageStart = now;
                    }
                    else if (now - stageStart > 120)
                        throw new Exception("Path tracing did not start: " + (PathTracing.Available ? "available, not enabled" : PathTracing.UnavailableReason));
                    else if (now - stageStart > 3 && !RoyalTracerSystem.Enable()) // this session only; logs why not
                        throw new Exception($"Path tracing did not start (camera: {(Camera.main ? Camera.main.name : "none")}, " +
                                            $"{UnityEngine.SceneManagement.SceneManager.GetActiveScene().rootCount} root objects)");
                }
                else if (stage == 2 && now - stageStart > settle)
                {
                    camera = string.IsNullOrEmpty(cameraName) ? Camera.main : GameObject.Find(cameraName)?.GetComponent<Camera>();
                    if (!camera)
                        camera = UnityEngine.Object.FindObjectsByType<Camera>().FirstOrDefault(c => c.isActiveAndEnabled);
                    if (!camera)
                        throw new Exception("No camera");
                    // Batch mode draws no game view, so the camera renders only here: one frame per update, which
                    // lets HDRP's per-frame history (exposure adaptation, TAA) advance as in a game.
                    camera.targetTexture = new RenderTexture(1920, 1080, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                    frame = 0;
                    stage = PathTracing.Active ? 3 : 5;
                    if (stage == 5)
                        Debug.Log("[RoyalTracerPlayCapture] Unavailable: " + PathTracing.UnavailableReason);
                }
                else if (stage >= 3)
                {
                    camera.Render();
                    ++frame;
                    var rt = camera.targetTexture;
                    if (stage == 3 && frame == 120)
                    {
                        Save(rt, Path.Combine(outDir, "pathtraced.png"));
                        SaveTracerOutput(Path.Combine(outDir, "tracer_full.png"));
                        RoyalTracerSystem.ShowCompositeMask = true;
                        (stage, frame) = (4, 0);
                    }
                    else if (stage == 4 && frame == 3)
                    {
                        Save(rt, Path.Combine(outDir, "mask.png"));
                        RoyalTracerSystem.ShowCompositeMask = false;
                        Debug.Log("[RoyalTracerPlayCapture] Stream: " + RoyalTracerSystem.Stats);
                        foreach (var effect in UnityEngine.Object.FindObjectsByType<UnityEngine.VFX.VisualEffect>())
                            Debug.Log($"[RoyalTracerPlayCapture] VFX '{effect.name}': {effect.aliveParticleCount} particles, culled {effect.culled}");
                        RoyalTracerSystem.Disable();
                        (stage, frame) = (5, 0);
                    }
                    else if (stage == 5 && frame == 120)
                    {
                        Save(rt, Path.Combine(outDir, "raster.png"));
                        Debug.Log("[RoyalTracerPlayCapture] Wrote captures to " + outDir);
                        Finish(0);
                    }
                }
            }
            catch (Exception e)
            {
                Debug.LogError("[RoyalTracerPlayCapture] " + e);
                Finish(1);
            }
        }

        static void Finish(int code)
        {
            SessionState.EraseString(StateKey);
            EditorApplication.update -= Step;
            EditorApplication.Exit(code);
        }

        static void SaveTracerOutput(string path)
        {
            var source = RoyalTracerSystem.LastColor;
            if (!source)
                return;
            var raw = UnityEngine.Rendering.AsyncGPUReadback.Request(source, 0, TextureFormat.RGBAFloat);
            raw.WaitForCompletion();
            if (!raw.hasError)
            {
                int nonFinite = 0;
                float alphaMin = float.MaxValue, alphaMax = float.MinValue, peak = 0f;
                foreach (var p in raw.GetData<Vector4>())
                {
                    if (!float.IsFinite(p.x) || !float.IsFinite(p.y) || !float.IsFinite(p.z) || !float.IsFinite(p.w))
                    {
                        ++nonFinite;
                        continue;
                    }
                    alphaMin = Mathf.Min(alphaMin, p.w);
                    alphaMax = Mathf.Max(alphaMax, p.w);
                    peak = Mathf.Max(peak, Mathf.Max(p.x, Mathf.Max(p.y, p.z)));
                }
                Debug.Log($"[RoyalTracerPlayCapture] Tracer output: {nonFinite} non-finite pixels, exposure log2 {alphaMin:F2}..{alphaMax:F2}, peak {peak:F2}");
            }
            var preview = new Material(Resources.Load<Shader>("RoyalTracer/TextureBake"));
            var rt = new RenderTexture(source.width, source.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
            Graphics.Blit(source, rt, preview, 3);
            Save(rt, path);
            rt.Release();
            UnityEngine.Object.DestroyImmediate(preview);
        }

        static void Save(RenderTexture rt, string path)
        {
            var previous = RenderTexture.active;
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            RenderTexture.active = previous;
            File.WriteAllBytes(path, tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);
        }

        static string Arg(string name)
        {
            var args = Environment.GetCommandLineArgs();
            int i = Array.IndexOf(args, name);
            return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        }
    }
}
