using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace RoyalTracer.Editor
{
    // Unattended check without play mode:
    //   Unity.exe -batchmode -force-d3d12 -projectPath <project> -executeMethod RoyalTracer.Editor.RoyalTracerCapture.Run
    //             -rtCaptureOut <folder> [-rtScene <scene asset path>] [-rtCamera <object name>] [-rtPrefab <prefab asset path>]
    //             [-rtWidth 1920 -rtHeight 1080] [-rtExpectUnavailable]
    // Without -rtScene it builds RoyalTracerTestScene's scene (-rtNight: lamps only). -rtPrefab places that prefab in front of the camera.
    // Writes raster.png, pathtraced.png, tracer_full.png (the tracer's own image), mask.png (green = pixels the tracer
    // replaced) and tracer.png (the tracer wherever it hit geometry). If path tracing is unavailable it writes only
    // raster.png and logs the reason; -rtExpectUnavailable makes that the passing outcome.
    public static class RoyalTracerCapture
    {
        public static void Run()
        {
            int exitCode = 1;
            try
            {
                string outDir = Arg("-rtCaptureOut") ?? Path.GetFullPath("RoyalTracerCapture");
                Directory.CreateDirectory(outDir);
                if (Environment.GetCommandLineArgs().Contains("-rtVerbose"))
                    RoyalTracerSettings.Instance.verboseLog = true;
                var scene = Arg("-rtScene");
                if (scene != null)
                    EditorSceneManager.OpenScene(scene);
                else
                    RoyalTracerTestScene.Build(Environment.GetCommandLineArgs().Contains("-rtNight"));
                var cameraName = Arg("-rtCamera");
                var camera = cameraName != null ? GameObject.Find(cameraName)?.GetComponent<Camera>() : Camera.main;
                if (!camera)
                    camera = UnityEngine.Object.FindObjectsByType<Camera>().FirstOrDefault(c => c.isActiveAndEnabled);
                if (!camera)
                    throw new Exception("No camera" + (cameraName != null ? " named " + cameraName : ""));
                var prefab = Arg("-rtPrefab");
                if (prefab != null)
                    Place(camera, prefab);

                int width = int.Parse(Arg("-rtWidth") ?? "1920"), height = int.Parse(Arg("-rtHeight") ?? "1080");
                var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                camera.targetTexture = rt;
                Render(camera, 60);
                Save(rt, Path.Combine(outDir, "raster.png"));

                bool expectUnavailable = Environment.GetCommandLineArgs().Contains("-rtExpectUnavailable");
                if (!PathTracing.Available)
                {
                    Debug.Log("[RoyalTracerCapture] Unavailable: " + PathTracing.UnavailableReason);
                    exitCode = expectUnavailable ? 0 : 1;
                }
                else if (expectUnavailable)
                {
                    Debug.LogError("[RoyalTracerCapture] Path tracing is available on " + PathTracing.AdapterName + " although unavailability was expected");
                }
                else
                {
                    if (!RoyalTracerSystem.Enable(camera))
                        throw new Exception("RoyalTracerSystem.Enable failed (nothing to trace?)");
                    Render(camera, 90);
                    Save(rt, Path.Combine(outDir, "pathtraced.png"));
                    SaveTracerOutput(Path.Combine(outDir, "tracer_full.png"));
                    RoyalTracerSystem.ShowCompositeMask = true;
                    Render(camera, 2);
                    Save(rt, Path.Combine(outDir, "mask.png"));
                    RoyalTracerSystem.ShowCompositeMask = false;
                    RoyalTracerSystem.ShowRawTracer = true;
                    Render(camera, 2);
                    Save(rt, Path.Combine(outDir, "tracer.png"));
                    RoyalTracerSystem.ShowRawTracer = false;
                    RoyalTracerSystem.Shutdown();
                    exitCode = 0;
                }
                camera.targetTexture = null;
                Debug.Log("[RoyalTracerCapture] Wrote captures to " + outDir);
            }
            catch (Exception e)
            {
                Debug.LogError("[RoyalTracerCapture] " + e);
            }
            EditorApplication.Exit(exitCode);
        }

        // Puts the prefab on the ground plane in front of the camera, broadside, and aims the camera at it.
        static void Place(Camera camera, string path)
        {
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(path);
            if (!prefab)
                throw new Exception("No prefab at " + path);
            var instance = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            var forward = Vector3.ProjectOnPlane(camera.transform.forward, Vector3.up).normalized;
            var bounds = Bounds(instance);
            float distance = Mathf.Max(bounds.size.x, bounds.size.z) * 0.75f;
            var position = camera.transform.position + forward * distance - (bounds.center - instance.transform.position);
            instance.transform.position = new Vector3(position.x, 0f, position.z);
            instance.transform.rotation = Quaternion.LookRotation(Vector3.Cross(Vector3.up, forward));
            bounds = Bounds(instance);
            camera.transform.LookAt(bounds.center);
            Debug.Log($"[RoyalTracerCapture] Placed {prefab.name}, {bounds.size.magnitude:F0} m, camera at {camera.transform.position}");
        }

        static Bounds Bounds(GameObject go)
        {
            var bounds = new Bounds(go.transform.position, Vector3.zero);
            foreach (var r in go.GetComponentsInChildren<Renderer>())
                bounds.Encapsulate(r.bounds);
            return bounds;
        }

        static void Render(Camera camera, int frames)
        {
            for (int i = 0; i < frames; ++i)
            {
                camera.Render();
                UnityEngine.Rendering.AsyncGPUReadback.WaitAllRequests(); // delivers readbacks (HDRP's exposure) between frames
            }
        }

        // The tracer's own image, everything it saw, before the composite.
        static void SaveTracerOutput(string path)
        {
            var source = RoyalTracerSystem.LastColor;
            if (!source)
                return;
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
