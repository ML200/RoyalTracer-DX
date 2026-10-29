using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer.Editor
{
    // Project Settings > RoyalTracer: project defaults, checks of the project setup, and this machine's GPU.
    static class RoyalTracerSettingsProvider
    {
        const string AssetPath = "Assets/Resources/" + RoyalTracerSettings.ResourceName + ".asset";
        static SerializedObject serialized;
        static bool machineChecked;

        [SettingsProvider]
        static SettingsProvider Create() => new("Project/RoyalTracer", SettingsScope.Project)
        {
            label = "RoyalTracer",
            keywords = new[] { "path tracing", "DLSS", "ray reconstruction", "RoyalTracer" },
            guiHandler = _ => Draw(),
        };

        static void Draw()
        {
            var asset = Resources.Load<RoyalTracerSettings>(RoyalTracerSettings.ResourceName);
            if (!asset)
            {
                EditorGUILayout.HelpBox("No settings asset in a Resources folder, so the defaults apply.", MessageType.Info);
                if (GUILayout.Button("Create " + AssetPath))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(AssetPath)!);
                    AssetDatabase.CreateAsset(ScriptableObject.CreateInstance<RoyalTracerSettings>(), AssetPath);
                    AssetDatabase.SaveAssets();
                    RoyalTracerSettings.Reload();
                }
            }
            else
            {
                if (serialized == null || serialized.targetObject != asset)
                    serialized = new SerializedObject(asset);
                serialized.Update();
                foreach (var name in new[] { "enabledByDefault", "quality", "minVideoMemoryMB", "toggleKey", "useHdrpSky", "verboseLog" })
                    EditorGUILayout.PropertyField(serialized.FindProperty(name));
                if (serialized.ApplyModifiedProperties())
                    RoyalTracerSettings.Reload();
            }

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Project", EditorStyles.boldLabel);
            bool ok = true;
            foreach (var (message, type) in ProjectProblems())
            {
                EditorGUILayout.HelpBox(message, type);
                ok = false;
            }
            if (!FirstWindowsApiIsD3D12() && GUILayout.Button("Use Direct3D 12 first for Windows"))
            {
                var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.StandaloneWindows64, false);
                PlayerSettings.SetGraphicsAPIs(BuildTarget.StandaloneWindows64,
                    apis.Where(a => a != GraphicsDeviceType.Direct3D12).Prepend(GraphicsDeviceType.Direct3D12).ToArray());
            }
            if (ok)
                EditorGUILayout.HelpBox("Windows builds start on Direct3D 12 with HDRP.", MessageType.None);

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("This machine", EditorStyles.boldLabel);
            if (SystemInfo.graphicsDeviceType != GraphicsDeviceType.Direct3D12)
                EditorGUILayout.HelpBox($"The editor runs on {SystemInfo.graphicsDeviceType}; start it on Direct3D 12 to path trace in play mode.", MessageType.Info);
            if (!machineChecked)
            {
                if (GUILayout.Button("Check this GPU"))
                    machineChecked = true; // the probe creates the renderer's device in the editor process
            }
            else if (PathTracing.Available)
                EditorGUILayout.HelpBox(PathTracing.AdapterName + ": path tracing available.", MessageType.Info);
            else
                EditorGUILayout.HelpBox(PathTracing.UnavailableReason, MessageType.Warning);

            EditorGUILayout.LabelField("Player choice (editor PlayerPrefs)", PathTracing.UserEnabled ? "On" : "Off");
            using (new EditorGUILayout.HorizontalScope())
            {
                if (GUILayout.Button("Turn on")) PathTracing.SetUserEnabled(true);
                if (GUILayout.Button("Turn off")) PathTracing.SetUserEnabled(false);
            }
        }

        static bool FirstWindowsApiIsD3D12()
        {
            var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
            return apis.Length > 0 && apis[0] == GraphicsDeviceType.Direct3D12;
        }

        static System.Collections.Generic.IEnumerable<(string, MessageType)> ProjectProblems()
        {
            if (!FirstWindowsApiIsD3D12())
            {
                var apis = PlayerSettings.GetGraphicsAPIs(BuildTarget.StandaloneWindows64);
                yield return ($"Windows builds start on {(apis.Length > 0 ? apis[0].ToString() : "no API")}; path tracing needs " +
                              "Direct3D 12 first in Player > Other Settings > Graphics APIs for Windows.", MessageType.Warning);
            }
            if (GraphicsSettings.defaultRenderPipeline is not HDRenderPipelineAsset)
                yield return ("The project does not use HDRP; RoyalTracer is an HDRP renderer.", MessageType.Error);
        }

        // A once-per-editor-session console hint for projects that cannot path trace yet.
        [InitializeOnLoadMethod]
        static void WarnOnce()
        {
            const string Key = "RoyalTracer.ProjectChecked";
            if (SessionState.GetBool(Key, false) || Application.isBatchMode)
                return;
            SessionState.SetBool(Key, true);
            EditorApplication.delayCall += () =>
            {
                foreach (var (message, _) in ProjectProblems())
                    Debug.LogWarning("[RoyalTracer] " + message + " (Project Settings > RoyalTracer)");
            };
        }
    }
}
