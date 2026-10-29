using UnityEditor;
using UnityEditor.ShortcutManagement;
using UnityEngine;

namespace RoyalTracer.Editor
{
    // Path tracing in the Scene view outside play mode: Tools > RoyalTracer > Path Trace Scene View, or the toggle key
    // (F9 by default, rebindable under Shortcuts) while the Scene view has focus.
    static class RoyalTracerSceneView
    {
        const string MenuPath = "Tools/RoyalTracer/Path Trace Scene View";

        [MenuItem(MenuPath, priority = 0)]
        static void Toggle()
        {
            RoyalTracerSystem.SceneViewTracing = !RoyalTracerSystem.SceneViewTracing;
            if (RoyalTracerSystem.SceneViewTracing && !PathTracing.Available)
                Debug.LogWarning("[RoyalTracer] Path tracing unavailable: " + PathTracing.UnavailableReason);
        }

        [MenuItem(MenuPath, true)]
        static bool ToggleValidate()
        {
            Menu.SetChecked(MenuPath, RoyalTracerSystem.SceneViewTracing);
            return true;
        }

        [Shortcut("RoyalTracer/Path Trace Scene View", typeof(SceneView), KeyCode.F9)]
        static void ToggleShortcut() => Toggle();
    }
}
