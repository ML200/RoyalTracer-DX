using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace RoyalTracer.Editor
{
    // The plugin loads its runtime (Streamline, DXC, shaders, Agility 1.619) from <Game>_Data/Plugins/x86_64/
    // RoyalTracerData. The package keeps it in RoyalTracerData~, which Unity does not import, so builds copy it.
    sealed class RoyalTracerBuildProcessor : IPostprocessBuildWithReport
    {
        public int callbackOrder => 0;

        public void OnPostprocessBuild(BuildReport report)
        {
            var summary = report.summary;
            if (summary.platform != BuildTarget.StandaloneWindows64 ||
                EditorUserBuildSettings.standaloneBuildSubtarget == StandaloneBuildSubtarget.Server)
                return;
            var info = UnityEditor.PackageManager.PackageInfo.FindForAssembly(typeof(RoyalTracerSystem).Assembly);
            var source = info != null ? Path.Combine(info.resolvedPath, "RoyalTracerData~") : null;
            if (source == null || !Directory.Exists(source))
            {
                Debug.LogWarning("[RoyalTracer] No RoyalTracerData~ in the package; path tracing will be unavailable in this build.");
                return;
            }
            var dataFolder = Path.Combine(Path.GetDirectoryName(summary.outputPath)!,
                Path.GetFileNameWithoutExtension(summary.outputPath) + "_Data");
            Copy(source, Path.Combine(dataFolder, "Plugins", "x86_64", "RoyalTracerData"));
        }

        static void Copy(string from, string to)
        {
            Directory.CreateDirectory(to);
            foreach (var file in Directory.GetFiles(from))
                File.Copy(file, Path.Combine(to, Path.GetFileName(file)), true);
            foreach (var dir in Directory.GetDirectories(from))
                Copy(dir, Path.Combine(to, Path.GetFileName(dir)));
        }
    }
}
