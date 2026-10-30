using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.VFX;
using UnityEditor.VFX.Block;
using UnityEngine;
using UnityEngine.VFX;

namespace RoyalTracer.Editor.Vfx
{
    // Lets VFX Graph particles light the path-traced scene: a Custom HLSL block at the end of the Update context of
    // every system that emits light appends its live particles to the exposed GraphicsBuffer "RoyalTracerEmitters",
    // which RoyalTracer binds and reads back (RoyalTracer.VfxTracker). The block reads colour, alpha and size as they stand after the update: effects that
    // set them only in an output export their Update values. Part of the VFX Graph editor assembly (see the .asmref),
    // whose graph model is internal.
    public static class RoyalTracerVfxLightExport
    {
        public const string BufferName = "RoyalTracerEmitters";
        const string BlockName = "RoyalTracer Light Export";
        const string Code = @"
// Appends this particle to RoyalTracer's light export: a sphere of the particle's size showing colour * alpha *
// Intensity, as its output draws it (ExposureWeight 0: a screen value, 1: radiance in nits). Record: position, radius,
// colour, flags (1: simulated in local space, + 2 * ExposureWeight in 1/255).
void RoyalTracerLightExport(inout VFXAttributes attributes, RWByteAddressBuffer RoyalTracerEmitters, float Intensity,
    float ExposureWeight)
{
    float3 radiance = attributes.color * (attributes.alpha * Intensity);
    if (!attributes.alive || max(radiance.r, max(radiance.g, radiance.b)) <= 0.0)
        return;
    uint bytes;
    RoyalTracerEmitters.GetDimensions(bytes);
    uint index;
    RoyalTracerEmitters.InterlockedAdd(0, 1u, index);
    if (16u + (index + 1u) * 32u > bytes)
        return;
    float3 size = attributes.size * abs(float3(attributes.scaleX, attributes.scaleY, attributes.scaleZ));
#if VFX_LOCAL_SPACE
    float local = 1.0;
#else
    float local = 0.0;
#endif
    float flags = local + 2.0 * round(saturate(ExposureWeight) * 255.0);
    uint at = 16u + index * 32u;
    RoyalTracerEmitters.Store4(at, asuint(float4(attributes.position, 0.5 * max(size.x, max(size.y, size.z)))));
    RoyalTracerEmitters.Store4(at + 16u, asuint(float4(radiance, flags)));
}";

        [MenuItem("Assets/RoyalTracer/Add Light Export", true)]
        static bool CanAddToSelection() => Selection.objects.OfType<VisualEffectAsset>().Any();

        [MenuItem("Assets/RoyalTracer/Add Light Export")]
        static void AddToSelection()
        {
            foreach (var asset in Selection.objects.OfType<VisualEffectAsset>())
            {
                var path = AssetDatabase.GetAssetPath(asset);
                Debug.Log($"[RoyalTracer] {path}: light export in {Add(path)} system(s)\n{Report(path)}");
            }
        }

        // Whether a system's particles light the scene: it draws unlit particles, which show their colour as
        // radiance. Lit outputs (their colour is albedo) and Shader Graph outputs do not. Unlit particles that look
        // dark at the camera's exposure are dropped at runtime, so dim smoke costs nothing. reason says why.
        internal static bool Emits(VFXContext update, out string reason)
        {
            reason = "no particle output";
            bool emits = false;
            foreach (var output in update.GetData().owners.OfType<VFXAbstractParticleOutput>())
            {
                var blend = (VFXAbstractRenderedOutput.BlendMode)output.GetSettingValue("blendMode");
                bool lit = IsLit(output);
                bool graph = output is VFXShaderGraphParticleOutput sg && sg.GetOrRefreshShaderGraphObject(false) != null;
                reason = $"{output.GetType().Name}, {blend}{(lit ? ", lit" : "")}{(graph ? ", Shader Graph" : "")}";
                emits |= !lit && !graph;
            }
            return emits;
        }

        // What the system's unlit outputs multiply its light by over the particles' lives: colour and alpha over life
        // (often HDR) and size over life (light grows with the area), averaged. The export reads colour and size in
        // Update, before the outputs apply these, so its Intensity carries them. Curves driven by graph links count as 1.
        internal static float OutputScale(VFXContext update)
        {
            float scale = 1f;
            const int Samples = 32;
            float Average(System.Func<float, float> f)
            {
                float sum = 0f;
                for (int i = 0; i < Samples; ++i)
                    sum += f((i + 0.5f) / Samples);
                return sum / Samples;
            }
            var output = update.GetData().owners.OfType<VFXAbstractParticleOutput>().FirstOrDefault(o => !IsLit(o));
            if (output == null)
                return scale;
            foreach (var curve in output.children.OfType<AttributeFromCurve>())
            {
                if (!curve.enabled || curve.inputSlots.Count == 0 || curve.inputSlots[0].HasLink(true) ||
                    curve.Composition is not (AttributeCompositionMode.Multiply or AttributeCompositionMode.Overwrite))
                    continue;
                var value = curve.inputSlots[0].value;
                if (curve.attribute == "color" && value is Gradient gradient)
                {
                    bool alpha = (curve.ColorMode & AttributeFromCurve.ColorApplicationMode.Alpha) != 0;
                    scale *= Average(t =>
                    {
                        var c = gradient.Evaluate(t);
                        return (0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b) * (alpha ? c.a : 1f);
                    });
                }
                else if (curve.attribute == "alpha" && value is AnimationCurve alphaCurve)
                    scale *= Average(alphaCurve.Evaluate);
                else if (curve.attribute == "size" && value is AnimationCurve sizeCurve && curve.Composition == AttributeCompositionMode.Multiply)
                    scale *= Average(t => sizeCurve.Evaluate(t) * sizeCurve.Evaluate(t));
            }
            return scale;
        }

        // How the system's unlit output applies the camera's exposure to its colour: 0 when it bypasses it (the
        // default for unlit outputs: the colour is a screen value), 1 when the colour is radiance in nits, or the
        // output's exposure weight. RoyalTracer converts with HDRP's exposure.
        internal static float ExposureWeight(VFXContext update)
        {
            var output = update.GetData().owners.OfType<VFXAbstractParticleOutput>().FirstOrDefault(o => !IsLit(o));
            if (output == null)
                return 1f;
            const System.Reflection.BindingFlags Flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            bool Property(string name) => (bool)typeof(VFXAbstractParticleOutput).GetProperty(name, Flags).GetValue(output);
            if (!Property("hasExposure"))
                return 0f;
            if ((bool)typeof(VFXAbstractParticleOutput).GetField("useExposureWeight", Flags).GetValue(output))
            {
                var slot = output.inputSlots.FirstOrDefault(s => s.name == "exposureWeight");
                return slot != null && !slot.HasLink(true) && slot.value is float w ? w : 1f;
            }
            return Property("bypassExposure") ? 0f : 1f;
        }

        static bool IsLit(VFXAbstractParticleOutput output)
        {
            for (var type = output.GetType(); type != null; type = type.BaseType)
                if (type.Name.Contains("LitOutput")) // HDRP's and URP's lit outputs; "UnlitOutput" does not match
                    return true;
            return false;
        }

        // One line per system of the graph at path: whether it exports its light, and why.
        public static string Report(string path)
        {
            var graph = VisualEffectResource.GetResourceAtPath(path).GetOrCreateGraph();
            return string.Join("\n", graph.children.OfType<VFXBasicUpdate>().Select(update =>
            {
                bool emits = Emits(update, out var reason);
                var name = update.GetData().title;
                var scale = emits ? $", light x{OutputScale(update):0.##}, exposure weight {ExposureWeight(update):0.##}" : "";
                return $"  {(string.IsNullOrEmpty(name) ? "(unnamed)" : name)}: {(emits ? "exports" : "skipped")} ({reason}{scale})";
            }));
        }

        // Adds the export to the graph's emitting systems, or refreshes it, and removes it from the others; Intensity
        // scales the exported light. A graph without emitting systems or earlier export is left as it is, and an export
        // block disabled by hand keeps its system out. Returns the number of exporting systems. Code in the VFX Graph editor assembly can pass emits to choose the systems
        // itself (Emits otherwise).
        public static int Add(string path, float intensity = 1f) => Add(path, intensity, null);

        internal static int Add(string path, float intensity, System.Func<VFXContext, bool> emits)
        {
            var resource = VisualEffectResource.GetResourceAtPath(path);
            var graph = resource.GetOrCreateGraph();
            var buffer = graph.children.OfType<VFXParameter>().FirstOrDefault(p => p.exposedName == BufferName);
            var updates = graph.children.OfType<VFXBasicUpdate>().ToArray();
            var exporting = updates.Where(u => emits != null ? emits(u) : Emits(u, out _)).ToArray();
            if (exporting.Length == 0 && buffer == null)
                return 0;
            if (buffer == null)
            {
                buffer = ScriptableObject.CreateInstance<VFXParameter>();
                buffer.Init(typeof(GraphicsBuffer));
                buffer.SetSettingValue("m_ExposedName", BufferName);
                buffer.SetSettingValue("m_Exposed", true);
                buffer.position = new Vector2(-600, -200);
                graph.AddChild(buffer);
            }
            int count = 0;
            foreach (var update in updates)
            {
                var block = update.children.OfType<CustomHLSL>().FirstOrDefault(b => (string)b.GetSettingValue("m_BlockName") == BlockName);
                if (block != null && !block.enabled)
                    continue; // switched off by hand in the graph: that system stays out
                if (!exporting.Contains(update))
                {
                    if (block != null)
                        update.RemoveChild(block);
                    continue;
                }
                if (block == null)
                {
                    block = ScriptableObject.CreateInstance<CustomHLSL>();
                    block.SetSettingValue("m_BlockName", BlockName);
                    update.AddChild(block);
                }
                block.SetSettingValue("m_HLSLCode", Code);
                // Custom HLSL inputs are named after the function's parameters, with a leading underscore.
                var bufferSlot = block.inputSlots.First(s => s.name.TrimStart('_') == BufferName);
                if (!bufferSlot.HasLink())
                    bufferSlot.Link(buffer.outputSlots[0]);
                block.inputSlots.First(s => s.name.TrimStart('_') == "Intensity").value = intensity * OutputScale(update);
                block.inputSlots.First(s => s.name.TrimStart('_') == "ExposureWeight").value = ExposureWeight(update);
                ++count;
            }
            if (!updates.Any(u => u.children.OfType<CustomHLSL>().Any(b => (string)b.GetSettingValue("m_BlockName") == BlockName)))
                graph.RemoveChild(buffer);
            resource.WriteAssetWithSubAssets();
            AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
            return count;
        }

        // The test scene's effect: VFX Graph's Simple Loop with glowing orange particles that export their light.
        // Writes it to path (inside a project folder or an editable package).
        public static void CreateTestEffect(string path, float glow)
        {
            File.Copy(FileUtil.GetPhysicalPath(VisualEffectAssetEditorUtility.templatePath + "02_Simple_Loop.vfx"),
                FileUtil.GetPhysicalPath(path), true);
            AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceUpdate);
            var resource = VisualEffectResource.GetResourceAtPath(path);
            var graph = resource.GetOrCreateGraph();
            foreach (var initialize in graph.children.OfType<VFXBasicInitialize>())
            {
                var block = ScriptableObject.CreateInstance<CustomHLSL>();
                block.SetSettingValue("m_BlockName", "Glow");
                block.SetSettingValue("m_HLSLCode", @"
void Glow(inout VFXAttributes attributes, float Brightness)
{
    attributes.color = float3(1.0, 0.4, 0.12) * Brightness;
}");
                initialize.AddChild(block);
                block.inputSlots.First(s => s.name.TrimStart('_') == "Brightness").value = glow;
            }
            resource.WriteAssetWithSubAssets();
            Add(path);
        }

        // Batch mode, -executeMethod RoyalTracer.Editor.Vfx.RoyalTracerVfxLightExport.<method>:
        //   CreateTestEffectBatch -rtEffectPath <asset path> [-rtGlow <screen value>]
        //   ReportBatch -rtVfxPaths <asset path>|<asset path>...   (what Add would export)
        //   AddBatch -rtVfxPaths <asset path>|<asset path>... [-rtIntensity 1]
        static void CreateTestEffectBatch()
        {
            CreateTestEffect(Arg("-rtEffectPath", "Assets/RoyalTracerLightExport.vfx"), FloatArg("-rtGlow", 6f));
            EditorApplication.Exit(0);
        }

        static void ReportBatch()
        {
            foreach (var path in Arg("-rtVfxPaths", "").Split('|', System.StringSplitOptions.RemoveEmptyEntries))
                Debug.Log($"[RoyalTracer] {path}\n{Report(path)}");
            EditorApplication.Exit(0);
        }

        static void AddBatch()
        {
            foreach (var path in Arg("-rtVfxPaths", "").Split('|', System.StringSplitOptions.RemoveEmptyEntries))
                Debug.Log($"[RoyalTracer] {path}: light export in {Add(path, FloatArg("-rtIntensity", 1f))} system(s)\n{Report(path)}");
            EditorApplication.Exit(0);
        }

        static string Arg(string name, string fallback)
        {
            var args = System.Environment.GetCommandLineArgs();
            int i = System.Array.IndexOf(args, name);
            return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
        }

        static float FloatArg(string name, float fallback) =>
            float.Parse(Arg(name, fallback.ToString(System.Globalization.CultureInfo.InvariantCulture)), System.Globalization.CultureInfo.InvariantCulture);
    }
}
