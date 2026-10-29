using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.VFX;
using UnityEditor.VFX.Block;
using UnityEngine;
using UnityEngine.VFX;

namespace RoyalTracer.Editor.Vfx
{
    // Lets VFX Graph particles light the path-traced scene: a Custom HLSL block at the end of every Update context
    // appends the live particles to the exposed GraphicsBuffer "RoyalTracerEmitters", which RoyalTracer binds and reads
    // back (RoyalTracer.VfxTracker). The block reads colour, alpha and size as they stand after the update: effects that
    // set them only in an output export their Update values. Part of the VFX Graph editor assembly (see the .asmref),
    // whose graph model is internal.
    public static class RoyalTracerVfxLightExport
    {
        public const string BufferName = "RoyalTracerEmitters";
        const string BlockName = "RoyalTracer Light Export";
        const string Code = @"
// Appends this particle to RoyalTracer's light export: a sphere of the particle's size whose radiance (nits) is
// colour * alpha * Intensity. Record: position, radius, radiance, 1 when simulated in local space.
void RoyalTracerLightExport(inout VFXAttributes attributes, RWByteAddressBuffer RoyalTracerEmitters, float Intensity)
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
    uint at = 16u + index * 32u;
    RoyalTracerEmitters.Store4(at, asuint(float4(attributes.position, 0.5 * max(size.x, max(size.y, size.z)))));
    RoyalTracerEmitters.Store4(at + 16u, asuint(float4(radiance, local)));
}";

        [MenuItem("Assets/RoyalTracer/Add Light Export", true)]
        static bool CanAddToSelection() => Selection.objects.OfType<VisualEffectAsset>().Any();

        [MenuItem("Assets/RoyalTracer/Add Light Export")]
        static void AddToSelection()
        {
            foreach (var asset in Selection.objects.OfType<VisualEffectAsset>())
            {
                var path = AssetDatabase.GetAssetPath(asset);
                Debug.Log($"[RoyalTracer] {path}: light export in {Add(path)} update context(s)");
            }
        }

        // Adds the export to the graph at path, or refreshes it; Intensity scales the exported light. Returns the
        // number of Update contexts that export.
        public static int Add(string path, float intensity = 1f)
        {
            var resource = VisualEffectResource.GetResourceAtPath(path);
            var graph = resource.GetOrCreateGraph();
            var buffer = graph.children.OfType<VFXParameter>().FirstOrDefault(p => p.exposedName == BufferName);
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
            foreach (var update in graph.children.OfType<VFXBasicUpdate>())
            {
                var block = update.children.OfType<CustomHLSL>().FirstOrDefault(b => (string)b.GetSettingValue("m_BlockName") == BlockName);
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
                block.inputSlots.First(s => s.name.TrimStart('_') == "Intensity").value = intensity;
                ++count;
            }
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
void Glow(inout VFXAttributes attributes, float Radiance)
{
    attributes.color = float3(1.0, 0.4, 0.12) * Radiance;
}");
                initialize.AddChild(block);
                block.inputSlots.First(s => s.name.TrimStart('_') == "Radiance").value = glow;
            }
            resource.WriteAssetWithSubAssets();
            Add(path);
        }

        // Batch mode: -executeMethod RoyalTracer.Editor.Vfx.RoyalTracerVfxLightExport.CreateTestEffectBatch
        // -rtEffectPath <asset path> [-rtGlow <nits>]
        static void CreateTestEffectBatch()
        {
            var args = System.Environment.GetCommandLineArgs();
            string Arg(string name, string fallback)
            {
                int i = System.Array.IndexOf(args, name);
                return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
            }
            CreateTestEffect(Arg("-rtEffectPath", "Assets/RoyalTracerLightExport.vfx"),
                float.Parse(Arg("-rtGlow", "200"), System.Globalization.CultureInfo.InvariantCulture));
            EditorApplication.Exit(0);
        }
    }
}
