using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace RoyalTracer
{
    // HDRP/Lit and HDRP/LayeredLit (layer 0) to the tracer's material. The tracer's albedo, RMA and normal textures
    // replace the constants they stand for, so tint, remaps and normal strength are baked into them on the GPU,
    // which also reads textures that are compressed or not marked readable.
    sealed class MaterialConverter
    {
        const int MaxTextureSize = 2048;

        readonly Dictionary<EntityId, int> materials = new();
        readonly Dictionary<(EntityId, int, Vector4, Vector4), int> textures = new();
        readonly Material bake;
        bool warnedOffset;

        public MaterialConverter() => bake = CoreUtils.CreateEngineMaterial(Resources.Load<Shader>("RoyalTracer/TextureBake"));

        public void Reset()
        {
            materials.Clear();
            textures.Clear();
        }

        public void Dispose() => CoreUtils.Destroy(bake);

        // Property suffix of the layer the tracer reads, or null when the material stays raster-shaded.
        public static string Layer(Material m)
        {
            if (!m || !m.shader)
                return null;
            var name = m.shader.name;
            string layer = name is "HDRP/Lit" or "HDRP/LitTessellation" ? ""
                : name is "HDRP/LayeredLit" or "HDRP/LayeredLitTessellation" ? "0" : null;
            // HDRP draws transparents after the composite, so they stay HDRP's.
            if (layer == null || (m.HasProperty("_SurfaceType") && m.GetFloat("_SurfaceType") != 0f))
                return null;
            return layer;
        }

        public int Convert(Material m)
        {
            if (materials.TryGetValue(m.GetEntityId(), out int id))
                return id;
            string l = Layer(m);
            Color baseColor = Get(m, "_BaseColor" + l, Color.white).linear;
            var albedoMap = Tex(m, "_BaseColorMap" + l);
            var maskMap = Tex(m, "_MaskMap" + l);
            var normalMap = Tex(m, "_NormalMap" + l);
            bool alphaTest = Get(m, "_AlphaCutoffEnable", 0f) > 0.5f;
            Vector4 st = m.HasProperty("_BaseColorMap" + l + "_ST") ? m.GetVector("_BaseColorMap" + l + "_ST") : new Vector4(1, 1, 0, 0);
            if (!warnedOffset && (st.z != 0f || st.w != 0f))
            {
                warnedOffset = true;
                Debug.LogWarning($"[RoyalTracer] UV offsets are not supported yet ({m.name}); tiling is.");
            }
            Color emission = Get(m, "_EmissiveColor", Color.black); // HDR, already linear in HDRP

            var desc = new Native.MaterialDesc
            {
                baseR = baseColor.r, baseG = baseColor.g, baseB = baseColor.b, baseA = 1f,
                emissionR = emission.r * RoyalTracerSystem.UnitScale,
                emissionG = emission.g * RoyalTracerSystem.UnitScale,
                emissionB = emission.b * RoyalTracerSystem.UnitScale,
                roughness = 1f - Get(m, "_Smoothness" + l, 0.5f),
                metallic = Get(m, "_Metallic" + l, 0f),
                coat = Get(m, "_CoatMask", 0f),
                coatRoughness = 0.1f,
                ior = 1.5f,
                alphaCutoff = alphaTest ? Get(m, "_AlphaCutoff", 0.5f) : 1f,
                albedoTexture = -1, normalTexture = -1, rmaTexture = -1,
                uvScaleX = st.x, uvScaleY = st.y,
            };
            if (albedoMap)
                desc.albedoTexture = Bake(albedoMap, 0, Native.TextureAlbedo, (Vector4)baseColor, Vector4.zero, true);
            if (maskMap)
            {
                var metallicSmoothness = new Vector4(Get(m, "_MetallicRemapMin" + l, 0f), Get(m, "_MetallicRemapMax" + l, 1f),
                    Get(m, "_SmoothnessRemapMin" + l, 0f), Get(m, "_SmoothnessRemapMax" + l, 1f));
                var ao = new Vector4(Get(m, "_AORemapMin" + l, 0f), Get(m, "_AORemapMax" + l, 1f), 0f, 0f);
                desc.rmaTexture = Bake(maskMap, 1, Native.TextureRma, metallicSmoothness, ao, false);
            }
            if (normalMap)
                desc.normalTexture = Bake(normalMap, 2, Native.TextureNormal,
                    new Vector4(Get(m, "_NormalScale" + l, 1f), 0f, 0f, 0f), Vector4.zero, false);

            id = Native.RT_AddMaterial(ref desc);
            materials[m.GetEntityId()] = id;
            return id;
        }

        int Bake(Texture source, int pass, uint kind, Vector4 a, Vector4 b, bool srgb)
        {
            var key = (source.GetEntityId(), pass, a, b);
            if (textures.TryGetValue(key, out int id))
                return id;
            int w = Mathf.Min(source.width, MaxTextureSize), h = Mathf.Min(source.height, MaxTextureSize);
            var rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32,
                srgb ? RenderTextureReadWrite.sRGB : RenderTextureReadWrite.Linear);
            bake.SetVector("_ParamsA", a);
            bake.SetVector("_ParamsB", b);
            Graphics.Blit(source, rt, bake, pass);
            var request = AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32);
            request.WaitForCompletion();
            RenderTexture.ReleaseTemporary(rt);
            if (request.hasError)
            {
                Debug.LogWarning($"[RoyalTracer] Could not read back {source.name}");
                return -1;
            }
            var pixels = request.GetData<byte>().ToArray();
            var handle = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try
            {
                var desc = new Native.TextureDesc { width = (uint)w, height = (uint)h, kind = kind, rgba8 = handle.AddrOfPinnedObject() };
                id = Native.RT_AddTexture(ref desc);
            }
            finally
            {
                handle.Free();
            }
            textures[key] = id;
            return id;
        }

        static Texture Tex(Material m, string name) => m.HasProperty(name) ? m.GetTexture(name) : null;
        static float Get(Material m, string name, float fallback) => m.HasProperty(name) ? m.GetFloat(name) : fallback;
        static Color Get(Material m, string name, Color fallback) => m.HasProperty(name) ? m.GetColor(name) : fallback;
    }
}
