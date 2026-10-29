using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace RoyalTracer
{
    // HDRP/Lit and HDRP/LayeredLit (layer 0, or one material per layer when vertex colours blend them) to the tracer's
    // material, and Armored Horizon's Ship Palette shader to one tracer material per palette row. The tracer's albedo,
    // RMA and normal textures replace the constants they stand for, so tint, remaps and normal strength are baked into
    // them on the GPU, which also reads textures that are compressed or not marked readable.
    sealed class MaterialConverter
    {
        const int MaxTextureSize = 2048;
        const int PaletteTextureSize = 1024; // a fleet's banks hold hundreds of slices
        public const string PaletteShader = "Fleet/HDRP Ship Palette";

        readonly Dictionary<EntityId, int> materials = new();
        readonly Dictionary<EntityId, int[]> palettes = new();
        readonly Dictionary<EntityId, int[]> layered = new(); // per vertex-blended LayeredLit: one material per layer
        readonly Dictionary<(EntityId, int, int, Vector4, Vector4), int> textures = new();
        readonly Material bake;
        bool warnedOffset;

        public MaterialConverter() => bake = CoreUtils.CreateEngineMaterial(Resources.Load<Shader>("RoyalTracer/TextureBake"));

        public void Reset()
        {
            materials.Clear();
            palettes.Clear();
            layered.Clear();
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

        // Transparent HDRP/Lit (windows, optics): traced as thin glass, and HDRP stops drawing it meanwhile.
        public static bool IsGlass(Material m) =>
            m && m.shader && m.shader.name is "HDRP/Lit" or "HDRP/LitTessellation" &&
            m.HasProperty("_SurfaceType") && m.GetFloat("_SurfaceType") == 1f;

        public static bool Supported(Material m) => Layer(m) != null || IsPalette(m) || IsGlass(m) || IsFallback(m);

        // Any other opaque surface shader (Shader Graph, custom, legacy): traced with its main colour and texture.
        // Unlit and transparent shaders stay HDRP's: the tracer would light or occlude them wrongly.
        public static bool IsFallback(Material m) =>
            m && m.shader && Layer(m) == null && !IsPalette(m) && !IsGlass(m) && m.renderQueue < 2450 &&
            !m.shader.name.Contains("Unlit") && !m.shader.name.StartsWith("Hidden/");

        public int ConvertFallback(Material m)
        {
            if (materials.TryGetValue(m.GetEntityId(), out int id))
                return id;
            var shader = m.shader;
            string colorName = null, textureName = null;
            for (int i = 0; i < shader.GetPropertyCount(); ++i)
            {
                var flags = shader.GetPropertyFlags(i);
                if ((flags & ShaderPropertyFlags.MainColor) != 0) colorName = shader.GetPropertyName(i);
                if ((flags & ShaderPropertyFlags.MainTexture) != 0) textureName = shader.GetPropertyName(i);
            }
            colorName ??= new[] { "_BaseColor", "_Color", "_MainColor", "_Tint" }.FirstOrDefault(m.HasColor);
            textureName ??= new[] { "_BaseColorMap", "_BaseMap", "_MainTex", "_Albedo" }.FirstOrDefault(m.HasTexture);
            Color baseColor = colorName != null ? m.GetColor(colorName).linear : Color.white;
            var albedoMap = textureName != null ? m.GetTexture(textureName) : null;
            Vector2 tiling = textureName != null ? m.GetTextureScale(textureName) : Vector2.one;
            Color emission = m.HasColor("_EmissiveColor") ? m.GetColor("_EmissiveColor")
                : m.HasColor("_EmissionColor") && m.IsKeywordEnabled("_EMISSION") ? m.GetColor("_EmissionColor").linear : Color.black;
            var desc = new Native.MaterialDesc
            {
                baseR = baseColor.r, baseG = baseColor.g, baseB = baseColor.b, baseA = 1f,
                emissionR = emission.r * RoyalTracerSystem.UnitScale,
                emissionG = emission.g * RoyalTracerSystem.UnitScale,
                emissionB = emission.b * RoyalTracerSystem.UnitScale,
                roughness = 1f - Get(m, "_Smoothness", Get(m, "_Glossiness", 0.4f)),
                metallic = Get(m, "_Metallic", 0f),
                ior = 1.5f,
                alphaCutoff = 1f,
                albedoTexture = -1, normalTexture = -1, rmaTexture = -1,
                uvScaleX = tiling.x, uvScaleY = tiling.y,
            };
            if (albedoMap is Texture2D)
                desc.albedoTexture = Bake(albedoMap, -1, 0, Native.TextureAlbedo, (Vector4)baseColor, Vector4.zero, true);
            id = Native.RT_CreateMaterial(ref desc);
            materials[m.GetEntityId()] = id;
            return id;
        }

        public int ConvertGlass(Material m)
        {
            if (materials.TryGetValue(m.GetEntityId(), out int id))
                return id;
            Color c = Get(m, "_BaseColor", Color.white).linear;
            // HDRP blends the surface colour by alpha; transmission is white where the glass is clear.
            var tint = Color.Lerp(Color.white, c, c.a);
            var desc = new Native.MaterialDesc
            {
                baseR = tint.r, baseG = tint.g, baseB = tint.b, baseA = 1f,
                roughness = 1f - Get(m, "_Smoothness", 0.95f),
                ior = Get(m, "_Ior", 1.5f),
                alphaCutoff = 1f,
                albedoTexture = -1, normalTexture = -1, rmaTexture = -1,
                uvScaleX = 1f, uvScaleY = 1f,
                flags = Native.MaterialThinGlass,
            };
            id = Native.RT_CreateMaterial(ref desc);
            materials[m.GetEntityId()] = id;
            return id;
        }

        // HDRP/LayeredLit blending its layers by vertex colour (multiply or add): its layer count, else 0. Its layers are
        // traced per triangle (SceneTracker picks the one HDRP weighs most at the triangle's vertices); a layer mask
        // texture alone would need per-pixel blending, so such materials keep their main layer.
        public static int VertexBlendedLayers(Material m)
        {
            if (Layer(m) != "0" || !m.HasProperty("_VertexColorMode"))
                return 0;
            int mode = Mathf.RoundToInt(m.GetFloat("_VertexColorMode")), count = Mathf.RoundToInt(Get(m, "_LayerCount", 1f));
            return mode is 1 or 2 && count > 1 ? Mathf.Min(count, 4) : 0;
        }

        public static bool VertexColorAdds(Material m) => Mathf.RoundToInt(Get(m, "_VertexColorMode", 0f)) == 2;

        public int[] ConvertLayers(Material m)
        {
            if (layered.TryGetValue(m.GetEntityId(), out var ids))
                return ids;
            ids = new int[VertexBlendedLayers(m)];
            for (int i = 0; i < ids.Length; ++i)
                ids[i] = ConvertLayer(m, i.ToString());
            layered[m.GetEntityId()] = ids;
            return ids;
        }

        public int Convert(Material m)
        {
            if (materials.TryGetValue(m.GetEntityId(), out int id))
                return id;
            id = ConvertLayer(m, Layer(m));
            materials[m.GetEntityId()] = id;
            return id;
        }

        int ConvertLayer(Material m, string l)
        {
            Color baseColor = Get(m, "_BaseColor" + l, Color.white).linear;
            var albedoMap = Tex(m, "_BaseColorMap" + l);
            var maskMap = Tex(m, "_MaskMap" + l);
            var normalMap = Tex(m, "_NormalMap" + l);
            // Planar and triplanar mapping (HDRP _UVBase 4 and 5) project from world space; the tracer samples by
            // UV, so such layers keep their average colour instead of a wrongly mapped texture.
            if (Get(m, "_UVBase" + l, 0f) >= 4f)
            {
                if (albedoMap)
                    baseColor *= AverageColor(albedoMap);
                albedoMap = maskMap = normalMap = null;
            }
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
                desc.albedoTexture = Bake(albedoMap, -1, 0, Native.TextureAlbedo, (Vector4)baseColor, Vector4.zero, true);
            if (maskMap)
            {
                var metallicSmoothness = new Vector4(Get(m, "_MetallicRemapMin" + l, 0f), Get(m, "_MetallicRemapMax" + l, 1f),
                    Get(m, "_SmoothnessRemapMin" + l, 0f), Get(m, "_SmoothnessRemapMax" + l, 1f));
                var ao = new Vector4(Get(m, "_AORemapMin" + l, 0f), Get(m, "_AORemapMax" + l, 1f), 0f, 0f);
                desc.rmaTexture = Bake(maskMap, -1, 1, Native.TextureRma, metallicSmoothness, ao, false);
            }
            if (normalMap)
                desc.normalTexture = Bake(normalMap, -1, 2, Native.TextureNormal,
                    new Vector4(Get(m, "_NormalScale" + l, 1f), 0f, 0f, 0f), Vector4.zero, false);
            return Native.RT_CreateMaterial(ref desc);
        }

        // Mean linear colour of a texture, from a 32x32 downsample.
        Color AverageColor(Texture source) => Average(source, premultiply: false);

        // Colour times alpha, averaged: the light a blended particle texture adds over its quad.
        public Color AveragePremultiplied(Texture source) => Average(source, premultiply: true);

        Color Average(Texture source, bool premultiply)
        {
            var rt = RenderTexture.GetTemporary(32, 32, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            bake.SetVector("_ParamsA", Vector4.one);
            bake.SetVector("_ParamsB", Vector4.zero);
            Graphics.Blit(source, rt, bake, 0);
            var request = AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBAFloat);
            request.WaitForCompletion();
            RenderTexture.ReleaseTemporary(rt);
            if (request.hasError)
                return Color.white;
            var texels = request.GetData<Vector4>();
            var sum = Vector4.zero;
            foreach (var t in texels)
                sum += premultiply ? new Vector4(t.x * t.w, t.y * t.w, t.z * t.w, t.w) : t;
            sum /= texels.Length;
            return new Color(sum.x, sum.y, sum.z, 1f);
        }

        public static bool IsPalette(Material m) => m && m.shader && m.shader.name == PaletteShader;

        // Tracer material per row of the material's _ShipPalette (RGBAFloat, linear, 16 columns: see Armored Horizon's
        // ShipPaletteResources.hlsl, ShipLoadMaterial). A triangle's row is its vertices' COLOR.r. Null when the
        // palette cannot be read.
        public int[] ConvertPalette(Material m)
        {
            if (palettes.TryGetValue(m.GetEntityId(), out var rows))
                return rows;
            var palette = Tex(m, "_ShipPalette");
            if (!palette)
                return palettes[m.GetEntityId()] = null;
            var request = AsyncGPUReadback.Request(palette, 0, TextureFormat.RGBAFloat);
            request.WaitForCompletion();
            if (request.hasError)
            {
                Debug.LogWarning("[RoyalTracer] Could not read the ship palette of " + m.name);
                return palettes[m.GetEntityId()] = null;
            }
            var texels = request.GetData<Vector4>();
            int width = palette.width;
            bool alphaTest = m.IsKeywordEnabled("_ALPHATEST_ON");
            rows = new int[palette.height];
            for (int r = 0; r < rows.Length; ++r)
            {
                Vector4 C(int column) => texels[r * width + column];
                Vector4 baseColor = C(0), st = C(1), c2 = C(2), c3 = C(3), c4 = C(4), c5 = C(5), c6 = C(6), emission = C(8);
                var desc = new Native.MaterialDesc
                {
                    baseR = baseColor.x, baseG = baseColor.y, baseB = baseColor.z, baseA = 1f,
                    emissionR = emission.x * RoyalTracerSystem.UnitScale,
                    emissionG = emission.y * RoyalTracerSystem.UnitScale,
                    emissionB = emission.z * RoyalTracerSystem.UnitScale,
                    roughness = 1f - c2.y,
                    metallic = c2.x,
                    coat = C(11).x,
                    coatRoughness = 0.1f,
                    ior = 1.5f,
                    alphaCutoff = alphaTest && c6.z > 0.5f ? c2.w : 1f,
                    albedoTexture = -1, normalTexture = -1, rmaTexture = -1,
                    uvScaleX = st.x, uvScaleY = st.y,
                };
                if (BankSlice(m, "_ShipBaseBank", c5.x, c5.y, out var bank, out int slice))
                    desc.albedoTexture = Bake(bank, slice, 4, Native.TextureAlbedo, baseColor, Vector4.zero, true);
                if (BankSlice(m, "_ShipMaskBank", c6.x, c6.y, out bank, out slice))
                    desc.rmaTexture = Bake(bank, slice, 5, Native.TextureRma, c3, new Vector4(c4.x, c4.y, 0f, 0f), false);
                if (BankSlice(m, "_ShipNormalBank", c5.z, c5.w, out bank, out slice))
                    desc.normalTexture = Bake(bank, slice, 6, Native.TextureNormal, new Vector4(c2.z, 0f, 0f, 0f), Vector4.zero, false);
                rows[r] = Native.RT_CreateMaterial(ref desc);
            }
            return palettes[m.GetEntityId()] = rows;
        }

        // A palette row's bank reference; a negative bank means none.
        static bool BankSlice(Material m, string prefix, float bank, float slice, out Texture texture, out int index)
        {
            int b = Mathf.RoundToInt(bank);
            texture = b >= 0 ? Tex(m, prefix + b) : null;
            index = Mathf.RoundToInt(slice);
            return texture is Texture2DArray array && index >= 0 && index < array.depth;
        }

        // slice < 0: a 2D texture through pass; otherwise that slice of a Texture2DArray through pass (4-6).
        int Bake(Texture source, int slice, int pass, uint kind, Vector4 a, Vector4 b, bool srgb)
        {
            var key = (source.GetEntityId(), slice, pass, a, b);
            if (textures.TryGetValue(key, out int id))
                return id;
            int maxSize = slice < 0 ? MaxTextureSize : PaletteTextureSize;
            int w = Mathf.Min(source.width, maxSize), h = Mathf.Min(source.height, maxSize);
            var rt = RenderTexture.GetTemporary(w, h, 0, RenderTextureFormat.ARGB32,
                srgb ? RenderTextureReadWrite.sRGB : RenderTextureReadWrite.Linear);
            bake.SetVector("_ParamsA", a);
            bake.SetVector("_ParamsB", b);
            if (slice < 0)
                Graphics.Blit(source, rt, bake, pass);
            else
            {
                bake.SetTexture("_ArrayTex", source);
                bake.SetFloat("_Slice", slice);
                Graphics.Blit(Texture2D.whiteTexture, rt, bake, pass);
            }
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
                id = Native.RT_CreateTexture(ref desc);
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
