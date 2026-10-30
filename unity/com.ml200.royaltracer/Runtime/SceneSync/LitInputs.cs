using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace RoyalTracer
{
    // A scalar of the surface: a channel of a texture remapped to [min, max], or the constant max without a texture.
    struct Scalar
    {
        public Texture texture;
        public Vector4 channel; // weights that pick the channel: (1, 0, 0, 0) is red
        public float min, max;
        public string property; // where it came from, for the log

        public static Scalar Constant(float value, string property = null) =>
            new Scalar { min = value, max = value, property = property };

        public static Scalar Map(Texture texture, Vector4 channel, float min, float max, string property) =>
            new Scalar { texture = texture, channel = channel, min = min, max = max, property = property };
    }

    // A material described by HDRP/Lit's inputs, which is what the tracer converts. HDRP/Lit (and a LayeredLit layer)
    // is read as it is; any other shader is translated by its properties' flags, names and display names, so that
    // Shader Graph, legacy and custom materials keep their normal, metallic, smoothness, occlusion, cutout and emission
    // where they name them in a recognisable way.
    struct LitInputs
    {
        public static readonly Vector4 R = new(1, 0, 0, 0), G = new(0, 1, 0, 0), A = new(0, 0, 0, 1);

        public Color baseColor;   // linear
        public Texture baseMap;
        public Vector2 tiling;
        public Vector2 offset;
        public bool projected;    // planar or triplanar mapping: world space, not UVs
        public Texture normalMap;
        public float normalScale;
        public Scalar metallic, smoothness, occlusion;
        public bool alphaTest;
        public float alphaCutoff;
        public Color emission;    // linear, HDR
        public Texture emissionMap;
        public float coat;
        public string mapping;    // the properties each input came from, for the log

        public static LitInputs FromLit(Material m, string l)
        {
            var mask = Tex(m, "_MaskMap" + l);
            var st = m.HasProperty("_BaseColorMap" + l + "_ST") ? m.GetVector("_BaseColorMap" + l + "_ST") : new Vector4(1, 1, 0, 0);
            var lit = new LitInputs
            {
                baseColor = Get(m, "_BaseColor" + l, Color.white).linear,
                baseMap = Tex(m, "_BaseColorMap" + l),
                tiling = new Vector2(st.x, st.y),
                offset = new Vector2(st.z, st.w),
                projected = Get(m, "_UVBase" + l, 0f) >= 4f,
                normalMap = Tex(m, "_NormalMap" + l),
                normalScale = Get(m, "_NormalScale" + l, 1f),
                alphaTest = Get(m, "_AlphaCutoffEnable", 0f) > 0.5f,
                alphaCutoff = Get(m, "_AlphaCutoff", 0.5f),
                emission = Get(m, "_EmissiveColor", Color.black), // HDR, already linear in HDRP
                coat = Get(m, "_CoatMask", 0f),
                mapping = "HDRP/Lit",
            };
            // With a mask map HDRP takes metallic, occlusion and smoothness from it through its remaps (projected
            // layers keep the constants: their maps are not UV mapped).
            if (mask && !lit.projected)
            {
                lit.metallic = Scalar.Map(mask, R, Get(m, "_MetallicRemapMin" + l, 0f), Get(m, "_MetallicRemapMax" + l, 1f), "_MaskMap");
                lit.smoothness = Scalar.Map(mask, A, Get(m, "_SmoothnessRemapMin" + l, 0f), Get(m, "_SmoothnessRemapMax" + l, 1f), "_MaskMap");
                lit.occlusion = Scalar.Map(mask, G, Get(m, "_AORemapMin" + l, 0f), Get(m, "_AORemapMax" + l, 1f), "_MaskMap");
            }
            else
            {
                lit.metallic = Scalar.Constant(Get(m, "_Metallic" + l, 0f));
                lit.smoothness = Scalar.Constant(Get(m, "_Smoothness" + l, 0.5f));
                lit.occlusion = Scalar.Constant(1f);
            }
            return lit;
        }

        // ------------------------------------------------------------------------------------------ other shaders

        struct Property
        {
            public int index;
            public string name, words; // words: name and display name, lower case, letters and digits only
            public ShaderPropertyType type;
            public ShaderPropertyFlags flags;
            public string defaultTexture;
        }

        // Words that rule a texture out as the base colour map, and a colour out as the base colour.
        static readonly string[] NotBase =
        {
            "normal", "bump", "mask", "metal", "rough", "smooth", "gloss", "occlusion", "emiss", "glow", "height",
            "parallax", "displace", "detail", "noise", "spec", "flow", "foam", "splat", "cube", "reflect", "lut", "ramp",
            "dissolve", "alpha", "opacity", "fresnel", "rim", "shadow", "fog", "edge", "outline", "subsurface", "scatter",
            "transmission", "sheen", "coat", "thickness", "wetness", "dirt", "damage", "decal", "lightmap", "matcap",
        };

        // Words that rule a number out as a material value (it selects, remaps or switches something instead).
        static readonly string[] NotValue =
            { "channel", "source", "mode", "remap", "map", "texture", "tex", "toggle", "enable", "use", "workflow" };

        public static LitInputs Infer(Material m)
        {
            var shader = m.shader;
            var props = new List<Property>();
            for (int i = 0; i < shader.GetPropertyCount(); ++i)
            {
                var type = shader.GetPropertyType(i);
                if (type == ShaderPropertyType.Texture && shader.GetPropertyTextureDimension(i) != TextureDimension.Tex2D)
                    continue;
                props.Add(new Property
                {
                    index = i,
                    name = shader.GetPropertyName(i),
                    words = Words(shader.GetPropertyName(i)) + " " + Words(shader.GetPropertyDescription(i)),
                    type = type,
                    flags = shader.GetPropertyFlags(i),
                    defaultTexture = type == ShaderPropertyType.Texture ? shader.GetPropertyTextureDefaultName(i) : null,
                });
            }
            var taken = new HashSet<string>();
            var mapping = new List<string>();

            Property? Find(bool texture, string[] words, string[] not = null, System.Func<Property, bool> also = null)
            {
                foreach (var word in words)
                    foreach (var p in props)
                    {
                        bool isTexture = p.type == ShaderPropertyType.Texture;
                        bool isNumber = p.type is ShaderPropertyType.Float or ShaderPropertyType.Range or ShaderPropertyType.Int;
                        if (taken.Contains(p.name) || (texture ? !isTexture : !(isNumber || p.type == ShaderPropertyType.Color)))
                            continue;
                        if (!Matches(p, word) || (not != null && not.Any(p.words.Contains)) || (also != null && !also(p)))
                            continue;
                        taken.Add(p.name);
                        return p;
                    }
                return null;
            }
            Texture Texture(Property? p, string role)
            {
                var t = p.HasValue ? m.GetTexture(p.Value.name) : null;
                if (t)
                    mapping.Add(role + " " + p.Value.name);
                return t;
            }
            float Number(Property? p, float fallback) =>
                !p.HasValue || p.Value.type == ShaderPropertyType.Color ? fallback
                : p.Value.type == ShaderPropertyType.Int ? m.GetInteger(p.Value.name) : m.GetFloat(p.Value.name);
            bool IsColor(Property p) => p.type == ShaderPropertyType.Color;
            bool IsNumber(Property p) => p.type != ShaderPropertyType.Color;

            // Textures, the most specific roles first.
            var normal = Find(true, new[] { "normalmap", "normal", "bump", "nrm", "nml" }, new[] { "detail", "second", "macro" })
                         ?? props.Where(p => p.type == ShaderPropertyType.Texture && p.defaultTexture == "bump" && !taken.Contains(p.name))
                             .Select(p => (Property?)p).FirstOrDefault();
            if (normal.HasValue)
                taken.Add(normal.Value.name);
            var mask = Find(true, new[] { "maskmap", "=mask", "metallicaosmoothness", "masktexture" }, new[] { "detail", "alpha", "layer" });
            var metallicGloss = Find(true, new[] { "metallicglossmap", "metallicgloss", "metallicsmoothness", "metalsmooth" });
            var specGloss = Find(true, new[] { "specglossmap", "specularglossiness", "specgloss" });
            var metallicMap = Find(true, new[] { "metallicmap", "metalnessmap", "metallic", "metalness", "metal" });
            var smoothnessMap = Find(true, new[] { "smoothnessmap", "glossinessmap", "glossmap", "smoothness", "glossiness", "gloss" });
            var roughnessMap = Find(true, new[] { "roughnessmap", "roughness", "rough" });
            var occlusionMap = Find(true, new[] { "occlusionmap", "ambientocclusion", "occlusion", "aomap", "aotex", "aotexture", "=ao" });
            var emissionMap = Find(true, new[] { "emissivecolormap", "emissionmap", "emissivemap", "emissive", "emission", "glow" });
            var baseMap = props.Where(p => (p.flags & ShaderPropertyFlags.MainTexture) != 0 && !taken.Contains(p.name))
                              .Select(p => (Property?)p).FirstOrDefault()
                          ?? Find(true, new[] { "basecolormap", "basemap", "basecolor", "albedo", "diffuse", "maintex", "maintexture",
                              "colormap", "colortexture", "colortex", "basetexture", "basetex", "colour", "color", "texture", "tex" },
                              NotBase, p => p.defaultTexture != "bump");
            if (baseMap.HasValue)
                taken.Add(baseMap.Value.name);

            // Colours and numbers.
            var emissionColor = Find(false, new[] { "emissivecolor", "emissioncolor", "emission", "emissive", "glow" }, null, IsColor);
            var baseColorProp = props.Where(p => (p.flags & ShaderPropertyFlags.MainColor) != 0 && !taken.Contains(p.name))
                                    .Select(p => (Property?)p).FirstOrDefault()
                                ?? Find(false, new[] { "basecolor", "maincolor", "albedocolor", "albedo", "color", "colour", "tint" },
                                    NotBase, IsColor);
            var normalScale = Find(false, new[] { "normalscale", "bumpscale", "normalstrength", "normalintensity", "bumpstrength",
                "normalamount", "normalpower" }, null, IsNumber);
            var metallicValue = Find(false, new[] { "metallic", "metalness", "metal" }, NotValue, IsNumber);
            var glossMapScale = Find(false, new[] { "glossmapscale" }, null, IsNumber);
            var smoothnessValue = Find(false, new[] { "smoothness", "glossiness", "gloss" }, NotValue, IsNumber);
            var roughnessValue = Find(false, new[] { "roughness", "rough" }, NotValue, IsNumber);
            var occlusionStrength = Find(false, new[] { "occlusionstrength", "aostrength", "aointensity", "occlusionintensity" }, null, IsNumber);
            var cutoff = Find(false, new[] { "alphacutoff", "alphaclipthreshold", "cutoff", "clipthreshold", "alphathreshold" }, null, IsNumber);
            var clipToggle = Find(false, new[] { "alphacutoffenable", "alphaclip", "alphatest", "usealphaclip", "alphaclipping" }, null, IsNumber);
            var emissionToggle = Find(false, new[] { "useemission", "emissionenabled", "enableemission", "emissiontoggle" }, null, IsNumber);

            var lit = new LitInputs { baseColor = Color.white, tiling = Vector2.one };
            if (baseColorProp.HasValue)
            {
                lit.baseColor = m.GetColor(baseColorProp.Value.name).linear;
                mapping.Add("colour " + baseColorProp.Value.name);
            }
            lit.baseMap = Texture(baseMap, "base");
            if (lit.baseMap)
            {
                lit.tiling = m.GetTextureScale(baseMap.Value.name);
                lit.offset = m.GetTextureOffset(baseMap.Value.name);
            }
            lit.normalMap = Texture(normal, "normal");
            lit.normalScale = Number(normalScale, 1f);

            float metallicConstant = Number(metallicValue, 0f);
            float smoothnessConstant = smoothnessValue.HasValue ? Number(smoothnessValue, 0.5f)
                : roughnessValue.HasValue ? 1f - Number(roughnessValue, 0.5f) : 0.5f;
            lit.metallic = Scalar.Constant(metallicConstant, metallicValue?.name);
            lit.smoothness = Scalar.Constant(smoothnessConstant, (smoothnessValue ?? roughnessValue)?.name);
            if (Texture(mask, "mask (HDRP layout)") is { } maskTexture)
            {
                lit.metallic = Scalar.Map(maskTexture, R, 0f, 1f, mask.Value.name);
                lit.smoothness = Scalar.Map(maskTexture, A, 0f, 1f, mask.Value.name);
                lit.occlusion = Scalar.Map(maskTexture, G, 0f, 1f, mask.Value.name);
            }
            else
            {
                // Unity's metallic-gloss convention: metallic in red, smoothness in alpha (of the base map when
                // _SmoothnessTextureChannel says so), scaled by _GlossMapScale or URP's _Smoothness.
                float glossScale = Number(glossMapScale, smoothnessValue.HasValue ? smoothnessConstant : 1f);
                bool glossInBaseAlpha = Mathf.RoundToInt(Get(m, "_SmoothnessTextureChannel", 0f)) == 1 && lit.baseMap;
                if (Texture(metallicGloss, "metallic-gloss") is { } mg)
                {
                    lit.metallic = Scalar.Map(mg, R, 0f, 1f, metallicGloss.Value.name);
                    lit.smoothness = Scalar.Map(glossInBaseAlpha ? lit.baseMap : mg, A, 0f, glossScale, metallicGloss.Value.name);
                }
                else if (Texture(specGloss, "specular-gloss") is { } sg)
                    lit.smoothness = Scalar.Map(glossInBaseAlpha ? lit.baseMap : sg, A, 0f, glossScale, specGloss.Value.name);
                else
                {
                    if (Texture(metallicMap, "metallic") is { } metal)
                        lit.metallic = Scalar.Map(metal, R, 0f, 1f, metallicMap.Value.name);
                    if (Texture(smoothnessMap, "smoothness") is { } smooth)
                        lit.smoothness = Scalar.Map(smooth, R, 0f, 1f, smoothnessMap.Value.name);
                    else if (Texture(roughnessMap, "roughness") is { } rough)
                        lit.smoothness = Scalar.Map(rough, R, 1f, 0f, roughnessMap.Value.name);
                }
            }
            float occlusion = Number(occlusionStrength, 1f);
            lit.occlusion = Texture(occlusionMap, "occlusion") is { } ao
                ? Scalar.Map(ao, G, 1f - occlusion, 1f, occlusionMap.Value.name)
                : lit.occlusion.texture ? lit.occlusion : Scalar.Constant(1f);

            // Cutout: an enabled alpha-test keyword or toggle, or the alpha-test queue.
            lit.alphaTest = m.IsKeywordEnabled("_ALPHATEST_ON") || m.IsKeywordEnabled("_ALPHACLIP_ON") ||
                            Number(clipToggle, 0f) > 0.5f ||
                            (m.renderQueue >= (int)RenderQueue.AlphaTest && m.renderQueue < (int)RenderQueue.GeometryLast);
            lit.alphaCutoff = Number(cutoff, 0.5f);
            if (lit.alphaTest)
                mapping.Add("cutout " + (cutoff?.name ?? "0.5"));

            // Emission, unless a keyword or toggle has it off.
            var emissionKeyword = shader.keywordSpace.FindKeyword("_EMISSION");
            bool emissionOff = (emissionKeyword.isValid && !m.IsKeywordEnabled(emissionKeyword)) ||
                               (emissionToggle.HasValue && Number(emissionToggle, 1f) < 0.5f);
            if (emissionColor.HasValue && !emissionOff)
            {
                var c = m.GetColor(emissionColor.Value.name);
                lit.emission = emissionColor.Value.name == "_EmissiveColor" ? c : c.linear; // HDRP's is stored linear
                lit.emissionMap = Texture(emissionMap, "emission");
                mapping.Add("emission " + emissionColor.Value.name);
            }
            lit.mapping = string.Join(", ", mapping);
            return lit;
        }

        // Name or display name in lower case, letters and digits only.
        static string Words(string s) => new(s.ToLowerInvariant().Where(char.IsLetterOrDigit).ToArray());

        // "=word": the property's name, without its leading underscore, is exactly the word; otherwise contains it.
        static bool Matches(Property p, string word) =>
            word[0] == '=' ? Words(p.name) == word.Substring(1) : p.words.Contains(word);

        static Texture Tex(Material m, string name) => m.HasProperty(name) ? m.GetTexture(name) : null;
        static float Get(Material m, string name, float fallback) => m.HasProperty(name) ? m.GetFloat(name) : fallback;
        static Color Get(Material m, string name, Color fallback) => m.HasProperty(name) ? m.GetColor(name) : fallback;
    }
}
