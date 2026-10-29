using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // HDRP's ocean drives the tracer's: each frame the water surface's wave bands, which HDRP simulates on the GPU, are
    // copied into textures shared with the tracer, along with HDRP's band mapping (patch, current, amplitude, distance
    // fade). The tracer traces the heightfield HDRP draws, and path traced cameras stop drawing HDRP's water (HDRP keeps
    // simulating it). HDRP keeps the band data internal, so they are read by reflection; where that fails (another HDRP
    // version) the tracer keeps its flat stand-in plane and HDRP its water.
    sealed class WaterTracker
    {
        const int MaxBands = 3;
        const int FormatRGBA16F = 10; // DXGI_FORMAT_R16G16B16A16_FLOAT, as HDRP's displacement buffer
        const int FormatRG16F = 34, FormatR16F = 54;

        const BindingFlags Any = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        static readonly FieldInfo SimulationField = typeof(WaterSurface).GetField("simulation", Any);
        static FieldInfo simulationTime, activeBands, spectrumField, renderingField, gpuBuffers, displacementBuffer;
        static FieldInfo patchSizes, patchOrientation, patchWindSpeed, amplitudes, currentSpeeds, fadeA, fadeB;
        static MethodInfo maxAmplitude;
        static string missing; // what Bind could not find
        static readonly bool Bound = Bind();

        WaterSurface surface;
        readonly int[] ids = { -1, -1, -1 };
        readonly Texture2D[] bands = new Texture2D[MaxBands];
        int resolution;
        bool warned;

        struct Taken
        {
            public WaterOverlay overlay;
            public int height, foam; // regions in the atlases
        }
        readonly List<WaterOverlay> overlays = new();
        readonly Taken[] taken = new Taken[WaterOverlays.Max];
        readonly Native.WaterOverlay[] native = new Native.WaterOverlay[WaterOverlays.Max];
        readonly OverlayAtlas heightAtlas = new(FormatR16F, TextureFormat.RHalf, GraphicsFormat.R16_SFloat, "RoyalTracer water heights");
        readonly OverlayAtlas foamAtlas = new(FormatRG16F, TextureFormat.RGHalf, GraphicsFormat.R16G16_SFloat, "RoyalTracer water foam");

        // The tracer's ocean shows HDRP's waves this frame.
        public bool Active { get; private set; }

        // Overlays the tracer's ocean took this frame.
        public int Overlays { get; private set; }

        static bool Bind()
        {
            try
            {
                var simulation = SimulationField.FieldType;
                simulationTime = simulation.GetField("simulationTime", Any);
                activeBands = simulation.GetField("numActiveBands", Any);
                spectrumField = simulation.GetField("spectrum", Any);
                renderingField = simulation.GetField("rendering", Any);
                gpuBuffers = simulation.GetField("gpuBuffers", Any);
                displacementBuffer = gpuBuffers.FieldType.GetField("displacementBuffer", Any);
                var spectrum = spectrumField.FieldType;
                patchSizes = spectrum.GetField("patchSizes", Any);
                patchOrientation = spectrum.GetField("patchOrientation", Any);
                patchWindSpeed = spectrum.GetField("patchWindSpeed", Any);
                var rendering = renderingField.FieldType;
                amplitudes = rendering.GetField("patchAmplitudeMultiplier", Any);
                currentSpeeds = rendering.GetField("patchCurrentSpeed", Any);
                fadeA = rendering.GetField("patchFadeA", Any);
                fadeB = rendering.GetField("patchFadeB", Any);
                maxAmplitude = typeof(HDRenderPipeline).Assembly.GetType("UnityEngine.Rendering.HighDefinition.WaterSystem")?
                    .GetMethod("EvaluateMaxAmplitude", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
                var members = new (string name, object value)[]
                {
                    ("simulationTime", simulationTime), ("numActiveBands", activeBands), ("displacementBuffer", displacementBuffer),
                    ("patchSizes", patchSizes), ("patchOrientation", patchOrientation), ("patchWindSpeed", patchWindSpeed),
                    ("patchAmplitudeMultiplier", amplitudes), ("patchCurrentSpeed", currentSpeeds), ("patchFadeA", fadeA),
                    ("patchFadeB", fadeB), ("WaterSystem.EvaluateMaxAmplitude", maxAmplitude),
                };
                missing = string.Join(", ", Array.FindAll(members, m => m.value == null).Select(m => m.name));
                return missing.Length == 0;
            }
            catch (Exception e)
            {
                missing = e.Message;
                return false;
            }
        }

        public void Dispose()
        {
            ReleaseTextures();
            heightAtlas.Release();
            foamAtlas.Release();
            Active = false;
        }

        // The loaded scenes' first active water surface.
        public void Scan()
        {
            if (surface && surface.isActiveAndEnabled)
                return;
            surface = null;
            foreach (var candidate in UnityEngine.Object.FindObjectsByType<WaterSurface>())
                if (candidate.isActiveAndEnabled)
                {
                    surface = candidate;
                    break;
                }
            if (surface && !Bound && !warned)
            {
                warned = true;
                Debug.LogWarning($"[RoyalTracer] This HDRP version's water data could not be read ({missing}): HDRP draws the water, the tracer a flat stand-in.");
            }
        }

        // Before the frame's render event: copies the surface's bands and the frame's overlays into the shared textures
        // and hands their mapping to the tracer, or hides the tracer's ocean.
        public void Record(CommandBuffer cmd, ulong ticket)
        {
            var desc = new Native.WaterDesc();
            Active = Bound && surface && surface.isActiveAndEnabled && Read(cmd, ref desc);
            if (!Active)
                desc = new Native.WaterDesc();
            Native.RT_SetWater(ref desc);
            Overlays = 0;
            if (Active)
                RecordOverlays(cmd, ticket);
        }

        void RecordOverlays(CommandBuffer cmd, ulong ticket)
        {
            overlays.Clear();
            AddHdrpOverlays();
            WaterOverlays.Gather(overlays);
            heightAtlas.Clear();
            foamAtlas.Clear();
            int count = 0;
            foreach (var o in overlays)
            {
                if (count == WaterOverlays.Max)
                    break;
                bool height = Usable(o.height, o.heightSlice, o.heightTexels), foam = Usable(o.foam, o.foamSlice, o.foamTexels);
                if (o.size.x <= 0f || o.size.y <= 0f || !(height || foam))
                    continue;
                taken[count++] = new Taken
                {
                    overlay = o,
                    height = height ? heightAtlas.Add(o.height, o.heightSlice, o.heightTexels) : -1,
                    foam = foam ? foamAtlas.Add(o.foam, o.foamSlice, o.foamTexels) : -1,
                };
            }
            bool heights = heightAtlas.Upload(cmd), foams = foamAtlas.Upload(cmd);
            for (int i = 0; i < count; ++i)
            {
                var t = taken[i];
                var h = heights ? heightAtlas.Region(t.height) : Vector4.zero;
                var f = foams ? foamAtlas.Region(t.foam) : Vector4.zero;
                native[i] = new Native.WaterOverlay
                {
                    originX = t.overlay.origin.x, originZ = t.overlay.origin.y,
                    sizeX = t.overlay.size.x, sizeZ = t.overlay.size.y,
                    heightX = h.x, heightY = h.y, heightW = h.z, heightH = h.w,
                    foamX = f.x, foamY = f.y, foamW = f.z, foamH = f.w,
                    heightScale = t.overlay.heightScale, foamScale = t.overlay.foamScale, maxHeight = t.overlay.maxHeight,
                    foamMax = t.overlay.foamMax,
                };
            }
            Native.RT_SetWaterOverlays(ticket, heightAtlas.Id, foamAtlas.Id, (uint)count, native);
            Overlays = count;
        }

        // HDRP's own overlays: its water deformers and foam generators, over the surface's decal region.
        void AddHdrpOverlays()
        {
            var deformation = surface.GetDeformationBuffer();
            var foam = surface.foam ? surface.GetFoamBuffer(out _) : null;
            if (!deformation && !foam)
                return;
            surface.GetDecalRegion(out var center, out var size);
            overlays.Add(new WaterOverlay
            {
                origin = new Vector2(center.x - 0.5f * size.x, center.y - 0.5f * size.y),
                size = new Vector2(size.x, size.y),
                height = deformation,
                heightTexels = deformation ? new RectInt(0, 0, deformation.width, deformation.height) : default,
                heightScale = 1f,
                foam = foam,
                foamTexels = foam ? new RectInt(0, 0, foam.width, foam.height) : default,
                foamScale = 2f, // HDRP's 0..1 cover: 1 shows as 86 %
            });
        }

        static bool Usable(Texture texture, int slice, RectInt texels) =>
            texture && slice >= 0 && texels.width > 0 && texels.height > 0 && texels.xMin >= 0 && texels.yMin >= 0 &&
            texels.xMax <= texture.width && texels.yMax <= texture.height;

        bool Read(CommandBuffer cmd, ref Native.WaterDesc desc)
        {
            var simulation = SimulationField.GetValue(surface);
            if (simulation == null)
                return false;
            int count = Mathf.Min((int)activeBands.GetValue(simulation), MaxBands);
            var gpu = gpuBuffers.GetValue(simulation);
            var source = gpu != null ? (displacementBuffer.GetValue(gpu) as RTHandle)?.rt : null;
            if (count == 0 || !source)
                return false;
            if (source.width != resolution || !bands[0])
                CreateTextures(source.width);
            if (ids[0] < 0)
                return false;

            object spectrum = spectrumField.GetValue(simulation), rendering = renderingField.GetValue(simulation);
            var size = (Vector4)patchSizes.GetValue(spectrum);
            var orientation = (Vector4)patchOrientation.GetValue(spectrum);
            var wind = (Vector4)patchWindSpeed.GetValue(spectrum);
            var amplitude = (Vector4)amplitudes.GetValue(rendering);
            var current = (Vector4)currentSpeeds.GetValue(rendering);
            var a = (Vector4)fadeA.GetValue(rendering);
            var b = (Vector4)fadeB.GetValue(rendering);
            float time = (float)simulationTime.GetValue(simulation);
            // HDRP samples in water space: the surface's position shifts the pattern (its rotation is not followed).
            var position = surface.transform.position;

            desc.bandCount = (uint)count;
            desc.resolution = (uint)resolution;
            desc.seaLevel = position.y;
            for (int i = 0; i < count; ++i)
            {
                cmd.CopyTexture(source, i, 0, bands[i], 0, 0);
                float invPatch = 1f / size[i], angle = orientation[i] * Mathf.Deg2Rad, travel = current[i] * time * invPatch;
                desc.Set(i, new Native.WaterBand
                {
                    texture = ids[i], invPatchSize = invPatch,
                    offsetU = Mathf.Cos(angle) * travel + position.x * invPatch,
                    offsetV = Mathf.Sin(angle) * travel + position.z * invPatch,
                    amplitude = amplitude[i], fadeA = a[i], fadeB = b[i],
                });
                // HDRP's bound for one band (wind in km/h), times its multiplier.
                desc.maxWaveHeight += amplitude[i] * (float)maxAmplitude.Invoke(null, new object[] { size[i], wind[i] * 3.6f });
            }

            // HDRP's water body as coefficients: its extinction, split so that deep water reflects the scattering colour
            // (a semi-infinite isotropic medium reflects R = (1 - s) / (1 + s), s = sqrt(1 - albedo)).
            var absorption = new float[3];
            var scattering = new float[3];
            float perMeter = -Mathf.Log(0.02f) / Mathf.Max(surface.absorptionDistance, 1e-3f);
            for (int c = 0; c < 3; ++c)
            {
                float extinction = perMeter * Mathf.Max(1f - surface.refractionColor[c], 0.01f);
                float reflectance = Mathf.Clamp(surface.scatteringColor[c], 0f, 0.95f);
                float s = (1f - reflectance) / (1f + reflectance), albedo = 1f - s * s;
                scattering[c] = albedo * extinction;
                absorption[c] = (1f - albedo) * extinction;
            }
            (desc.absorptionR, desc.absorptionG, desc.absorptionB) = (absorption[0], absorption[1], absorption[2]);
            (desc.scatteringR, desc.scatteringG, desc.scatteringB) = (scattering[0], scattering[1], scattering[2]);
            // Whitecaps: HDRP's wind, and its foam amount relative to its default (0.3).
            desc.windSpeed = wind[0];
            desc.foamCoverage = surface.foam ? surface.simulationFoamAmount / 0.3f : 0f;
            return true;
        }

        void CreateTextures(int size)
        {
            ReleaseTextures();
            resolution = size;
            for (int i = 0; i < MaxBands; ++i)
            {
                ids[i] = Native.RT_CreateHostTexture((uint)size, (uint)size, FormatRGBA16F, out var texture);
                if (ids[i] < 0)
                {
                    Debug.LogWarning("[RoyalTracer] The tracer cannot take HDRP's water: " + Native.LastError);
                    ReleaseTextures();
                    return;
                }
                bands[i] = Texture2D.CreateExternalTexture(size, size, TextureFormat.RGBAHalf, false, true, texture);
            }
        }

        void ReleaseTextures()
        {
            for (int i = 0; i < MaxBands; ++i)
            {
                if (ids[i] >= 0)
                    Native.RT_ReleaseHostTexture(ids[i]);
                ids[i] = -1;
                if (bands[i])
                    CoreUtils.Destroy(bands[i]);
                bands[i] = null;
            }
            resolution = 0;
        }
    }

    // A texture shared with the tracer that the frame's overlay regions are packed into (shelves, tallest first),
    // grown to fit and copied every frame; regions in another format go through a blit into this one.
    sealed class OverlayAtlas
    {
        const int MinSize = 256, MaxSize = 8192;
        static readonly int ScratchId = Shader.PropertyToID("_RoyalTracerOverlayScratch");

        readonly int dxgiFormat;
        readonly TextureFormat format;
        readonly GraphicsFormat graphicsFormat;
        readonly string name;
        readonly List<(Texture texture, int slice, RectInt texels)> pending = new();
        readonly List<Vector2Int> placed = new();
        readonly List<int> order = new();
        Texture2D atlas;
        int size;
        bool warned;

        public int Id { get; private set; } = -1;

        public OverlayAtlas(int dxgiFormat, TextureFormat format, GraphicsFormat graphicsFormat, string name)
        {
            this.dxgiFormat = dxgiFormat;
            this.format = format;
            this.graphicsFormat = graphicsFormat;
            this.name = name;
        }

        public void Clear() => pending.Clear();

        public int Add(Texture texture, int slice, RectInt texels)
        {
            pending.Add((texture, slice, texels));
            return pending.Count - 1;
        }

        // (x, y, w, h) in texels; zero when not placed.
        public Vector4 Region(int index) => index < 0 || index >= placed.Count ? Vector4.zero
            : new Vector4(placed[index].x, placed[index].y, pending[index].texels.width, pending[index].texels.height);

        // Packs and copies this frame's regions; false when they do not fit (none of them then shows).
        public bool Upload(CommandBuffer cmd)
        {
            placed.Clear();
            if (pending.Count == 0)
                return true;
            int s = Mathf.Max(size, MinSize);
            while (!Pack(s))
                if ((s *= 2) > MaxSize)
                {
                    Warn("its overlays do not fit in " + MaxSize + " x " + MaxSize);
                    placed.Clear();
                    return false;
                }
            if (s != size && !Create(s))
            {
                placed.Clear();
                return false;
            }
            for (int i = 0; i < pending.Count; ++i)
                Copy(cmd, pending[i], placed[i]);
            return true;
        }

        bool Pack(int s)
        {
            order.Clear();
            for (int i = 0; i < pending.Count; ++i)
                order.Add(i);
            order.Sort((a, b) => pending[b].texels.height.CompareTo(pending[a].texels.height));
            placed.Clear();
            for (int i = 0; i < pending.Count; ++i)
                placed.Add(default);
            int x = 0, y = 0, shelf = 0;
            foreach (int i in order)
            {
                var t = pending[i].texels;
                if (x + t.width > s)
                {
                    y += shelf;
                    x = shelf = 0;
                }
                if (t.width > s || y + t.height > s)
                    return false;
                placed[i] = new Vector2Int(x, y);
                x += t.width;
                shelf = Mathf.Max(shelf, t.height);
            }
            return true;
        }

        void Copy(CommandBuffer cmd, (Texture texture, int slice, RectInt texels) r, Vector2Int to)
        {
            var t = r.texels;
            if (r.texture.graphicsFormat == graphicsFormat)
            {
                cmd.CopyTexture(r.texture, r.slice, 0, t.x, t.y, t.width, t.height, atlas, 0, 0, to.x, to.y);
                return;
            }
            cmd.GetTemporaryRT(ScratchId, new RenderTextureDescriptor(t.width, t.height, graphicsFormat, 0));
            var scale = new Vector2((float)t.width / r.texture.width, (float)t.height / r.texture.height);
            var offset = new Vector2((float)t.x / r.texture.width, (float)t.y / r.texture.height);
            cmd.Blit(r.texture, ScratchId, scale, offset, r.slice, 0);
            cmd.CopyTexture(ScratchId, 0, 0, 0, 0, t.width, t.height, atlas, 0, 0, to.x, to.y);
            cmd.ReleaseTemporaryRT(ScratchId);
        }

        bool Create(int s)
        {
            Release();
            Id = Native.RT_CreateHostTexture((uint)s, (uint)s, dxgiFormat, out var texture);
            if (Id < 0)
            {
                Warn(Native.LastError);
                return false;
            }
            atlas = Texture2D.CreateExternalTexture(s, s, format, false, true, texture);
            atlas.name = name;
            size = s;
            return true;
        }

        public void Release()
        {
            if (Id >= 0)
                Native.RT_ReleaseHostTexture(Id);
            Id = -1;
            if (atlas)
                CoreUtils.Destroy(atlas);
            atlas = null;
            size = 0;
        }

        void Warn(string why)
        {
            if (!warned)
                Debug.LogWarning($"[RoyalTracer] The traced water leaves out {name}: {why}");
            warned = true;
        }
    }
}
