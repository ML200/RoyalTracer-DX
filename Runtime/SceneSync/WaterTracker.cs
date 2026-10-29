using System;
using System.Linq;
using System.Reflection;
using UnityEngine;
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

        // The tracer's ocean shows HDRP's waves this frame.
        public bool Active { get; private set; }

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

        // Before the frame's render event: copies the surface's bands into the shared textures and hands their mapping to
        // the tracer, or hides the tracer's ocean.
        public void Record(CommandBuffer cmd)
        {
            var desc = new Native.WaterDesc();
            Active = Bound && surface && surface.isActiveAndEnabled && Read(cmd, ref desc);
            if (!Active)
                desc = new Native.WaterDesc();
            Native.RT_SetWater(ref desc);
        }

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
                    UnityEngine.Object.Destroy(bands[i]);
                bands[i] = null;
            }
            resolution = 0;
        }
    }
}
