using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace RoyalTracer
{
    // Emissive Shuriken particles as emitters in the tracer's light tree: each live particle lights the scene like a
    // sphere of its size and colour and shows in reflections, while HDRP keeps drawing the particles. A system emits
    // when its renderer's material does: unlit shaders, whose colour HDRP writes unexposed (its radiance is that colour
    // over the camera's exposure), and emission in nits. Large systems are thinned to MaxPerSystem emitters that carry
    // the skipped particles' light.
    sealed class ParticleTracker
    {
        const int MaxPerSystem = EmitterSet.MaxEmitters;
        const float MinRadius = 0.01f;

        sealed class Tracked
        {
            public ParticleSystem system;
            public ParticleSystemRenderer renderer;
            public readonly EmitterSet set = new();
        }

        // What a particle of colour white emits with a material, per unit of its texture's coverage.
        readonly struct Emission
        {
            public readonly Color screen, nits; // unexposed screen value (unlit colour), and radiance
            public Emission(Color screen, Color nits) { this.screen = screen; this.nits = nits; }
            public bool Emits => screen.maxColorComponent > 0f || nits.maxColorComponent > 0f;
        }

        readonly Dictionary<EntityId, Tracked> systems = new();
        readonly Dictionary<Texture, Color> coverages = new(); // averaged once: a GPU readback each
        static readonly string[] UnlitColors = { "_UnlitColor", "_BaseColor", "_Color", "_TintColor" };
        static readonly string[] ColorMaps = { "_UnlitColorMap", "_BaseColorMap", "_MainTex" };
        readonly MaterialConverter converter = new();
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[256];
        Native.Emitter[] emitters = new Native.Emitter[256];

        public int Count => systems.Count;
        public int Emitters { get; private set; }

        public void Dispose()
        {
            foreach (var t in systems.Values)
                t.set.Release();
            systems.Clear();
            converter.Dispose();
        }

        // Tracks the loaded scenes' particle systems with emissive materials; releases those that went away.
        public void Scan()
        {
            var seen = new HashSet<EntityId>();
            foreach (var system in Object.FindObjectsByType<ParticleSystem>())
            {
                if (!system.TryGetComponent(out ParticleSystemRenderer renderer) || !EmissionOf(renderer.sharedMaterial).Emits)
                    continue;
                var id = system.GetEntityId();
                seen.Add(id);
                if (!systems.ContainsKey(id))
                    systems[id] = new Tracked { system = system, renderer = renderer };
            }
            List<EntityId> gone = null;
            foreach (var (id, t) in systems)
                if (!seen.Contains(id))
                {
                    t.set.Release();
                    (gone ??= new List<EntityId>()).Add(id);
                }
            if (gone != null)
                foreach (var id in gone)
                    systems.Remove(id);
        }

        // Once per frame: every system's live particles. exposure: HDRP's exposure multiplier, 0 while unknown.
        public void Update(float exposure)
        {
            int total = 0;
            foreach (var t in systems.Values)
            {
                int count = 0;
                if (t.system && t.renderer && t.renderer.enabled && t.system.gameObject.activeInHierarchy)
                {
                    var e = EmissionOf(t.renderer.sharedMaterial);
                    var perColour = e.nits + (exposure > 0f ? e.screen / exposure : Color.black);
                    if (perColour.maxColorComponent > 0f)
                        count = Collect(t.system, perColour * RoyalTracerSystem.UnitScale);
                }
                t.set.Submit(emitters, count, t.system);
                total += count;
            }
            Emitters = total;
        }

        // The system's live particles as emitters in tracer space; radiance: tracer units per unit of particle colour.
        int Collect(ParticleSystem system, Color radiance)
        {
            int alive = system.particleCount;
            if (alive == 0)
                return 0;
            if (particles.Length < alive)
                particles = new ParticleSystem.Particle[Mathf.NextPowerOfTwo(alive)];
            alive = system.GetParticles(particles, alive);
            int step = (alive + MaxPerSystem - 1) / MaxPerSystem;
            if (emitters.Length < MaxPerSystem)
                emitters = new Native.Emitter[MaxPerSystem];
            var main = system.main;
            Transform space = main.simulationSpace switch
            {
                ParticleSystemSimulationSpace.Local => system.transform,
                ParticleSystemSimulationSpace.Custom => main.customSimulationSpace,
                _ => null,
            };
            float sizeScale = space && main.scalingMode != ParticleSystemScalingMode.Shape ? MaxComponent(space.lossyScale) : 1f;
            int n = 0;
            for (int i = 0; i < alive && n < MaxPerSystem; i += step)
            {
                Color c = ((Color)particles[i].GetCurrentColor(system)).linear;
                var light = radiance * c * (c.a * step); // a kept particle carries the light of the skipped ones
                if (light.maxColorComponent <= 0f)
                    continue;
                var position = Space.Point(space ? space.TransformPoint(particles[i].position) : particles[i].position);
                float radius = 0.5f * MaxComponent(particles[i].GetCurrentSize3D(system)) * sizeScale;
                emitters[n++] = new Native.Emitter
                {
                    x = position.x, y = position.y, z = position.z, radius = Mathf.Max(radius, MinRadius),
                    r = light.r, g = light.g, b = light.b,
                };
            }
            return n;
        }

        Emission EmissionOf(Material m)
        {
            if (!m || !m.shader)
                return default;
            // HDRP's unlit shaders (and unlit Shader Graphs) draw in ForwardOnly; lit ones also have a GBuffer pass.
            bool unlit = m.FindPass("ForwardOnly") >= 0 && m.FindPass("GBuffer") < 0;
            var colorName = unlit ? UnlitColors.FirstOrDefault(m.HasColor) : null;
            var mapName = ColorMaps.FirstOrDefault(m.HasTexture);
            var map = mapName != null ? m.GetTexture(mapName) : null;
            // The quad's light: the texture's colour times alpha, averaged; a sphere of the particle's size shows
            // pi/4 of the quad.
            var coverage = Color.white;
            if (map && !coverages.TryGetValue(map, out coverage))
                coverages[map] = coverage = converter.AveragePremultiplied(map);
            coverage *= 4f / Mathf.PI;
            Color screen = colorName != null ? m.GetColor(colorName).linear * coverage : Color.black;
            Color nits = m.HasColor("_EmissiveColor") ? m.GetColor("_EmissiveColor") * coverage : Color.black; // linear in HDRP
            return new Emission(screen, nits);
        }

        static float MaxComponent(Vector3 v) => Mathf.Max(Mathf.Abs(v.x), Mathf.Max(Mathf.Abs(v.y), Mathf.Abs(v.z)));
    }
}
