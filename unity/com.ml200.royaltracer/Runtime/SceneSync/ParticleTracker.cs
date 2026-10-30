using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace RoyalTracer
{
    // Emissive Shuriken particles, trails and lines as emitters in the tracer's light tree, while HDRP keeps drawing
    // them. Each live particle lights the scene like a sphere of its size and colour; a trail or line becomes a chain of
    // spheres along its length, one per width, each carrying its stretch of the line. Both show in reflections. A
    // renderer emits when its material does: unlit shaders, whose colour HDRP writes unexposed (its radiance is that
    // colour over the camera's exposure), and emission in nits. Particles and stretches darker than MinScreenValue on
    // screen are dropped. All sources share a budget of Budget emitters a frame: large systems are thinned, and the
    // kept particles carry the skipped ones' light; long lines become at most MaxPerLine spheres.
    sealed class ParticleTracker
    {
        const int MaxPerSystem = EmitterSet.MaxEmitters;
        const int MaxPerLine = 64;
        const int Budget = 2048, MinPerSource = 16;
        const float MinRadius = 0.01f;
        const float MinScreenValue = 0.02f; // darker on screen does not light the scene noticeably

        sealed class Tracked
        {
            public ParticleSystem system; // null for trails and lines
            public Renderer renderer;
        }

        // What a particle of colour white emits with a material, per unit of its texture's coverage.
        readonly struct Emission
        {
            public readonly Color screen, nits; // unexposed screen value (unlit colour), and radiance
            public Emission(Color screen, Color nits) { this.screen = screen; this.nits = nits; }
            public bool Emits => screen.maxColorComponent > 0f || nits.maxColorComponent > 0f;
        }

        readonly Dictionary<EntityId, Tracked> tracked = new();
        readonly Dictionary<Texture, Color> coverages = new(); // averaged once: a GPU readback each
        static readonly string[] UnlitColors = { "_UnlitColor", "_BaseColor", "_Color", "_TintColor" };
        static readonly string[] ColorMaps = { "_UnlitColorMap", "_BaseColorMap", "_MainTex" };
        readonly MaterialConverter converter = new();
        ParticleSystem.Particle[] particles = new ParticleSystem.Particle[256];
        Vector3[] points = new Vector3[64];
        readonly Native.Emitter[] emitters = new Native.Emitter[Budget];
        readonly EmitterSet set = new(Budget);

        public int Count => tracked.Count;
        public int Emitters { get; private set; }
        int activeSources = 1, perSource = MaxPerSystem;
        float minLight; // tracer units: MinScreenValue at the current exposure

        public void Dispose()
        {
            set.Release();
            tracked.Clear();
            converter.Dispose();
        }

        // Tracks the loaded scenes' particle systems, trails and lines with emissive materials; releases those that
        // went away.
        public void Scan()
        {
            var seen = new HashSet<EntityId>();
            void Track(Component owner, ParticleSystem system, Renderer renderer)
            {
                if (!renderer || !EmissionOf(renderer.sharedMaterial).Emits)
                    return;
                var id = owner.GetEntityId();
                seen.Add(id);
                if (!tracked.ContainsKey(id))
                    tracked[id] = new Tracked { system = system, renderer = renderer };
            }
            foreach (var system in Object.FindObjectsByType<ParticleSystem>())
                Track(system, system, system.TryGetComponent(out ParticleSystemRenderer r) ? r : null);
            foreach (var trail in Object.FindObjectsByType<TrailRenderer>())
                Track(trail, null, trail);
            foreach (var line in Object.FindObjectsByType<LineRenderer>())
                Track(line, null, line);
            List<EntityId> gone = null;
            foreach (var id in tracked.Keys)
                if (!seen.Contains(id))
                    (gone ??= new List<EntityId>()).Add(id);
            if (gone != null)
                foreach (var id in gone)
                    tracked.Remove(id);
        }

        // Once per frame: every system's live particles and every line. exposure: HDRP's exposure multiplier, 0 while
        // unknown.
        public void Update(float exposure)
        {
            int total = 0, active = 0;
            perSource = Mathf.Clamp(Budget / activeSources, MinPerSource, MaxPerSystem);
            minLight = exposure > 0f ? MinScreenValue / exposure * RoyalTracerSystem.UnitScale : 0f;
            foreach (var t in tracked.Values)
            {
                int count = 0;
                if (t.renderer && t.renderer.enabled && t.renderer.gameObject.activeInHierarchy)
                {
                    var e = EmissionOf(t.renderer.sharedMaterial);
                    var perColour = (e.nits + (exposure > 0f ? e.screen / exposure : Color.black)) * RoyalTracerSystem.UnitScale;
                    if (perColour.maxColorComponent > 0f)
                        count = t.system ? Collect(t.system, perColour, total) : CollectLine(t.renderer, perColour, total);
                }
                total += count;
                active += count > 0 ? 1 : 0;
            }
            set.Submit(emitters, total, "Particles, trails and lines");
            Emitters = total;
            activeSources = Mathf.Max(active, 1);
        }

        // The system's live particles as emitters in tracer space, written from emitters[at]; radiance: tracer units per
        // unit of particle colour. Returns how many.
        int Collect(ParticleSystem system, Color radiance, int at)
        {
            int alive = system.particleCount;
            if (alive == 0)
                return 0;
            if (particles.Length < alive)
                particles = new ParticleSystem.Particle[Mathf.NextPowerOfTwo(alive)];
            alive = system.GetParticles(particles, alive);
            int step = (alive + perSource - 1) / perSource;
            var main = system.main;
            Transform space = main.simulationSpace switch
            {
                ParticleSystemSimulationSpace.Local => system.transform,
                ParticleSystemSimulationSpace.Custom => main.customSimulationSpace,
                _ => null,
            };
            float sizeScale = space && main.scalingMode != ParticleSystemScalingMode.Shape ? MaxComponent(space.lossyScale) : 1f;
            int n = 0;
            for (int i = 0; i < alive && n < perSource && at + n < emitters.Length; i += step)
            {
                Color c = ((Color)particles[i].GetCurrentColor(system)).linear;
                var light = radiance * c * c.a;
                if (light.maxColorComponent <= minLight)
                    continue;
                light *= step; // a kept particle carries the light of the skipped ones
                var position = Space.Point(space ? space.TransformPoint(particles[i].position) : particles[i].position);
                float radius = 0.5f * MaxComponent(particles[i].GetCurrentSize3D(system)) * sizeScale;
                emitters[at + n++] = new Native.Emitter
                {
                    x = position.x, y = position.y, z = position.z, radius = Mathf.Max(radius, MinRadius),
                    r = light.r, g = light.g, b = light.b,
                };
            }
            return n;
        }

        // A trail or line as a chain of spheres in tracer space, written from emitters[at], colour and width from its
        // gradient and curve; radiance: tracer units per unit of line colour. Trails run from their head, the end at the
        // emitting object. Returns how many.
        int CollectLine(Renderer renderer, Color radiance, int at)
        {
            int count;
            Gradient gradient;
            AnimationCurve widthCurve;
            float widthMultiplier;
            bool loop = false, local = false, headLast = false;
            switch (renderer)
            {
                case TrailRenderer trail:
                    count = Points(trail.positionCount, trail.GetPositions);
                    (gradient, widthCurve, widthMultiplier) = (trail.colorGradient, trail.widthCurve, trail.widthMultiplier);
                    if (count > 1)
                    {
                        var head = trail.transform.position;
                        headLast = (points[count - 1] - head).sqrMagnitude < (points[0] - head).sqrMagnitude;
                    }
                    break;
                case LineRenderer line:
                    count = Points(line.positionCount, line.GetPositions);
                    (gradient, widthCurve, widthMultiplier) = (line.colorGradient, line.widthCurve, line.widthMultiplier);
                    (loop, local) = (line.loop, !line.useWorldSpace);
                    break;
                default:
                    return 0;
            }
            if (count < 2)
                return 0;
            if (loop)
                points[count++] = points[0];
            if (local)
                for (int i = 0; i < count; ++i)
                    points[i] = renderer.transform.TransformPoint(points[i]);
            if (headLast)
                System.Array.Reverse(points, 0, count);
            float length = 0f;
            for (int i = 1; i < count; ++i)
                length += Vector3.Distance(points[i - 1], points[i]);
            float typicalWidth = Mathf.Max(widthMultiplier * widthCurve.Evaluate(0.5f), 1e-3f);
            int spheres = Mathf.Clamp(Mathf.CeilToInt(length / typicalWidth), 1, Mathf.Min(MaxPerLine, perSource));
            float stretch = length / spheres;
            int n = 0, segment = 1;
            float segmentStart = 0f;
            for (int k = 0; k < spheres && at + n < emitters.Length; ++k)
            {
                float along = (k + 0.5f) * stretch, t = along / Mathf.Max(length, 1e-6f);
                while (segment < count - 1 && segmentStart + Vector3.Distance(points[segment - 1], points[segment]) < along)
                    segmentStart += Vector3.Distance(points[segment - 1], points[segment++]);
                float segmentLength = Vector3.Distance(points[segment - 1], points[segment]);
                var position = Vector3.Lerp(points[segment - 1], points[segment], segmentLength > 0f ? (along - segmentStart) / segmentLength : 0f);
                float width = Mathf.Max(widthMultiplier * widthCurve.Evaluate(t), 1e-3f);
                Color c = gradient.Evaluate(t).linear;
                if ((radiance * c).maxColorComponent * c.a <= minLight)
                    continue;
                // The sphere stands for a stretch-by-width quad; radiance already holds a width-square's light.
                var light = radiance * c * (c.a * stretch / width);
                position = Space.Point(position);
                emitters[at + n++] = new Native.Emitter
                {
                    x = position.x, y = position.y, z = position.z, radius = Mathf.Max(0.5f * width, MinRadius),
                    r = light.r, g = light.g, b = light.b,
                };
            }
            return n;
        }

        // Fills points with a renderer's positions (one spare for a loop); returns how many.
        int Points(int count, System.Func<Vector3[], int> get)
        {
            if (points.Length < count + 1)
                points = new Vector3[Mathf.NextPowerOfTwo(count + 1)];
            return count > 0 ? get(points) : 0;
        }

        Emission EmissionOf(Material m)
        {
            if (!m || !m.shader)
                return default;
            // HDRP's unlit shaders (and unlit Shader Graphs) draw in ForwardOnly; lit ones also have a GBuffer pass.
            bool unlit = m.FindPass("ForwardOnly") >= 0 && m.FindPass("GBuffer") < 0;
            var colorName = unlit ? UnlitColors.FirstOrDefault(m.HasColor) ?? TintOf(m.shader) : null;
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

        // A custom unlit shader's colour: its first HDR colour property, else its first colour property.
        static string TintOf(Shader shader)
        {
            string first = null;
            for (int i = 0; i < shader.GetPropertyCount(); ++i)
            {
                if (shader.GetPropertyType(i) != UnityEngine.Rendering.ShaderPropertyType.Color)
                    continue;
                if ((shader.GetPropertyFlags(i) & UnityEngine.Rendering.ShaderPropertyFlags.HDR) != 0)
                    return shader.GetPropertyName(i);
                first ??= shader.GetPropertyName(i);
            }
            return first;
        }

        static float MaxComponent(Vector3 v) => Mathf.Max(Mathf.Abs(v.x), Mathf.Max(Mathf.Abs(v.y), Mathf.Abs(v.z)));
    }
}
