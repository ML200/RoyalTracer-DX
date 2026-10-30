using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // HDRP's punctual and area lights as emitters in the tracer's scene: a small sphere for a point light, a disc
    // facing along a spot light (emitting into the spot's cone), the light's own rectangle or disc for an area light. They join the light tree and
    // show in reflections; camera and shadow rays pass through them (HDRP draws the lights on screen). Intensity and
    // colour follow the light every frame; the directional sun is the tracer's sun (RoyalTracerSystem.Frame).
    sealed class LightTracker
    {
        const float MinRadius = 0.05f;          // smaller proxies make radiance spikes the denoiser cannot settle
        const float MaxCandela = 1e6f;          // a stadium floodlight; beyond is a data error
        const int SphereSubdivisions = 1;

        sealed class Proxy
        {
            public Light light;
            public int material = -1, mesh = -1, instance = -1;
            public string shape;
            public Color radiance;
            public bool visible = true;
            public Matrix4x4 sent;
        }

        readonly Dictionary<EntityId, Proxy> proxies = new();
        readonly float[] matrix = new float[12];
        readonly HashSet<EntityId> warned = new();

        public int Count => proxies.Count;

        public void Dispose()
        {
            foreach (var p in proxies.Values)
                Release(p);
            proxies.Clear();
        }

        // Adds new lights, drops gone ones, and rebuilds proxies whose shape changed.
        public void Scan()
        {
            var seen = new HashSet<EntityId>();
            foreach (var light in Object.FindObjectsByType<Light>())
            {
                if (light.type is not (LightType.Point or LightType.Spot or LightType.Rectangle or LightType.Disc))
                    continue;
                seen.Add(light.GetEntityId());
                var shape = Shape(light, out _, out _);
                if (proxies.TryGetValue(light.GetEntityId(), out var p) && p.shape == shape)
                    continue;
                if (p != null)
                    Release(p);
                proxies[light.GetEntityId()] = Create(light, shape);
            }
            List<EntityId> gone = null;
            foreach (var id in proxies.Keys)
                if (!seen.Contains(id))
                    (gone ??= new List<EntityId>()).Add(id);
            if (gone != null)
                foreach (var id in gone)
                {
                    Release(proxies[id]);
                    proxies.Remove(id);
                }
        }

        // Once per frame: placement, on/off, intensity and colour.
        public void Update()
        {
            foreach (var p in proxies.Values)
            {
                if (p.instance < 0 || !p.light)
                    continue;
                var radiance = Radiance(p.light);
                bool visible = p.light.isActiveAndEnabled && radiance.maxColorComponent > 0f;
                if (visible != p.visible)
                {
                    p.visible = visible;
                    Native.RT_SetInstanceVisible(p.instance, visible ? 1 : 0);
                }
                if (!visible)
                    continue;
                var world = Matrix4x4.TRS(p.light.transform.position, p.light.transform.rotation, Vector3.one);
                if (world != p.sent)
                {
                    p.sent = world;
                    Space.Matrix(world, matrix, 0);
                    Native.RT_SetInstanceTransforms(1, new[] { p.instance }, matrix);
                }
                if (Changed(radiance, p.radiance))
                {
                    p.radiance = radiance;
                    var desc = Material(p.light, radiance);
                    Native.RT_UpdateMaterial(p.material, ref desc);
                }
            }
        }

        Proxy Create(Light light, string shape)
        {
            var p = new Proxy { light = light, shape = shape, radiance = Radiance(light) };
            var desc = Material(light, p.radiance);
            p.material = Native.RT_CreateMaterial(ref desc);
            Shape(light, out var kind, out var size);
            Geometry(kind, size, out var positions, out var normals, out var indices);
            var materials = new int[indices.Length / 3];
            for (int i = 0; i < materials.Length; ++i)
                materials[i] = p.material;
            p.mesh = SceneTracker.AddMesh(positions, normals, null, indices, materials);
            if (p.mesh < 0)
                return p;
            p.sent = Matrix4x4.TRS(light.transform.position, light.transform.rotation, Vector3.one);
            Space.Matrix(p.sent, matrix, 0);
            p.instance = Native.RT_CreateInstance(p.mesh, matrix);
            Native.RT_ReleaseMesh(p.mesh); // lives with its instance
            if (p.instance >= 0)
                Native.RT_SetInstanceFlags(p.instance, Native.InstanceEmitter);
            if (RoyalTracerSettings.Instance.verboseLog)
                Debug.Log($"[RoyalTracer] Light '{light.name}': {kind} {size.x:0.###}x{size.y:0.###} m, radiance {p.radiance} (tracer units)");
            return p;
        }

        static void Release(Proxy p)
        {
            if (p.instance >= 0)
                Native.RT_ReleaseInstance(p.instance);
            if (p.material >= 0)
                Native.RT_ReleaseMaterial(p.material);
            p.instance = p.material = -1;
        }

        static Native.MaterialDesc Material(Light light, Color radiance)
        {
            var desc = new Native.MaterialDesc
            {
                baseR = 0f, baseG = 0f, baseB = 0f, baseA = 1f,
                emissionR = radiance.r, emissionG = radiance.g, emissionB = radiance.b,
                roughness = 1f, ior = 1.5f, alphaCutoff = 1f,
                albedoTexture = -1, normalTexture = -1, rmaTexture = -1, uvScaleX = 1f, uvScaleY = 1f,
            };
            if (light.type == LightType.Spot) // HDRP's cone: full intensity inside innerSpotAngle, none past spotAngle
            {
                desc.flags |= Native.MaterialSpotEmission;
                desc.spotCosOuter = Mathf.Clamp01(Mathf.Cos(0.5f * light.spotAngle * Mathf.Deg2Rad));
                desc.spotCosInner = Mathf.Clamp01(Mathf.Cos(0.5f * light.innerSpotAngle * Mathf.Deg2Rad));
            }
            return desc;
        }

        static bool Changed(Color a, Color b) =>
            Mathf.Abs(a.r - b.r) + Mathf.Abs(a.g - b.g) + Mathf.Abs(a.b - b.b) > 0.01f * (b.r + b.g + b.b) + 1e-9f;

        enum Kind { Sphere, Disc, Rectangle }

        // The proxy's shape: kind and size (radius, or width and height), clamped to MinRadius.
        static string Shape(Light light, out Kind kind, out Vector2 size)
        {
            float radius = light.shapeRadius;
            switch (light.type)
            {
                case LightType.Rectangle:
                    kind = Kind.Rectangle;
                    size = Vector2.Max(light.areaSize, Vector2.one * MinRadius);
                    break;
                case LightType.Disc:
                    kind = Kind.Disc;
                    size = Vector2.one * Mathf.Max(light.areaSize.x, MinRadius);
                    break;
                case LightType.Spot:
                    kind = Kind.Disc;
                    size = Vector2.one * Mathf.Max(radius, MinRadius);
                    break;
                default:
                    kind = Kind.Sphere;
                    size = Vector2.one * Mathf.Max(radius, MinRadius);
                    break;
            }
            return kind + ":" + size.x.ToString("R") + "x" + size.y.ToString("R");
        }

        // Emitted radiance in tracer units. HDRP stores Light.intensity in candela for punctual lights and in nits for
        // area lights, whatever unit the inspector shows (lightUnit only picks the display unit).
        Color Radiance(Light light)
        {
            Shape(light, out var kind, out var size);
            float nits;
            if (light.type is LightType.Rectangle or LightType.Disc)
                nits = light.intensity;
            else // punctual: the intensity from a sphere or disc whose projected area is pi r^2
            {
                float candela = light.intensity;
                if (!float.IsFinite(candela) || candela > MaxCandela)
                {
                    if (warned.Add(light.GetEntityId()))
                        Debug.LogWarning($"[RoyalTracer] Light '{light.name}' has an implausible intensity ({candela} cd); traced at {MaxCandela} cd.");
                    candela = MaxCandela;
                }
                nits = candela / (Mathf.PI * size.x * size.x); // a spot's disc keeps this intensity across its cone
            }
            var tint = light.color.linear * (light.useColorTemperature ? Mathf.CorrelatedColorTemperatureToRGB(light.colorTemperature) : Color.white);
            return tint * (Mathf.Max(nits, 0f) * RoyalTracerSystem.UnitScale);
        }

        // Proxy geometry in tracer space (right-handed; Unity's +z forward is -z here), front faces counter-clockwise.
        static void Geometry(Kind kind, Vector2 size, out float[] positions, out float[] normals, out uint[] indices)
        {
            var p = new List<Vector3>();
            var n = new List<Vector3>();
            var idx = new List<uint>();
            void Tri(Vector3 a, Vector3 b, Vector3 c, Vector3 outward)
            {
                if (Vector3.Dot(Vector3.Cross(b - a, c - a), outward) < 0f)
                    (b, c) = (c, b);
                uint at = (uint)p.Count;
                p.Add(a); p.Add(b); p.Add(c);
                bool sphere = kind == Kind.Sphere;
                n.Add(sphere ? a.normalized : outward); n.Add(sphere ? b.normalized : outward); n.Add(sphere ? c.normalized : outward);
                idx.Add(at); idx.Add(at + 1); idx.Add(at + 2);
            }
            var forward = new Vector3(0f, 0f, -1f);
            if (kind == Kind.Sphere)
            {
                var faces = new List<(Vector3, Vector3, Vector3)>();
                var axes = new[] { Vector3.right, Vector3.up, Vector3.forward };
                for (int s = 0; s < 8; ++s)
                    faces.Add(((s & 1) != 0 ? -axes[0] : axes[0], (s & 2) != 0 ? -axes[1] : axes[1], (s & 4) != 0 ? -axes[2] : axes[2]));
                for (int level = 0; level < SphereSubdivisions; ++level)
                {
                    var next = new List<(Vector3, Vector3, Vector3)>();
                    foreach (var (a, b, c) in faces)
                    {
                        Vector3 ab = (a + b).normalized, bc = (b + c).normalized, ca = (c + a).normalized;
                        next.Add((a, ab, ca)); next.Add((ab, b, bc)); next.Add((ca, bc, c)); next.Add((ab, bc, ca));
                    }
                    faces = next;
                }
                foreach (var (a, b, c) in faces)
                    Tri(a * size.x, b * size.x, c * size.x, (a + b + c).normalized);
            }
            else if (kind == Kind.Disc)
            {
                const int Segments = 16;
                for (int s = 0; s < Segments; ++s)
                {
                    float a0 = 2f * Mathf.PI * s / Segments, a1 = 2f * Mathf.PI * (s + 1) / Segments;
                    Tri(Vector3.zero, new Vector3(Mathf.Cos(a0), Mathf.Sin(a0), 0f) * size.x,
                        new Vector3(Mathf.Cos(a1), Mathf.Sin(a1), 0f) * size.x, forward);
                }
            }
            else
            {
                float w = size.x * 0.5f, h = size.y * 0.5f;
                Tri(new Vector3(-w, -h, 0f), new Vector3(w, -h, 0f), new Vector3(w, h, 0f), forward);
                Tri(new Vector3(-w, -h, 0f), new Vector3(w, h, 0f), new Vector3(-w, h, 0f), forward);
            }
            positions = new float[p.Count * 3];
            normals = new float[p.Count * 3];
            for (int i = 0; i < p.Count; ++i)
            {
                positions[i * 3] = p[i].x; positions[i * 3 + 1] = p[i].y; positions[i * 3 + 2] = p[i].z;
                normals[i * 3] = n[i].x; normals[i * 3 + 1] = n[i].y; normals[i * 3 + 2] = n[i].z;
            }
            indices = idx.ToArray();
        }
    }
}
