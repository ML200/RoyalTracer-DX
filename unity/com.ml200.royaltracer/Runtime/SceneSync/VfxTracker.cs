using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.VFX;

namespace RoyalTracer
{
    // VFX Graph particles as emitters. A graph opts in with RoyalTracer's light export block (Assets > RoyalTracer >
    // Add Light Export), which appends its live particles to the exposed GraphicsBuffer "RoyalTracerEmitters" during
    // the particle update. This binds a buffer to every such effect, reads it back once per frame (the data arrives a
    // frame or two later) and feeds the effect's emitter set. Colours come as their outputs draw them: unlit outputs
    // write screen values (exposure bypassed) unless they use an exposure weight; HDRP's exposure turns them into
    // radiance.
    sealed class VfxTracker
    {
        public const string BufferName = "RoyalTracerEmitters";
        const int Capacity = 2048; // records per effect; further particles are counted, the stored ones carry their light
        const int HeaderWords = 4, RecordWords = 8; // header: particles written; record: position, radius, colour, flags
        const float MinScreenValue = 0.02f; // particles darker than this on screen do not light the scene noticeably
        static readonly int BufferId = Shader.PropertyToID(BufferName);
        static readonly uint[] Zero = { 0u };

        sealed class Tracked
        {
            public VisualEffect effect;
            public GraphicsBuffer buffer;
            public readonly Native.Emitter[] latest = new Native.Emitter[EmitterSet.MaxEmitters]; // the last readback's
            public bool reading, released;
            public int emitters, particles;
        }

        readonly Dictionary<EntityId, Tracked> effects = new();
        const int Budget = 2048; // emitters a frame over all effects
        readonly Native.Emitter[] emitters = new Native.Emitter[Budget];
        readonly EmitterSet set = new(Budget);
        int recordedFrame = -1;
        float exposure;

        public int Count => effects.Count;
        public int Emitters { get; private set; }  // in the last readbacks
        public int Particles { get; private set; } // exported in the last readbacks, before thinning and dropping dark ones

        public void Dispose()
        {
            foreach (var t in effects.Values)
                Release(t);
            effects.Clear();
            set.Release();
        }

        // Tracks the loaded scenes' effects that export light; releases those that went away.
        public void Scan()
        {
            var seen = new HashSet<EntityId>();
            foreach (var effect in Object.FindObjectsByType<VisualEffect>())
            {
                if (!effect.HasGraphicsBuffer(BufferId))
                    continue;
                var id = effect.GetEntityId();
                seen.Add(id);
                if (!effects.TryGetValue(id, out var t))
                {
                    t = new Tracked { effect = effect };
                    t.buffer = new GraphicsBuffer(GraphicsBuffer.Target.Raw, HeaderWords + Capacity * RecordWords, sizeof(uint));
                    t.buffer.SetData(Zero);
                    effects[id] = t;
                }
                effect.SetGraphicsBuffer(BufferId, t.buffer); // again on every scan: a reloaded asset resets its properties
            }
            List<EntityId> gone = null;
            foreach (var (id, t) in effects)
                if (!seen.Contains(id))
                {
                    Release(t);
                    (gone ??= new List<EntityId>()).Add(id);
                }
            if (gone != null)
                foreach (var id in gone)
                    effects.Remove(id);
        }

        // In the composite pass, after HDRP's VFX update: reads this frame's particles and clears the buffers for the
        // next update, and hands the emitters of the readbacks that arrived to the tracer. Once per frame, whatever the
        // number of cameras. exposure: HDRP's exposure multiplier, 0 while unknown.
        public void Record(CommandBuffer cmd, float exposure)
        {
            this.exposure = exposure;
            if (Time.frameCount == recordedFrame)
                return;
            recordedFrame = Time.frameCount;
            foreach (var t in effects.Values)
            {
                if (!t.effect)
                    continue;
                if (!t.reading) // one readback in flight per effect; frames in between are dropped
                {
                    t.reading = true;
                    var tracked = t;
                    cmd.RequestAsyncReadback(t.buffer, r => Receive(tracked, r));
                }
                cmd.SetBufferData(t.buffer, Zero, 0, 0, 1);
            }
            int n = 0;
            foreach (var t in effects.Values)
            {
                int take = Mathf.Min(t.emitters, Budget - n);
                System.Array.Copy(t.latest, 0, emitters, n, take);
                n += take;
            }
            set.Submit(emitters, n, "VFX particles");
            Emitters = n;
        }

        void Receive(Tracked t, AsyncGPUReadbackRequest r)
        {
            t.reading = false;
            Particles -= t.particles;
            t.emitters = t.particles = 0;
            if (r.hasError || t.released || !t.effect)
                return;
            var words = r.GetData<uint>();
            var floats = words.Reinterpret<float>(sizeof(uint));
            uint written = words[0];
            t.particles = (int)written;
            Particles += t.particles;
            int stored = (int)Mathf.Min(written, Capacity);
            int step = Mathf.Max(1, (stored + EmitterSet.MaxEmitters - 1) / EmitterSet.MaxEmitters);
            // A kept particle carries the light of the skipped ones and of those past the buffer's capacity.
            float carry = stored > 0 ? step * (float)written / stored * RoyalTracerSystem.UnitScale : 0f;
            var toWorld = t.effect.transform.localToWorldMatrix;
            var scale = t.effect.transform.lossyScale;
            float radiusScale = Mathf.Max(Mathf.Abs(scale.x), Mathf.Max(Mathf.Abs(scale.y), Mathf.Abs(scale.z)));
            int n = 0;
            for (int i = 0; i < stored && n < t.latest.Length; i += step)
            {
                int at = HeaderWords + i * RecordWords;
                var position = new Vector3(floats[at], floats[at + 1], floats[at + 2]);
                float radius = floats[at + 3];
                // flags: 1 = local space, + 2 * exposure weight in 1/255 (0: the colour is a screen value, 1: nits)
                int flags = (int)floats[at + 7];
                float weight = (flags >> 1) / 255f;
                if (exposure <= 0f && weight < 1f)
                    continue; // screen values wait for HDRP's exposure
                // screen = colour * lerp(1, exposure, weight); radiance = screen / exposure
                float screenScale = Mathf.Lerp(1f, exposure, weight);
                if (exposure > 0f && Mathf.Max(floats[at + 4], Mathf.Max(floats[at + 5], floats[at + 6])) * screenScale < MinScreenValue)
                    continue;
                float toNits = weight >= 1f ? 1f : screenScale / exposure;
                if ((flags & 1) != 0) // simulated in the effect's local space
                {
                    position = toWorld.MultiplyPoint3x4(position);
                    radius *= radiusScale;
                }
                position = Space.Point(position);
                t.latest[n++] = new Native.Emitter
                {
                    x = position.x, y = position.y, z = position.z, radius = Mathf.Max(radius, 0.01f),
                    r = floats[at + 4] * toNits * carry, g = floats[at + 5] * toNits * carry, b = floats[at + 6] * toNits * carry,
                };
            }
            t.emitters = n;
        }

        static void Release(Tracked t)
        {
            t.released = true;
            t.emitters = 0;
            if (t.effect && t.effect.HasGraphicsBuffer(BufferId))
                t.effect.SetGraphicsBuffer(BufferId, null);
            t.buffer?.Release();
        }
    }
}
