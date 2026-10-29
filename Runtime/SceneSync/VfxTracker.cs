using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.VFX;

namespace RoyalTracer
{
    // VFX Graph particles as emitters. A graph opts in with RoyalTracer's light export block (Assets > RoyalTracer >
    // Add Light Export), which appends its live particles to the exposed GraphicsBuffer "RoyalTracerEmitters" during
    // the particle update. This binds a buffer to every such effect, reads it back once per frame (the data arrives a
    // frame or two later) and feeds the effect's emitter set. Colours are radiance in nits, as HDRP's VFX outputs
    // treat them.
    sealed class VfxTracker
    {
        public const string BufferName = "RoyalTracerEmitters";
        const int Capacity = 2048; // records per effect; further particles are counted, the stored ones carry their light
        const int HeaderWords = 4, RecordWords = 8; // header: particles written; record: position, radius, radiance, local
        static readonly int BufferId = Shader.PropertyToID(BufferName);
        static readonly uint[] Zero = { 0u };

        sealed class Tracked
        {
            public VisualEffect effect;
            public GraphicsBuffer buffer;
            public readonly EmitterSet set = new();
            public bool reading, released;
            public int emitters;
        }

        readonly Dictionary<EntityId, Tracked> effects = new();
        readonly Native.Emitter[] emitters = new Native.Emitter[EmitterSet.MaxEmitters];
        int recordedFrame = -1;

        public int Count => effects.Count;
        public int Emitters { get; private set; } // in the last readbacks

        public void Dispose()
        {
            foreach (var t in effects.Values)
                Release(t);
            effects.Clear();
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
        // next update. Once per frame, whatever the number of cameras.
        public void Record(CommandBuffer cmd)
        {
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
        }

        void Receive(Tracked t, AsyncGPUReadbackRequest r)
        {
            t.reading = false;
            Emitters -= t.emitters;
            t.emitters = 0;
            if (r.hasError || t.released || !t.effect)
                return;
            var words = r.GetData<uint>();
            var floats = words.Reinterpret<float>(sizeof(uint));
            uint written = words[0];
            int stored = (int)Mathf.Min(written, Capacity);
            int step = Mathf.Max(1, (stored + EmitterSet.MaxEmitters - 1) / EmitterSet.MaxEmitters);
            // A kept particle carries the light of the skipped ones and of those past the buffer's capacity.
            float carry = stored > 0 ? step * (float)written / stored * RoyalTracerSystem.UnitScale : 0f;
            var toWorld = t.effect.transform.localToWorldMatrix;
            var scale = t.effect.transform.lossyScale;
            float radiusScale = Mathf.Max(Mathf.Abs(scale.x), Mathf.Max(Mathf.Abs(scale.y), Mathf.Abs(scale.z)));
            int n = 0;
            for (int i = 0; i < stored && n < emitters.Length; i += step)
            {
                int at = HeaderWords + i * RecordWords;
                var position = new Vector3(floats[at], floats[at + 1], floats[at + 2]);
                float radius = floats[at + 3];
                if (floats[at + 7] > 0.5f) // simulated in the effect's local space
                {
                    position = toWorld.MultiplyPoint3x4(position);
                    radius *= radiusScale;
                }
                position = Space.Point(position);
                emitters[n++] = new Native.Emitter
                {
                    x = position.x, y = position.y, z = position.z, radius = Mathf.Max(radius, 0.01f),
                    r = floats[at + 4] * carry, g = floats[at + 5] * carry, b = floats[at + 6] * carry,
                };
            }
            t.set.Submit(emitters, n, t.effect);
            t.emitters = n;
            Emitters += n;
        }

        static void Release(Tracked t)
        {
            t.released = true;
            t.set.Release();
            if (t.effect && t.effect.HasGraphicsBuffer(BufferId))
                t.effect.SetGraphicsBuffer(BufferId, null);
            t.buffer?.Release();
        }
    }
}
