using UnityEngine;

namespace RoyalTracer
{
    // One native emitter set, grown to fit its emitters up to a limit; the plugin hides empty sets. A tracker pools all
    // its sources into one set: one update a frame, and one light tree over all of them.
    sealed class EmitterSet
    {
        public const int MaxEmitters = 512; // per source; larger sources are thinned and the kept emitters carry the rest

        readonly int limit;
        int set = -1, capacity;
        bool empty = true;

        public EmitterSet(int limit) => this.limit = limit;

        public void Submit(Native.Emitter[] emitters, int count, string owner)
        {
            if (count == 0 && empty)
                return;
            if (count > capacity)
            {
                Release();
                capacity = Mathf.Min(Mathf.NextPowerOfTwo(count), limit);
                set = Native.RT_CreateEmitterSet((uint)capacity);
                if (set < 0)
                    Debug.LogWarning($"[RoyalTracer] {owner} do not light the scene: {Native.LastError}");
            }
            if (set >= 0)
                Native.RT_UpdateEmitterSet(set, emitters, (uint)count);
            empty = count == 0;
        }

        public void Release()
        {
            if (set >= 0)
                Native.RT_ReleaseEmitterSet(set);
            set = -1;
            capacity = 0;
            empty = true;
        }
    }
}
