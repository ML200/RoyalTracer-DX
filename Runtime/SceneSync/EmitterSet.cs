using UnityEngine;

namespace RoyalTracer
{
    // One native emitter set for a particle source, grown to fit its emitters; the plugin hides empty sets.
    sealed class EmitterSet
    {
        public const int MaxEmitters = 512; // per source; larger sources are thinned and the kept emitters carry the rest

        int set = -1, capacity;
        bool empty = true;

        public void Submit(Native.Emitter[] emitters, int count, Object owner)
        {
            if (count == 0 && empty)
                return;
            if (count > capacity)
            {
                Release();
                capacity = Mathf.Min(Mathf.NextPowerOfTwo(count), MaxEmitters);
                set = Native.RT_CreateEmitterSet((uint)capacity);
                if (set < 0)
                    Debug.LogWarning($"[RoyalTracer] Particles of '{(owner ? owner.name : "?")}' do not light the scene: {Native.LastError}");
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
