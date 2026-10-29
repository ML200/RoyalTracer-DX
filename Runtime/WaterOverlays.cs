using System;
using System.Collections.Generic;
using UnityEngine;

namespace RoyalTracer
{
    // A rectangle of extra height and foam on the path traced ocean: a ship's wake, a water deformer. Its textures are
    // read on the GPU each frame, a texel region of a 2D texture or of one array slice; the region spans the rectangle
    // from origin to origin + size along world x and z (its first texel column at origin.x, its first row at origin.y).
    public struct WaterOverlay
    {
        public Vector2 origin; // world x, z
        public Vector2 size;   // m along world x and z

        public Texture height; // height in red, m (times heightScale); null: none
        public int heightSlice;
        public RectInt heightTexels;
        public float heightScale;
        public float maxHeight; // bound on |height| after the scale, 0 if unknown

        // Whitecaps in red, bubbles in green, as optical thickness: cover 1 - exp(-value * foamScale). null: none
        public Texture foam;
        public int foamSlice;
        public RectInt foamTexels;
        public float foamScale;
        public float foamMax; // cap on the whitecap cover, 0..1 (0: none)
    }

    public static class WaterOverlays
    {
        public const int Max = 32;

        // Raised once per path traced frame, during its rendering: add the frame's overlays. HDRP's own water
        // deformers and foam generators come first; past Max the rest are left out.
        public static event Action<List<WaterOverlay>> Collect;

        static bool warned;

        internal static void Gather(List<WaterOverlay> overlays)
        {
            if (Collect == null)
                return;
            foreach (Action<List<WaterOverlay>> source in Collect.GetInvocationList())
            {
                try
                {
                    source(overlays);
                }
                catch (Exception e)
                {
                    if (!warned)
                        Debug.LogException(e);
                    warned = true;
                }
            }
        }
    }
}
