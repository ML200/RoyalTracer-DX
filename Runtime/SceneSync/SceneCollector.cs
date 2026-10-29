using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer
{
    // Unity is left-handed, the tracer right-handed with the same X right and Y up: negating z maps one onto the
    // other. That mirrors geometry, so triangle winding flips as well.
    static class Space
    {
        public static Vector3 Point(Vector3 v) => new(v.x, v.y, -v.z);

        // S·M·S with S = diag(1, 1, -1), written as a row-major 3x4.
        public static void Matrix(Matrix4x4 m, float[] dst, int offset)
        {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    dst[offset + r * 4 + c] = m[r, c] * (r == 2 ? -1f : 1f) * (c == 2 ? -1f : 1f);
        }
    }

    // Which renderers the tracer sees, and their transforms. A scene snapshot covers the active MeshRenderers whose
    // materials MaterialConverter supports; everything else stays HDRP-shaded (the composite keeps HDRP's pixel
    // wherever the tracer did not see the same surface).
    sealed class SceneCollector
    {
        readonly MaterialConverter materials = new();
        readonly List<(Renderer renderer, int instance, Matrix4x4 sent)> tracked = new();
        readonly List<int> dirtyInstances = new();
        readonly List<float> dirtyMatrices = new();
        int signature;

        public int Signature => signature;

        public void Dispose() => materials.Dispose();

        // Order-independent hash of what a snapshot would contain; a change means the scene must be rebuilt.
        public int ComputeSignature(Camera camera)
        {
            int hash = 0, count = 0;
            foreach (var r in Eligible(camera))
            {
                hash += r.GetEntityId().GetHashCode() * 486187739 + r.sharedMaterials.Length;
                ++count;
            }
            return hash ^ count;
        }

        public bool Commit(Camera camera)
        {
            tracked.Clear();
            materials.Reset();
            Native.RT_SceneBegin();
            var meshes = new Dictionary<string, int>();
            int skippedStatic = 0, skippedUnreadable = 0;
            var unreadable = new HashSet<string>();
            var matrix = new float[12];
            foreach (var r in Eligible(camera))
            {
                if (r.isPartOfStaticBatch) { ++skippedStatic; continue; }
                var mesh = r.GetComponent<MeshFilter>().sharedMesh;
                if (!mesh.isReadable)
                {
                    ++skippedUnreadable;
                    if (unreadable.Count < 8) unreadable.Add(mesh.name);
                    continue;
                }
                var shared = r.sharedMaterials;
                var key = mesh.GetEntityId().ToString();
                var materialIds = new int[shared.Length];
                for (int i = 0; i < shared.Length; ++i)
                {
                    materialIds[i] = i < mesh.subMeshCount && MaterialConverter.Layer(shared[i]) != null ? materials.Convert(shared[i]) : -1;
                    key += "/" + materialIds[i];
                }
                if (!meshes.TryGetValue(key, out int meshId))
                    meshes[key] = meshId = AddMesh(mesh, materialIds);
                if (meshId < 0)
                    continue;
                Space.Matrix(r.localToWorldMatrix, matrix, 0);
                tracked.Add((r, Native.RT_AddInstance(meshId, matrix), r.localToWorldMatrix));
            }
            bool water = AddWaterProxy();
            signature = ComputeSignature(camera);
            if (tracked.Count == 0 && !water)
                return false; // nothing to trace; HDRP renders alone until the scene changes
            if (skippedStatic + skippedUnreadable > 0 && RoyalTracerSettings.Instance.verboseLog)
                Debug.Log($"[RoyalTracer] Left to HDRP: {skippedStatic} statically batched and {skippedUnreadable} unreadable " +
                          $"renderers (e.g. {string.Join(", ", unreadable)}). Traced: {tracked.Count}.");
            if (Native.RT_SceneCommit() != 1)
            {
                Debug.LogError("[RoyalTracer] Scene commit failed: " + Native.LastError);
                return false;
            }
            return true;
        }

        // Sends every tracked renderer that moved since the last frame.
        public void PushTransforms()
        {
            dirtyInstances.Clear();
            dirtyMatrices.Clear();
            var matrix = new float[12];
            for (int i = 0; i < tracked.Count; ++i)
            {
                var (renderer, instance, sent) = tracked[i];
                if (!renderer || instance < 0)
                    continue;
                var world = renderer.localToWorldMatrix;
                if (world == sent)
                    continue;
                tracked[i] = (renderer, instance, world);
                Space.Matrix(world, matrix, 0);
                dirtyInstances.Add(instance);
                dirtyMatrices.AddRange(matrix);
            }
            if (dirtyInstances.Count > 0)
                Native.RT_SetInstanceTransforms(dirtyInstances.Count, dirtyInstances.ToArray(), dirtyMatrices.ToArray());
        }

        static IEnumerable<MeshRenderer> Eligible(Camera camera)
        {
            var lowerLods = new HashSet<Renderer>();
            foreach (var group in Object.FindObjectsByType<LODGroup>())
            {
                var lods = group.GetLODs();
                var first = lods.Length > 0 ? new HashSet<Renderer>(lods[0].renderers) : new HashSet<Renderer>();
                for (int l = 1; l < lods.Length; ++l)
                    foreach (var r in lods[l].renderers)
                        if (r && !first.Contains(r))
                            lowerLods.Add(r);
            }
            int mask = camera ? camera.cullingMask : ~0;
            foreach (var r in Object.FindObjectsByType<MeshRenderer>())
            {
                if (!r.enabled || !r.gameObject.activeInHierarchy || r.forceRenderingOff ||
                    r.shadowCastingMode == UnityEngine.Rendering.ShadowCastingMode.ShadowsOnly ||
                    (mask & (1 << r.gameObject.layer)) == 0 || lowerLods.Contains(r))
                    continue;
                var filter = r.GetComponent<MeshFilter>();
                if (!filter || !filter.sharedMesh)
                    continue;
                bool any = false;
                foreach (var m in r.sharedMaterials)
                    any |= MaterialConverter.Layer(m) != null;
                if (any)
                    yield return r;
            }
        }

        static int AddMesh(Mesh mesh, int[] materialIds)
        {
            var positions = new List<Vector3>();
            var normals = new List<Vector3>();
            var uvs = new List<Vector2>();
            mesh.GetVertices(positions);
            mesh.GetNormals(normals);
            mesh.GetUVs(0, uvs);
            if (normals.Count != positions.Count)
                return -1;
            var indices = new List<uint>();
            var triangleMaterials = new List<int>();
            var submesh = new List<int>();
            for (int s = 0; s < Mathf.Min(mesh.subMeshCount, materialIds.Length); ++s)
            {
                if (materialIds[s] < 0 || mesh.GetTopology(s) != MeshTopology.Triangles)
                    continue;
                mesh.GetIndices(submesh, s);
                for (int t = 0; t + 2 < submesh.Count; t += 3)
                {
                    indices.Add((uint)submesh[t]);
                    indices.Add((uint)submesh[t + 2]); // mirrored: swap winding
                    indices.Add((uint)submesh[t + 1]);
                    triangleMaterials.Add(materialIds[s]);
                }
            }
            if (indices.Count == 0)
                return -1;
            var p = new float[positions.Count * 3];
            var n = new float[positions.Count * 3];
            for (int v = 0; v < positions.Count; ++v)
            {
                var pv = Space.Point(positions[v]);
                var nv = Space.Point(normals[v]);
                p[v * 3] = pv.x; p[v * 3 + 1] = pv.y; p[v * 3 + 2] = pv.z;
                n[v * 3] = nv.x; n[v * 3 + 1] = nv.y; n[v * 3 + 2] = nv.z;
            }
            float[] uv = null;
            if (uvs.Count == positions.Count)
            {
                uv = new float[positions.Count * 2];
                for (int v = 0; v < uvs.Count; ++v) { uv[v * 2] = uvs[v].x; uv[v * 2 + 1] = uvs[v].y; }
            }
            return AddMesh(p, n, uv, indices.ToArray(), triangleMaterials.ToArray());
        }

        static int AddMesh(float[] positions, float[] normals, float[] uvs, uint[] indices, int[] triangleMaterials)
        {
            var handles = new[]
            {
                GCHandle.Alloc(positions, GCHandleType.Pinned), GCHandle.Alloc(normals, GCHandleType.Pinned),
                GCHandle.Alloc(indices, GCHandleType.Pinned), GCHandle.Alloc(triangleMaterials, GCHandleType.Pinned),
                uvs != null ? GCHandle.Alloc(uvs, GCHandleType.Pinned) : default,
            };
            try
            {
                var desc = new Native.MeshDesc
                {
                    positions = handles[0].AddrOfPinnedObject(), normals = handles[1].AddrOfPinnedObject(),
                    uvs = uvs != null ? handles[4].AddrOfPinnedObject() : System.IntPtr.Zero,
                    vertexCount = (uint)(positions.Length / 3),
                    indices = handles[2].AddrOfPinnedObject(), indexCount = (uint)indices.Length,
                    triangleMaterials = handles[3].AddrOfPinnedObject(),
                };
                return Native.RT_AddMesh(ref desc);
            }
            finally
            {
                foreach (var h in handles)
                    if (h.IsAllocated) h.Free();
            }
        }

        // HDRP's ocean is not path traced yet. A dark dielectric plane at its height gives hulls something to
        // reflect and blocks the lower sky; it never shows, since HDRP draws the water over the composite.
        bool AddWaterProxy()
        {
            var water = Object.FindAnyObjectByType<WaterSurface>();
            if (!water)
                return false;
            float y = water.transform.position.y, e = 100000f;
            var p = new[] { -e, y, -e, e, y, -e, e, y, e, -e, y, e };
            var n = new[] { 0f, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 };
            var desc = new Native.MaterialDesc
            {
                baseR = 0.02f, baseG = 0.025f, baseB = 0.03f, baseA = 1f, roughness = 0.08f, ior = 1.33f,
                alphaCutoff = 1f, albedoTexture = -1, normalTexture = -1, rmaTexture = -1, uvScaleX = 1f, uvScaleY = 1f,
            };
            int material = Native.RT_AddMaterial(ref desc);
            int mesh = AddMesh(p, n, null, new uint[] { 0, 2, 1, 0, 3, 2 }, new[] { material, material });
            Native.RT_AddInstance(mesh, new float[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 });
            return true;
        }

        public static float WaterLevel()
        {
            var water = Object.FindAnyObjectByType<WaterSurface>();
            return water ? water.transform.position.y : float.NegativeInfinity;
        }
    }
}
