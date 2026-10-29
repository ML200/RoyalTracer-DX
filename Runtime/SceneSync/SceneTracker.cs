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

    // Mirrors the loaded scenes into the tracer's scene stream without rebuilding it: Scan adds the renderers that
    // became eligible and releases those that went away, PushTransforms sends movement every frame.
    // - MeshRenderers, including static batches (their part of the combined mesh, already in world space).
    // - SkinnedMeshRenderers: triangles whose vertices all follow one bone move with that bone as rigid pieces; the
    //   rest form a deforming piece, skinned on the CPU (BakeMesh) whenever its bones moved.
    // - Meshes without Read/Write are read back from the GPU, which takes a few frames.
    // A stream mesh is uploaded once per (mesh, materials, part) and released with its last instance. Renderers whose
    // materials MaterialConverter does not support stay HDRP-shaded.
    sealed class SceneTracker
    {
        const int MaxReadsInFlight = 64;

        sealed class Piece
        {
            public int instance = -1;
            public string key;
            public int bone = -1;        // -1: the renderer's transform, -2: fixed, -3: deforming, else a bone
            public Matrix4x4 placement;  // fixed placement, or the bone's bind pose
            public Matrix4x4 sent;
            public int mesh = -1;        // deforming: the stream mesh, its vertices (source indices) and pose
            public int[] vertices;
            public int pose;
        }

        sealed class Tracked
        {
            public Renderer renderer;
            public Mesh source;
            public readonly List<Piece> pieces = new();
            public bool seen, pending;
        }

        sealed class MeshEntry
        {
            public int id;
            public int users;
        }

        readonly MaterialConverter materials = new();
        readonly Dictionary<EntityId, Tracked> tracked = new();
        readonly Dictionary<string, MeshEntry> meshes = new();
        readonly Dictionary<EntityId, MeshRead> reads = new();
        readonly List<int> dirtyInstances = new();
        readonly List<float> dirtyMatrices = new();
        readonly float[] matrix = new float[12];
        readonly HashSet<Renderer> hidden = new(); // fully traced renderers with glass, which HDRP must not draw too
        readonly HashSet<LODGroup> held = new(); // drawn by HDRP at their full-detail level, the one traced
        readonly List<Vector3> bakedPositions = new(), bakedNormals = new();
        Mesh baked;
        int deformFrame = -1;
        int waterInstance = -1;

        public int Count => tracked.Count;
        public int Pending { get; private set; }

        public void Dispose()
        {
            foreach (var r in hidden)
                if (r)
                    r.forceRenderingOff = false;
            hidden.Clear();
            foreach (var group in held)
                if (group)
                    group.ForceLOD(-1);
            held.Clear();
            if (baked)
                Object.DestroyImmediate(baked);
            materials.Dispose();
        }

        // Brings the stream in line with the scene; cheap when nothing changed. wait: finish every mesh read now.
        public void Scan(Camera camera, bool wait = false)
        {
            foreach (var t in tracked.Values)
                t.seen = false;
            int added = 0;
            foreach (var r in Eligible(camera))
            {
                if (!tracked.TryGetValue(r.GetEntityId(), out var t))
                {
                    t = new Tracked { renderer = r, source = SourceMesh(r), pending = true };
                    tracked[r.GetEntityId()] = t;
                    ++added;
                }
                t.seen = true;
            }

            List<EntityId> gone = null;
            foreach (var (id, t) in tracked)
                if (!t.seen)
                    (gone ??= new List<EntityId>()).Add(id);
            if (gone != null)
                foreach (var id in gone)
                {
                    ReleasePieces(tracked[id]);
                    tracked.Remove(id);
                }

            do
            {
                CompletePending();
                PruneReads();
                if (wait && Pending > 0)
                    foreach (var read in reads.Values)
                        read.Wait();
            } while (wait && Pending > 0);

            UpdateWaterProxy();
            if (RoyalTracerSettings.Instance.verboseLog && (added > 0 || gone != null))
                Debug.Log($"[RoyalTracer] Scene: +{added} -{gone?.Count ?? 0} renderers, {Pending} waiting for mesh data; {Native.Stats}");
        }

        void CompletePending()
        {
            Pending = 0;
            int inFlight = 0;
            foreach (var read in reads.Values)
                inFlight += read.Done ? 0 : 1;
            foreach (var t in tracked.Values)
            {
                if (!t.pending)
                    continue;
                var id = t.source.GetEntityId();
                if (!reads.TryGetValue(id, out var read))
                {
                    if (inFlight >= MaxReadsInFlight)
                    {
                        ++Pending;
                        continue;
                    }
                    reads[id] = read = MeshRead.Start(t.source);
                    inFlight += read.Done ? 0 : 1;
                }
                if (!read.Poll())
                {
                    ++Pending;
                    continue;
                }
                t.pending = false;
                if (read.Data != null)
                    BuildPieces(t, read.Data);
            }
        }

        // Mesh data only lives while renderers wait for it.
        void PruneReads()
        {
            var needed = new HashSet<EntityId>();
            foreach (var t in tracked.Values)
                if (t.pending)
                    needed.Add(t.source.GetEntityId());
            List<EntityId> drop = null;
            foreach (var (id, read) in reads)
                if (read.Done && !needed.Contains(id))
                    (drop ??= new List<EntityId>()).Add(id);
            if (drop != null)
                foreach (var id in drop)
                    reads.Remove(id);
        }

        // Skins deforming pieces whose bones moved (once per frame) and sends their vertices.
        public void UpdateDeforming()
        {
            if (deformFrame == Time.frameCount)
                return;
            deformFrame = Time.frameCount;
            foreach (var t in tracked.Values)
            {
                if (t.renderer is not SkinnedMeshRenderer smr || !smr)
                    continue;
                foreach (var p in t.pieces)
                {
                    if (p.mesh < 0)
                        continue;
                    int pose = Pose(smr);
                    if (pose == p.pose)
                        continue;
                    p.pose = pose;
                    if (!baked)
                        baked = new Mesh { hideFlags = HideFlags.HideAndDontSave };
                    smr.BakeMesh(baked, true);
                    baked.GetVertices(bakedPositions);
                    baked.GetNormals(bakedNormals);
                    if (bakedPositions.Count <= Max(p.vertices)) // meshes without Read/Write do not bake
                    {
                        p.mesh = -1;
                        continue;
                    }
                    if (bakedPositions.Count == 0)
                        continue;
                    var pos = new float[p.vertices.Length * 3];
                    var nrm = new float[p.vertices.Length * 3];
                    bool normals = bakedNormals.Count == bakedPositions.Count;
                    for (int i = 0; i < p.vertices.Length; ++i)
                    {
                        var v = Space.Point(bakedPositions[p.vertices[i]]);
                        var n = normals ? Space.Point(bakedNormals[p.vertices[i]]) : Vector3.up;
                        pos[i * 3] = v.x; pos[i * 3 + 1] = v.y; pos[i * 3 + 2] = v.z;
                        nrm[i * 3] = n.x; nrm[i * 3 + 1] = n.y; nrm[i * 3 + 2] = n.z;
                    }
                    Native.RT_UpdateMeshVertices(p.mesh, pos, nrm);
                }
            }
        }

        static int Max(int[] values)
        {
            int max = -1;
            foreach (int v in values)
                max = Mathf.Max(max, v);
            return max;
        }

        // Changes whenever a bone moves relative to the renderer, or a blend shape changes.
        static int Pose(SkinnedMeshRenderer smr)
        {
            var root = smr.transform.worldToLocalMatrix;
            int hash = smr.sharedMesh.blendShapeCount;
            for (int b = 0; b < smr.sharedMesh.blendShapeCount; ++b)
                hash = hash * 31 + smr.GetBlendShapeWeight(b).GetHashCode();
            foreach (var bone in smr.bones)
                if (bone)
                    hash = hash * 31 + (root * bone.localToWorldMatrix).GetHashCode();
            return hash;
        }

        // Sends every piece that moved since the last frame.
        public void PushTransforms()
        {
            dirtyInstances.Clear();
            dirtyMatrices.Clear();
            foreach (var t in tracked.Values)
            {
                if (!t.renderer)
                    continue;
                foreach (var p in t.pieces)
                {
                    var world = World(t.renderer, p);
                    if (world == p.sent)
                        continue;
                    p.sent = world;
                    Space.Matrix(world, matrix, 0);
                    dirtyInstances.Add(p.instance);
                    dirtyMatrices.AddRange(matrix);
                }
            }
            if (dirtyInstances.Count > 0)
                Native.RT_SetInstanceTransforms(dirtyInstances.Count, dirtyInstances.ToArray(), dirtyMatrices.ToArray());
        }

        static Matrix4x4 World(Renderer r, Piece p)
        {
            if (p.bone == -2)
                return p.placement;
            if (p.bone == -3) // BakeMesh(useScale: true) applies the renderer's scale itself
                return Matrix4x4.TRS(r.transform.position, r.transform.rotation, Vector3.one);
            if (p.bone >= 0 && r is SkinnedMeshRenderer smr && smr.bones[p.bone])
                return smr.bones[p.bone].localToWorldMatrix * p.placement;
            return r.localToWorldMatrix;
        }

        // Batches combined under a root (StaticBatchingUtility.Combine) are in the root's space; Unity keeps that
        // root internal. Scene static batching is in world space and leaves it null.
        static readonly System.Reflection.PropertyInfo StaticBatchRoot = typeof(Renderer).GetProperty(
            "staticBatchRootTransform", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic |
                                        System.Reflection.BindingFlags.Public);

        static Mesh SourceMesh(Renderer r) => r is SkinnedMeshRenderer smr ? smr.sharedMesh : r.GetComponent<MeshFilter>().sharedMesh;

        // Splits a renderer into pieces and uploads each piece's triangles.
        void BuildPieces(Tracked t, MeshData data)
        {
            var r = t.renderer;
            var shared = r.sharedMaterials;
            int firstSubmesh = 0;
            var fixedPlacement = Matrix4x4.identity;
            bool batched = r is MeshRenderer mr && mr.isPartOfStaticBatch;
            if (batched)
            {
                firstSubmesh = ((MeshRenderer)r).subMeshStartIndex;
                if (StaticBatchRoot?.GetValue(r) is Transform root && root)
                    fixedPlacement = root.localToWorldMatrix;
            }

            // Per used submesh: its material ids (one, or a palette's rows), keyed into the stream mesh identity.
            var submeshMaterials = new int[shared.Length][];
            string materialKey = "";
            bool glass = false, all = true;
            for (int i = 0; i < shared.Length; ++i)
            {
                int s = firstSubmesh + i;
                if (s >= data.submeshes.Length || data.submeshes[s] == null)
                    continue;
                var m = shared[i];
                if (MaterialConverter.IsPalette(m))
                    submeshMaterials[i] = materials.ConvertPalette(m);
                else if (MaterialConverter.IsGlass(m))
                {
                    submeshMaterials[i] = new[] { materials.ConvertGlass(m) };
                    glass = true;
                }
                else if (MaterialConverter.Layer(m) != null)
                    submeshMaterials[i] = new[] { materials.Convert(m) };
                else if (MaterialConverter.IsFallback(m))
                    submeshMaterials[i] = new[] { materials.ConvertFallback(m) };
                all &= submeshMaterials[i] != null;
                materialKey += "/" + (submeshMaterials[i] == null ? "-" : submeshMaterials[i][0] + "x" + submeshMaterials[i].Length);
            }

            // Triangles per piece: bone index for rigid skinned triangles, else one piece.
            var groups = new Dictionary<int, List<(int submesh, int tri)>>();
            var smr = r as SkinnedMeshRenderer;
            bool rigidBones = smr && data.boneWeights != null && smr.bones.Length > 0;
            for (int i = 0; i < shared.Length; ++i)
            {
                if (submeshMaterials[i] == null)
                    continue;
                var idx = data.submeshes[firstSubmesh + i];
                for (int tri = 0; tri + 2 < idx.Length; tri += 3)
                {
                    int bone = batched ? -2 : smr ? -3 : -1;
                    if (rigidBones)
                    {
                        bone = RigidBone(data.boneWeights, idx[tri], idx[tri + 1], idx[tri + 2]);
                        if (bone < 0 || bone >= smr.bones.Length)
                            bone = -3;
                    }
                    if (!groups.TryGetValue(bone, out var list))
                        groups[bone] = list = new List<(int, int)>();
                    list.Add((i, tri));
                }
            }

            var meshId = t.source.GetEntityId().ToString();
            foreach (var (bone, triangles) in groups)
            {
                bool deforming = bone == -3;
                // A deforming mesh belongs to its renderer: the vertices are that renderer's pose.
                var key = meshId + materialKey + (batched ? "/sm" + firstSubmesh : "") + "/b" + bone +
                          (deforming ? "/r" + r.GetEntityId() : "");
                var entry = StreamMesh(key, data, triangles, submeshMaterials, firstSubmesh, deforming, out var vertices);
                if (entry == null)
                    continue;
                var piece = new Piece
                {
                    key = key,
                    bone = bone,
                    placement = bone >= 0 ? t.source.bindposes[bone] : fixedPlacement,
                    mesh = deforming ? entry.id : -1,
                    vertices = vertices,
                    pose = deforming ? int.MinValue : 0,
                };
                var world = World(r, piece);
                Space.Matrix(world, matrix, 0);
                piece.instance = Native.RT_CreateInstance(entry.id, matrix);
                piece.sent = world;
                if (piece.instance < 0)
                {
                    Debug.LogWarning("[RoyalTracer] " + Native.LastError);
                    Release(key);
                    continue;
                }
                t.pieces.Add(piece);
            }
            // HDRP draws transparents over the composite; the tracer's glass already shows in it.
            if (glass && all && !rigidBones && t.pieces.Count > 0)
            {
                r.forceRenderingOff = true;
                hidden.Add(r);
            }
        }

        static int RigidBone(BoneWeight[] w, int a, int b, int c)
        {
            int bone = Single(w[a]);
            return bone >= 0 && Single(w[b]) == bone && Single(w[c]) == bone ? bone : -1;
        }

        static int Single(BoneWeight w) => w.weight0 >= 0.999f ? w.boneIndex0 : -1;

        MeshEntry StreamMesh(string key, MeshData data, List<(int submesh, int tri)> triangles, int[][] submeshMaterials,
                             int firstSubmesh, bool deforming, out int[] vertices)
        {
            vertices = null;
            if (meshes.TryGetValue(key, out var entry))
            {
                ++entry.users;
                return entry;
            }
            int id = Upload(data, triangles, submeshMaterials, firstSubmesh, deforming, out vertices);
            if (id < 0)
                return null;
            meshes[key] = entry = new MeshEntry { id = id, users = 1 };
            return entry;
        }

        void ReleasePieces(Tracked t)
        {
            if (hidden.Remove(t.renderer) && t.renderer)
                t.renderer.forceRenderingOff = false;
            foreach (var p in t.pieces)
            {
                Native.RT_ReleaseInstance(p.instance);
                Release(p.key);
            }
            t.pieces.Clear();
        }

        void Release(string key)
        {
            if (!meshes.TryGetValue(key, out var entry) || --entry.users > 0)
                return;
            Native.RT_ReleaseMesh(entry.id);
            meshes.Remove(key);
        }

        IEnumerable<Renderer> Eligible(Camera camera)
        {
            var lowerLods = new HashSet<Renderer>();
            bool hold = RoyalTracerSettings.Instance.fullDetailLods;
            foreach (var group in Object.FindObjectsByType<LODGroup>())
            {
                // The tracer traces each group's first level; HDRP drawing another (or cross-fading) would disagree
                // with it along the outline, where HDRP's own pixels then show.
                if (hold && held.Add(group))
                    group.ForceLOD(0);
                var lods = group.GetLODs();
                var first = lods.Length > 0 ? new HashSet<Renderer>(lods[0].renderers) : new HashSet<Renderer>();
                for (int l = 1; l < lods.Length; ++l)
                    foreach (var r in lods[l].renderers)
                        if (r && !first.Contains(r))
                            lowerLods.Add(r);
            }
            int mask = camera ? camera.cullingMask : ~0;
            foreach (var r in Object.FindObjectsByType<Renderer>())
            {
                if (r is not (MeshRenderer or SkinnedMeshRenderer))
                    continue;
                if (!r.enabled || !r.gameObject.activeInHierarchy || (r.forceRenderingOff && !hidden.Contains(r)) ||
                    r.shadowCastingMode == UnityEngine.Rendering.ShadowCastingMode.ShadowsOnly ||
                    (mask & (1 << r.gameObject.layer)) == 0 || lowerLods.Contains(r))
                    continue;
                var mesh = r is SkinnedMeshRenderer smr ? smr.sharedMesh : r.GetComponent<MeshFilter>()?.sharedMesh;
                if (!mesh)
                    continue;
                bool any = false;
                foreach (var m in r.sharedMaterials)
                    any |= MaterialConverter.Supported(m);
                if (any)
                    yield return r;
            }
        }

        // The listed triangles as a stream mesh in tracer space: referenced vertices only, winding mirrored.
        static int Upload(MeshData data, List<(int submesh, int tri)> triangles, int[][] submeshMaterials, int firstSubmesh,
                          bool deforming, out int[] vertices)
        {
            var order = new List<int>();
            var remap = new Dictionary<int, int>();
            var positions = new List<float>();
            var normals = new List<float>();
            var uvs = data.uvs != null ? new List<float>() : null;
            var indices = new uint[triangles.Count * 3];
            var triangleMaterials = new int[triangles.Count];
            var computedNormals = data.normals ?? ComputeNormals(data);
            for (int t = 0; t < triangles.Count; ++t)
            {
                var (submesh, tri) = triangles[t];
                var idx = data.submeshes[firstSubmesh + submesh];
                var rows = submeshMaterials[submesh];
                int row = rows.Length > 1 && data.colors != null ? Mathf.Clamp(Mathf.RoundToInt(data.colors[idx[tri]].r), 0, rows.Length - 1) : 0;
                triangleMaterials[t] = rows[row];
                for (int k = 0; k < 3; ++k)
                {
                    int v = idx[tri + (k == 0 ? 0 : k == 1 ? 2 : 1)]; // mirrored: swap winding
                    if (!remap.TryGetValue(v, out int nv))
                    {
                        remap[v] = nv = remap.Count;
                        order.Add(v);
                        var p = Space.Point(data.positions[v]);
                        var n = Space.Point(computedNormals[v]);
                        positions.Add(p.x); positions.Add(p.y); positions.Add(p.z);
                        normals.Add(n.x); normals.Add(n.y); normals.Add(n.z);
                        if (uvs != null) { uvs.Add(data.uvs[v].x); uvs.Add(data.uvs[v].y); }
                    }
                    indices[t * 3 + k] = (uint)nv;
                }
            }
            vertices = deforming ? order.ToArray() : null;
            return AddMesh(positions.ToArray(), normals.ToArray(), uvs?.ToArray(), indices, triangleMaterials,
                           deforming ? Native.MeshDeforming : 0u);
        }

        static Vector3[] ComputeNormals(MeshData data)
        {
            var n = new Vector3[data.positions.Length];
            foreach (var idx in data.submeshes)
            {
                if (idx == null)
                    continue;
                for (int t = 0; t + 2 < idx.Length; t += 3)
                {
                    var face = Vector3.Cross(data.positions[idx[t + 1]] - data.positions[idx[t]], data.positions[idx[t + 2]] - data.positions[idx[t]]);
                    n[idx[t]] += face; n[idx[t + 1]] += face; n[idx[t + 2]] += face;
                }
            }
            for (int i = 0; i < n.Length; ++i)
                n[i] = n[i].sqrMagnitude > 0f ? n[i].normalized : Vector3.up;
            return n;
        }

        internal static int AddMesh(float[] positions, float[] normals, float[] uvs, uint[] indices, int[] triangleMaterials,
                           uint flags = 0)
        {
            if (indices.Length == 0)
                return -1;
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
                    flags = flags,
                };
                int id = Native.RT_CreateMesh(ref desc);
                if (id < 0)
                    Debug.LogWarning("[RoyalTracer] " + Native.LastError);
                return id;
            }
            finally
            {
                foreach (var h in handles)
                    if (h.IsAllocated) h.Free();
            }
        }

        // Where the tracer cannot draw HDRP's ocean (RoyalTracerSystem.WaterTraced), a dark dielectric plane at its height
        // gives hulls something to reflect and blocks the lower sky; it never shows, since HDRP draws its water then.
        void UpdateWaterProxy()
        {
            var water = Object.FindAnyObjectByType<WaterSurface>();
            if (!water || RoyalTracerSystem.WaterTraced)
            {
                if (waterInstance >= 0)
                    Native.RT_ReleaseInstance(waterInstance);
                waterInstance = -1;
                return;
            }
            if (waterInstance >= 0)
                return;
            float y = water.transform.position.y, e = 100000f;
            var p = new[] { -e, y, -e, e, y, -e, e, y, e, -e, y, e };
            var n = new[] { 0f, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0 };
            var desc = new Native.MaterialDesc
            {
                baseR = 0.02f, baseG = 0.025f, baseB = 0.03f, baseA = 1f, roughness = 0.08f, ior = 1.33f,
                alphaCutoff = 1f, albedoTexture = -1, normalTexture = -1, rmaTexture = -1, uvScaleX = 1f, uvScaleY = 1f,
            };
            int material = Native.RT_CreateMaterial(ref desc);
            int mesh = AddMesh(p, n, null, new uint[] { 0, 2, 1, 0, 3, 2 }, new[] { material, material });
            if (mesh < 0)
                return;
            waterInstance = Native.RT_CreateInstance(mesh, new float[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 });
            Native.RT_ReleaseMesh(mesh); // lives as long as the instance
        }

        public static float WaterLevel()
        {
            var water = Object.FindAnyObjectByType<WaterSurface>();
            return water ? water.transform.position.y : float.NegativeInfinity;
        }
    }
}
