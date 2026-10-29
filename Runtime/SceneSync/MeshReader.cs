using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace RoyalTracer
{
    // A mesh's vertices, colours and triangles on the CPU, in Unity's space.
    sealed class MeshData
    {
        public Vector3[] positions, normals;
        public Vector2[] uvs;            // null when absent
        public Color[] colors;           // null when absent
        public int[][] submeshes;        // triangle-list vertex indices per submesh; null for other topologies
        public BoneWeight[] boneWeights; // readable skinned meshes only, else null
    }

    // Reads a mesh, readable or not: unreadable meshes (Read/Write off, static batches) are read back from their GPU
    // buffers, which completes a few frames later. Done is true once Data is set; Data stays null on failure.
    //
    // The readbacks are taken in their callbacks: Unity discards a completed readback's data within a frame or two, and
    // reports it as an error after that, so polling only when the scene syncs lost them whenever completions outpaced
    // the syncs (many large meshes, or frames without a traced camera while a scene loads). The layout is captured at
    // the start, so a mesh destroyed while its read is under way only ends the read.
    sealed class MeshRead
    {
        public readonly Mesh mesh;
        public MeshData Data { get; private set; }
        public bool Done { get; private set; }

        readonly string name;
        readonly int vertexCount;
        readonly int[] strides;
        readonly (VertexAttributeDescriptor descriptor, int offset)[] attributes;
        readonly SubMeshDescriptor[] subMeshes;
        readonly bool wideIndices;
        // Per buffer (vertex streams, then the index buffer): the GPU buffer until its data arrives, then a copy.
        readonly GraphicsBuffer[] buffers;
        readonly AsyncGPUReadbackRequest[] requests;
        readonly NativeArray<byte>[] arrived;
        readonly bool[] taken;
        int outstanding;
        bool failed;

        MeshRead(Mesh mesh)
        {
            this.mesh = mesh;
            name = mesh.name;
            vertexCount = mesh.vertexCount;
            strides = new int[mesh.vertexBufferCount];
            for (int s = 0; s < strides.Length; ++s)
                strides[s] = mesh.GetVertexBufferStride(s);
            var descriptors = mesh.GetVertexAttributes();
            attributes = new (VertexAttributeDescriptor, int)[descriptors.Length];
            for (int a = 0; a < descriptors.Length; ++a)
                attributes[a] = (descriptors[a], mesh.GetVertexAttributeOffset(descriptors[a].attribute));
            subMeshes = new SubMeshDescriptor[mesh.subMeshCount];
            for (int s = 0; s < subMeshes.Length; ++s)
                subMeshes[s] = mesh.GetSubMesh(s);
            wideIndices = mesh.indexFormat == IndexFormat.UInt32;
            int count = strides.Length + 1;
            buffers = new GraphicsBuffer[count];
            requests = new AsyncGPUReadbackRequest[count];
            arrived = new NativeArray<byte>[count];
            taken = new bool[count];
        }

        public static MeshRead Start(Mesh mesh)
        {
            if (mesh.isReadable)
                return new MeshRead(mesh) { Data = ReadCpu(mesh), Done = true };
            var read = new MeshRead(mesh);
            int last = read.buffers.Length - 1;
            for (int s = 0; s < last; ++s)
                read.buffers[s] = mesh.GetVertexBuffer(s);
            read.buffers[last] = mesh.GetIndexBuffer();
            read.outstanding = read.buffers.Length;
            for (int i = 0; i < read.buffers.Length; ++i)
            {
                int slot = i;
                read.requests[i] = AsyncGPUReadback.Request(read.buffers[i], r => read.Take(slot, r));
            }
            return read;
        }

        // True when done; the readback callbacks finish the read.
        public bool Poll() => Done;

        public void Wait()
        {
            for (int i = 0; i < requests.Length && !Done; ++i)
                if (!taken[i])
                {
                    requests[i].WaitForCompletion();
                    Take(i, requests[i]); // in case the callback comes later; it is then ignored
                }
        }

        void Take(int slot, AsyncGPUReadbackRequest request)
        {
            if (taken[slot] || !request.done)
                return;
            taken[slot] = true;
            if (request.hasError)
                failed = true;
            else
                arrived[slot] = new NativeArray<byte>(request.GetData<byte>(), Allocator.Persistent);
            buffers[slot].Dispose();
            buffers[slot] = null;
            if (--outstanding == 0)
                Finish();
        }

        void Finish()
        {
            try
            {
                Data = failed ? null : ReadGpu();
                if (failed)
                    Debug.LogWarning("[RoyalTracer] Could not read back mesh " + name);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[RoyalTracer] Could not read mesh {name}: {e.Message}");
            }
            finally
            {
                foreach (var a in arrived)
                    if (a.IsCreated)
                        a.Dispose();
                Done = true;
            }
        }

        static MeshData ReadCpu(Mesh mesh)
        {
            var d = new MeshData
            {
                positions = mesh.vertices,
                normals = mesh.normals,
                submeshes = new int[mesh.subMeshCount][],
            };
            var uvs = new List<Vector2>();
            mesh.GetUVs(0, uvs);
            d.uvs = uvs.Count == d.positions.Length ? uvs.ToArray() : null;
            var colors = mesh.colors;
            d.colors = colors.Length == d.positions.Length ? colors : null;
            var weights = mesh.boneWeights;
            d.boneWeights = weights.Length == d.positions.Length ? weights : null;
            for (int s = 0; s < mesh.subMeshCount; ++s)
                d.submeshes[s] = mesh.GetTopology(s) == MeshTopology.Triangles ? mesh.GetIndices(s) : null;
            if (d.normals.Length != d.positions.Length)
                d.normals = null;
            return d;
        }

        MeshData ReadGpu()
        {
            int count = vertexCount;
            var positions = Attribute(VertexAttribute.Position, count, 3, out _);
            if (positions == null)
                return null;
            var d = new MeshData { positions = ToVector3(positions), submeshes = new int[subMeshes.Length][] };
            if (Attribute(VertexAttribute.Normal, count, 3, out _) is { } n)
                d.normals = ToVector3(n);
            if (Attribute(VertexAttribute.TexCoord0, count, 2, out _) is { } uv)
            {
                d.uvs = new Vector2[count];
                for (int v = 0; v < count; ++v)
                    d.uvs[v] = new Vector2(uv[v * 2], uv[v * 2 + 1]);
            }
            if (Attribute(VertexAttribute.Color, count, 4, out _) is { } c)
            {
                d.colors = new Color[count];
                for (int v = 0; v < count; ++v)
                    d.colors[v] = new Color(c[v * 4], c[v * 4 + 1], c[v * 4 + 2], c[v * 4 + 3]);
            }

            var indexData = arrived[arrived.Length - 1];
            bool wide = wideIndices;
            var wideIndexData = wide ? indexData.Reinterpret<int>(1) : default;
            var narrowIndexData = wide ? default : indexData.Reinterpret<ushort>(1);
            for (int s = 0; s < subMeshes.Length; ++s)
            {
                var sm = subMeshes[s];
                if (sm.topology != MeshTopology.Triangles)
                    continue;
                var list = new int[sm.indexCount];
                for (int i = 0; i < sm.indexCount; ++i)
                    list[i] = (wide ? wideIndexData[sm.indexStart + i] : narrowIndexData[sm.indexStart + i]) + sm.baseVertex;
                d.submeshes[s] = list;
            }
            return d;
        }

        // The attribute as `want` floats per vertex (missing components 0, colour alpha 1), or null when absent.
        float[] Attribute(VertexAttribute attribute, int count, int want, out bool present)
        {
            int a = System.Array.FindIndex(attributes, x => x.descriptor.attribute == attribute);
            present = a >= 0;
            if (!present)
                return null;
            var (descriptor, offset) = attributes[a];
            int stream = descriptor.stream;
            int stride = strides[stream];
            var format = descriptor.format;
            int dims = Mathf.Min(descriptor.dimension, want);
            var data = arrived[stream];
            var output = new float[count * want];
            if (want == 4)
                for (int v = 0; v < count; ++v)
                    output[v * 4 + 3] = 1f;
            switch (format)
            {
                case VertexAttributeFormat.Float32:
                {
                    var f = data.Reinterpret<float>(1);
                    for (int v = 0; v < count; ++v)
                        for (int k = 0; k < dims; ++k)
                            output[v * want + k] = f[(v * stride + offset) / 4 + k];
                    break;
                }
                case VertexAttributeFormat.Float16:
                case VertexAttributeFormat.UNorm16:
                case VertexAttributeFormat.SNorm16:
                {
                    var h = data.Reinterpret<ushort>(1);
                    for (int v = 0; v < count; ++v)
                        for (int k = 0; k < dims; ++k)
                        {
                            ushort x = h[(v * stride + offset) / 2 + k];
                            output[v * want + k] = format == VertexAttributeFormat.Float16 ? Mathf.HalfToFloat(x)
                                : format == VertexAttributeFormat.UNorm16 ? x / 65535f
                                : Mathf.Max((short)x / 32767f, -1f);
                        }
                    break;
                }
                case VertexAttributeFormat.UNorm8:
                case VertexAttributeFormat.SNorm8:
                {
                    var b = data;
                    for (int v = 0; v < count; ++v)
                        for (int k = 0; k < dims; ++k)
                        {
                            byte x = b[v * stride + offset + k];
                            output[v * want + k] = format == VertexAttributeFormat.UNorm8 ? x / 255f : Mathf.Max((sbyte)x / 127f, -1f);
                        }
                    break;
                }
                default:
                    Debug.LogWarning($"[RoyalTracer] {name}: vertex format {format} of {attribute} is not supported");
                    present = attribute != VertexAttribute.Position;
                    return null;
            }
            return output;
        }

        static Vector3[] ToVector3(float[] f)
        {
            var v = new Vector3[f.Length / 3];
            for (int i = 0; i < v.Length; ++i)
                v[i] = new Vector3(f[i * 3], f[i * 3 + 1], f[i * 3 + 2]);
            return v;
        }
    }
}
