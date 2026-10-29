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
    sealed class MeshRead
    {
        public readonly Mesh mesh;
        public MeshData Data { get; private set; }
        public bool Done { get; private set; }

        readonly List<GraphicsBuffer> buffers = new();
        readonly List<AsyncGPUReadbackRequest> requests = new(); // vertex streams, then the index buffer

        MeshRead(Mesh mesh) => this.mesh = mesh;

        public static MeshRead Start(Mesh mesh)
        {
            var read = new MeshRead(mesh);
            if (mesh.isReadable)
            {
                read.Data = ReadCpu(mesh);
                read.Done = true;
                return read;
            }
            for (int s = 0; s < mesh.vertexBufferCount; ++s)
            {
                var buffer = mesh.GetVertexBuffer(s);
                read.buffers.Add(buffer);
                read.requests.Add(AsyncGPUReadback.Request(buffer));
            }
            var indices = mesh.GetIndexBuffer();
            read.buffers.Add(indices);
            read.requests.Add(AsyncGPUReadback.Request(indices));
            return read;
        }

        // Finishes the read if the GPU has delivered; true when done.
        public bool Poll()
        {
            if (Done)
                return true;
            foreach (var r in requests)
                if (!r.done)
                    return false;
            Finish();
            return true;
        }

        public void Wait()
        {
            if (Done)
                return;
            foreach (var r in requests)
                r.WaitForCompletion();
            Finish();
        }

        void Finish()
        {
            try
            {
                bool failed = false;
                foreach (var r in requests)
                    failed |= r.hasError;
                Data = failed ? null : ReadGpu();
                if (failed)
                    Debug.LogWarning("[RoyalTracer] Could not read back mesh " + mesh.name);
            }
            finally
            {
                foreach (var b in buffers)
                    b.Dispose();
                buffers.Clear();
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
            int count = mesh.vertexCount;
            var positions = Attribute(VertexAttribute.Position, count, 3, out _);
            if (positions == null)
                return null;
            var d = new MeshData { positions = ToVector3(positions), submeshes = new int[mesh.subMeshCount][] };
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

            var indexRequest = requests[requests.Count - 1];
            bool wide = mesh.indexFormat == IndexFormat.UInt32;
            var wideIndices = wide ? indexRequest.GetData<int>() : default;
            var narrowIndices = wide ? default : indexRequest.GetData<ushort>();
            for (int s = 0; s < mesh.subMeshCount; ++s)
            {
                var sm = mesh.GetSubMesh(s);
                if (sm.topology != MeshTopology.Triangles)
                    continue;
                var list = new int[sm.indexCount];
                for (int i = 0; i < sm.indexCount; ++i)
                    list[i] = (wide ? wideIndices[sm.indexStart + i] : narrowIndices[sm.indexStart + i]) + sm.baseVertex;
                d.submeshes[s] = list;
            }
            return d;
        }

        // The attribute as `want` floats per vertex (missing components 0, colour alpha 1), or null when absent.
        float[] Attribute(VertexAttribute attribute, int count, int want, out bool present)
        {
            present = mesh.HasVertexAttribute(attribute);
            if (!present)
                return null;
            int stream = mesh.GetVertexAttributeStream(attribute);
            int offset = mesh.GetVertexAttributeOffset(attribute);
            int stride = mesh.GetVertexBufferStride(stream);
            var format = mesh.GetVertexAttributeFormat(attribute);
            int dims = Mathf.Min(mesh.GetVertexAttributeDimension(attribute), want);
            var request = requests[stream];
            var output = new float[count * want];
            if (want == 4)
                for (int v = 0; v < count; ++v)
                    output[v * 4 + 3] = 1f;
            switch (format)
            {
                case VertexAttributeFormat.Float32:
                {
                    var f = request.GetData<float>();
                    for (int v = 0; v < count; ++v)
                        for (int k = 0; k < dims; ++k)
                            output[v * want + k] = f[(v * stride + offset) / 4 + k];
                    break;
                }
                case VertexAttributeFormat.Float16:
                case VertexAttributeFormat.UNorm16:
                case VertexAttributeFormat.SNorm16:
                {
                    var h = request.GetData<ushort>();
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
                    var b = request.GetData<byte>();
                    for (int v = 0; v < count; ++v)
                        for (int k = 0; k < dims; ++k)
                        {
                            byte x = b[v * stride + offset + k];
                            output[v * want + k] = format == VertexAttributeFormat.UNorm8 ? x / 255f : Mathf.Max((sbyte)x / 127f, -1f);
                        }
                    break;
                }
                default:
                    Debug.LogWarning($"[RoyalTracer] {mesh.name}: vertex format {format} of {attribute} is not supported");
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
