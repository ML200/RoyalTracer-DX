using System;
using System.Runtime.InteropServices;

namespace RoyalTracer
{
    // Mirrors Pathtracer/unity/RoyalTracerApi.h in RoyalTracer-DX field for field. Vectors and transforms are in
    // tracer space (right-handed, Y up): Unity's z is negated, see Space.
    static class Native
    {
        const string Dll = "RoyalTracerUnity";

        public const int EventRender = 1, EventRelease = 2;
        public const uint TextureAlbedo = 0, TextureNormal = 1, TextureRma = 2;
        public const int LogInfo = 0, LogWarning = 1, LogError = 2;

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void LogFn(int level, IntPtr message);

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        public struct InitDesc
        {
            public string dataDir;
            public string cacheDir;
            public IntPtr log;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct Capabilities
        {
            public int supported;
            public uint vendorId;
            public ulong dedicatedVideoMemory;
            public uint shaderModel, raytracingTier;
            public int serReorders, dlssRayReconstruction;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 128)] public byte[] adapterName;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)] public byte[] reason;

            public string AdapterName => Utf8(adapterName);
            public string Reason => Utf8(reason);

            static string Utf8(byte[] bytes)
            {
                if (bytes == null) return "";
                int n = Array.IndexOf(bytes, (byte)0);
                return System.Text.Encoding.UTF8.GetString(bytes, 0, n < 0 ? bytes.Length : n);
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct TextureDesc
        {
            public uint width, height, kind;
            public IntPtr rgba8;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct MaterialDesc
        {
            public float baseR, baseG, baseB, baseA;
            public float emissionR, emissionG, emissionB;
            public float roughness, metallic, coat, coatRoughness, ior, alphaCutoff;
            public int albedoTexture, normalTexture, rmaTexture;
            public float uvScaleX, uvScaleY;
            public uint flags;
            public float spotCosInner, spotCosOuter;
        }

        public const uint MaterialThinGlass = 1;
        public const uint MaterialSpotEmission = 2;

        [StructLayout(LayoutKind.Sequential)]
        public struct MeshDesc
        {
            public IntPtr positions, normals, uvs;
            public uint vertexCount;
            public IntPtr indices;
            public uint indexCount;
            public IntPtr triangleMaterials;
            public uint flags;
        }

        public const uint MeshDeforming = 1;
        public const uint InstanceEmitter = 1;

        [StructLayout(LayoutKind.Sequential)]
        public struct FrameDesc
        {
            public float eyeX, eyeY, eyeZ;
            public float targetX, targetY, targetZ;
            public float upX, upY, upZ;
            public float fovYDegrees;
            public uint width, height;
            public float deltaTime;
            public float sunX, sunY, sunZ;
            public float sunIntensity, skyIntensity, turbidity, skyGroundY;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct FrameTicket
        {
            public ulong index;
            public uint slot, width, height, generation;
            public IntPtr color0, color1, depth0, depth1;
        }

        [DllImport(Dll)] public static extern int RT_Init(ref InitDesc desc);
        [DllImport(Dll)] public static extern int RT_GetCapabilities(out Capabilities caps);
        [DllImport(Dll)] public static extern void RT_Shutdown();
        [DllImport(Dll)] public static extern IntPtr RT_GetLastError();
        [DllImport(Dll)] public static extern int RT_CreateTexture(ref TextureDesc desc);
        [DllImport(Dll)] public static extern void RT_ReleaseTexture(int texture);
        [DllImport(Dll)] public static extern int RT_CreateMaterial(ref MaterialDesc desc);
        [DllImport(Dll)] public static extern int RT_UpdateMaterial(int material, ref MaterialDesc desc);
        [DllImport(Dll)] public static extern void RT_ReleaseMaterial(int material);
        [DllImport(Dll)] public static extern int RT_CreateMesh(ref MeshDesc desc);
        [DllImport(Dll)] public static extern void RT_ReleaseMesh(int mesh);
        [DllImport(Dll)] public static extern int RT_UpdateMeshVertices(int mesh, float[] positions, float[] normals);
        [DllImport(Dll)] public static extern int RT_CreateInstance(int mesh, float[] rowMajor3x4);
        [DllImport(Dll)] public static extern void RT_ReleaseInstance(int instance);
        [DllImport(Dll)] public static extern void RT_SetInstanceVisible(int instance, int visible);
        [DllImport(Dll)] public static extern void RT_SetInstanceFlags(int instance, uint flags);

        [StructLayout(LayoutKind.Sequential)]
        public struct Emitter
        {
            public float x, y, z, radius; // tracer space, m
            public float r, g, b, pad;    // radiance
        }

        [DllImport(Dll)] public static extern int RT_CreateEmitterSet(uint capacity);
        [DllImport(Dll)] public static extern int RT_UpdateEmitterSet(int set, Emitter[] emitters, uint count);
        [DllImport(Dll)] public static extern void RT_ReleaseEmitterSet(int set);
        [DllImport(Dll)] public static extern void RT_SetInstanceTransforms(int count, int[] instances, float[] rowMajor3x4s);
        [DllImport(Dll)] public static extern IntPtr RT_GetStats();
        [DllImport(Dll)] public static extern void RT_SetQuality(int quality);
        [DllImport(Dll)] public static extern int RT_SubmitFrame(ref FrameDesc desc, out FrameTicket ticket);
        [DllImport(Dll)] public static extern IntPtr RT_GetRenderEventFunc();

        public static string LastError => Marshal.PtrToStringUTF8(RT_GetLastError()) ?? "";
        public static string Stats => Marshal.PtrToStringUTF8(RT_GetStats()) ?? "";
    }
}
