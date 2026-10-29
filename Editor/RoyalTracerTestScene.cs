using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace RoyalTracer.Editor
{
    // A small HDRP scene that needs nothing from the project: ground, diffuse/metal/glossy/emissive HDRP/Lit
    // objects, a sun, physically based sky and fixed exposure. Used by RoyalTracerCapture when no scene is given.
    public static class RoyalTracerTestScene
    {
        [MenuItem("Tools/RoyalTracer/Create Test Scene")]
        static void Create()
        {
            if (EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                Build();
        }

        public static void Build()
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var cameraObject = new GameObject("Main Camera") { tag = "MainCamera" };
            var camera = cameraObject.AddComponent<Camera>();
            cameraObject.AddComponent<HDAdditionalCameraData>();
            camera.fieldOfView = 50f;
            camera.nearClipPlane = 0.1f;
            camera.farClipPlane = 2000f;
            cameraObject.transform.position = new Vector3(0f, 2.2f, -7f);
            cameraObject.transform.LookAt(new Vector3(0f, 0.8f, 0f));

            var sunObject = new GameObject("Sun");
            var sun = sunObject.AddComponent<Light>();
            sun.type = LightType.Directional;
            sunObject.AddComponent<HDAdditionalLightData>();
            sun.intensity = 100000f; // lux
            sun.shadows = LightShadows.Soft;
            sunObject.transform.rotation = Quaternion.Euler(35f, -30f, 0f);

            var profile = ScriptableObject.CreateInstance<VolumeProfile>();
            profile.Add<VisualEnvironment>(true).skyType.Override((int)SkyType.PhysicallyBased);
            profile.Add<PhysicallyBasedSky>(true);
            var exposure = profile.Add<Exposure>(true);
            exposure.mode.Override(ExposureMode.Fixed);
            exposure.fixedExposure.Override(14f);
            var volume = new GameObject("Sky and Exposure").AddComponent<Volume>();
            volume.isGlobal = true;
            volume.sharedProfile = profile;

            Shape(PrimitiveType.Plane, "Ground", new Vector3(0f, 0f, 0f), Vector3.one * 4f, 0f,
                Lit(new Color(0.45f, 0.45f, 0.45f), 0f, 0.3f));
            Shape(PrimitiveType.Cube, "Red Cube", new Vector3(-2.2f, 0.5f, 0f), Vector3.one, 15f,
                Lit(new Color(0.7f, 0.08f, 0.06f), 0f, 0.2f));
            Shape(PrimitiveType.Cube, "Gold Cube", new Vector3(0f, 0.75f, 1.2f), Vector3.one * 1.5f, 30f,
                Lit(new Color(1f, 0.78f, 0.34f), 1f, 0.85f));
            Shape(PrimitiveType.Sphere, "White Sphere", new Vector3(2.2f, 0.7f, -0.3f), Vector3.one * 1.4f, 0f,
                Lit(new Color(0.8f, 0.8f, 0.8f), 0f, 0.9f));
            Shape(PrimitiveType.Cube, "Pillar", new Vector3(-0.6f, 1.5f, 3.5f), new Vector3(0.6f, 3f, 0.6f), 0f,
                Lit(new Color(0.3f, 0.35f, 0.5f), 0f, 0.5f));
            var glow = Lit(new Color(0.05f, 0.05f, 0.05f), 0f, 0.5f);
            glow.SetColor("_EmissiveColor", new Color(0.2f, 1f, 1f) * 20000f); // nits
            HDMaterial.ValidateMaterial(glow);
            Shape(PrimitiveType.Sphere, "Glow Sphere", new Vector3(0.9f, 0.3f, -1.6f), Vector3.one * 0.6f, 0f, glow);

            // A shader the converter does not know: traced through the fallback (main colour and texture).
            var graph = AssetDatabase.LoadAssetAtPath<Shader>(
                "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipelineResources/ShaderGraph/AutodeskInteractive.shadergraph");
            if (graph)
            {
                var other = new Material(graph);
                other.color = new Color(0.9f, 0.45f, 0.1f);
                HDMaterial.ValidateMaterial(other);
                Shape(PrimitiveType.Cylinder, "Shader Graph Cylinder", new Vector3(3.6f, 0.75f, 1.6f), new Vector3(0.8f, 0.75f, 0.8f), 0f, other);
            }

            SkinnedBar(Lit(new Color(0.25f, 0.6f, 0.3f), 0f, 0.5f));
        }

        // A bar bent by two bones with blended weights in the middle: the tracer must follow the skinned pose.
        static void SkinnedBar(Material material)
        {
            const int Rings = 12;
            const float Height = 2.4f, Half = 0.18f;
            var vertices = new List<Vector3>();
            var weights = new List<BoneWeight>();
            var indices = new List<int>();
            var corners = new[] { new Vector2(-Half, -Half), new Vector2(Half, -Half), new Vector2(Half, Half), new Vector2(-Half, Half) };
            for (int r = 0; r <= Rings; ++r)
            {
                float y = Height * r / Rings, t = Mathf.InverseLerp(0.35f, 0.65f, (float)r / Rings);
                foreach (var c in corners)
                {
                    vertices.Add(new Vector3(c.x, y, c.y));
                    weights.Add(new BoneWeight { boneIndex0 = 0, weight0 = 1f - t, boneIndex1 = 1, weight1 = t });
                }
            }
            for (int r = 0; r < Rings; ++r)
                for (int s = 0; s < 4; ++s)
                {
                    int a = r * 4 + s, b = r * 4 + (s + 1) % 4, c = a + 4, d = b + 4;
                    indices.AddRange(new[] { a, c, b, b, c, d });
                }
            var mesh = new Mesh { name = "Skinned Bar" };
            mesh.SetVertices(vertices);
            mesh.SetTriangles(indices, 0);
            mesh.boneWeights = weights.ToArray();
            mesh.RecalculateNormals();

            var root = new GameObject("Skinned Bar");
            root.transform.position = new Vector3(-3.6f, 0f, 1.8f);
            var lower = new GameObject("Lower").transform;
            lower.SetParent(root.transform, false);
            var upper = new GameObject("Upper").transform;
            upper.SetParent(lower, false);
            upper.localPosition = new Vector3(0f, Height * 0.5f, 0f);
            mesh.bindposes = new[] { lower.worldToLocalMatrix * root.transform.localToWorldMatrix, upper.worldToLocalMatrix * root.transform.localToWorldMatrix };
            upper.localRotation = Quaternion.Euler(0f, 0f, -40f); // the pose differs from the bind pose
            var skin = root.AddComponent<SkinnedMeshRenderer>();
            skin.sharedMesh = mesh;
            skin.bones = new[] { lower, upper };
            skin.rootBone = lower;
            skin.sharedMaterial = material;
            skin.updateWhenOffscreen = true;
        }

        static Material Lit(Color color, float metallic, float smoothness)
        {
            var material = new Material(Shader.Find("HDRP/Lit"));
            material.SetColor("_BaseColor", color);
            material.SetFloat("_Metallic", metallic);
            material.SetFloat("_Smoothness", smoothness);
            HDMaterial.ValidateMaterial(material);
            return material;
        }

        static void Shape(PrimitiveType type, string name, Vector3 position, Vector3 scale, float yaw, Material material)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            go.transform.SetPositionAndRotation(position, Quaternion.Euler(0f, yaw, 0f));
            go.transform.localScale = scale;
            go.GetComponent<MeshRenderer>().sharedMaterial = material;
        }
    }
}
