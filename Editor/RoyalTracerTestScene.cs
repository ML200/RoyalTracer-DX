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
