#pragma once
#include <stdint.h>

// C ABI of RoyalTracerUnity.dll, mirrored field for field in RoyalTracerNative.cs (Armored Horizon package).
// Positions, directions and transforms are in tracer space: right-handed, Y up, metres. Light values are tracer
// units; the caller converts from HDRP's physical units.

#ifdef RT_UNITY_BUILD
#define RT_API extern "C" __declspec(dllexport)
#else
#define RT_API extern "C" __declspec(dllimport)
#endif

enum RT_LogLevel : int32_t { RT_LOG_INFO = 0, RT_LOG_WARNING = 1, RT_LOG_ERROR = 2 };
typedef void(__stdcall* RT_LogFn)(int32_t level, const char* message);

struct RT_InitDesc {
    const wchar_t* dataDir;  // sl.interposer.dll, Streamline plugins, dxcompiler.dll, shaders\ and D3D12\ (Agility 1.619)
    const wchar_t* cacheDir; // writable; compiled shaders are kept here
    RT_LogFn log;            // any thread; may be null
};

enum RT_TextureKind : uint32_t { RT_TEXTURE_ALBEDO = 0, RT_TEXTURE_NORMAL = 1, RT_TEXTURE_RMA = 2 };

struct RT_TextureDesc {
    uint32_t width;
    uint32_t height;
    uint32_t kind;        // RT_TextureKind. Albedo is sRGB with alpha; normal is tangent space *0.5+0.5;
                          // RMA is (unused, perceptual roughness, metallic)
    const uint8_t* rgba8; // width * height * 4 bytes, top row first; mips are generated
};

struct RT_MaterialDesc {
    float baseColor[4];  // linear RGB; alpha 1 for opaque
    float emission[3];   // radiance
    float roughness;     // perceptual (GGX alpha = roughness^2)
    float metallic;
    float coat;
    float coatRoughness;
    float ior;
    float alphaCutoff;   // 1 = no alpha test
    int32_t albedoTexture; // RT_CreateTexture ids, -1 = none. A texture replaces the constant it stands for.
    int32_t normalTexture;
    int32_t rmaTexture;
    float uvScale[2];
    uint32_t flags;      // RT_MaterialFlags
    float spotCosInner;  // RT_MATERIAL_SPOT_EMISSION: cosines of the cone's half-angles around the front normal
    float spotCosOuter;
};

enum RT_MaterialFlags : uint32_t {
    RT_MATERIAL_THIN_GLASS = 1, // a single refracting interface tinted by baseColor (rgb); roughness and ior apply
    // Emission fades from the inner to the outer cone, with the radiant intensity of a spot light: emission is the
    // radiance seen along the normal, and the intensity stays emission * area across the inner cone.
    RT_MATERIAL_SPOT_EMISSION = 2,
};

struct RT_MeshDesc {
    const float* positions;     // float3 per vertex
    const float* normals;       // float3 per vertex
    const float* uvs;           // float2 per vertex, or null
    uint32_t vertexCount;
    const uint32_t* indices;    // triangle list, counter-clockwise front faces
    uint32_t indexCount;
    const int32_t* triangleMaterials; // RT_CreateMaterial id per triangle; null or unknown ids: grey fallback
    uint32_t flags;                   // RT_MeshFlags
};

enum RT_MeshFlags : uint32_t {
    RT_MESH_DEFORMING = 1, // RT_UpdateMeshVertices moves its vertices (refits instead of rebuilds)
};

struct RT_FrameDesc {
    float eye[3];
    float target[3];
    float up[3];
    float fovYDegrees;
    uint32_t width;
    uint32_t height;
    float deltaTime;
    float sunDirection[3]; // towards the sun
    float sunIntensity;    // normal irradiance above the atmosphere
    float skyIntensity;
    float turbidity;
    float skyGroundY;      // world height below which the sky is treated as ground
};

// What the host composites for one submitted frame. Textures live on the host's device.
struct RT_FrameTicket {
    uint64_t index;   // pass to the render events as their data
    uint32_t slot;    // which output pair this frame writes (0 or 1)
    uint32_t width;
    uint32_t height;
    uint32_t generation; // changes when the output textures are recreated
    void* color[2];   // ID3D12Resource*: RGBA16F, pre-exposed; alpha = log2 of that exposure
    void* depth[2];   // ID3D12Resource*: R32F view depth (m) of the unjittered primary hit, 0 = miss
};

enum RT_RenderEvent : int32_t {
    RT_EVENT_RENDER = 1,  // render the ticket's frame; the host queue waits for it
    RT_EVENT_RELEASE = 2, // the host has read the ticket's outputs (optional: the next ticket's RT_EVENT_RENDER implies it)
    RT_EVENT_PRESENT = 3, // present the host's finished frame of the ticket (RT_PresenterSource), with frame generation
};

// The adapter the renderer runs on and whether it can path trace there. Filled by RT_Init, also when it fails.
struct RT_Capabilities {
    int32_t supported;             // RT_Init succeeded
    uint32_t vendorId;
    uint64_t dedicatedVideoMemory; // bytes
    uint32_t shaderModel;          // D3D_SHADER_MODEL, 0x69 = 6.9
    uint32_t raytracingTier;       // D3D12_RAYTRACING_TIER, 12 = 1.2
    int32_t serReorders;           // the driver actually reorders SER hits (performance only)
    int32_t dlssRayReconstruction;
    char adapterName[128];         // UTF-8
    char reason[256];              // why path tracing is unavailable; empty when supported
};

// Returns 1 on success. A failed RT_Init is final for the process: later calls return -1 without retrying.
RT_API int32_t RT_Init(const RT_InitDesc* desc);
RT_API int32_t RT_GetCapabilities(RT_Capabilities* caps); // 0 before the first RT_Init
RT_API void RT_Shutdown();
RT_API const char* RT_GetLastError();

// Scene content streams into the running renderer: objects can be created and released at any time and appear in
// the next submitted frame. Create calls return an id >= 0, or -1 (reason in RT_GetLastError). Released objects free
// their GPU memory a few frames later; a mesh stays alive while instances use it. RT_Shutdown releases everything.
RT_API int32_t RT_CreateTexture(const RT_TextureDesc* desc);
RT_API void RT_ReleaseTexture(int32_t texture);
RT_API int32_t RT_CreateMaterial(const RT_MaterialDesc* desc);
RT_API int32_t RT_UpdateMaterial(int32_t material, const RT_MaterialDesc* desc); // 1 on success
RT_API void RT_ReleaseMaterial(int32_t material);
RT_API int32_t RT_CreateMesh(const RT_MeshDesc* desc);
RT_API void RT_ReleaseMesh(int32_t mesh);
// A deforming mesh's vertices for the next frame, in creation order; positions and normals as in RT_MeshDesc.
RT_API int32_t RT_UpdateMeshVertices(int32_t mesh, const float* positions, const float* normals); // 1 on success
RT_API int32_t RT_CreateInstance(int32_t mesh, const float* rowMajor3x4); // object to world
RT_API void RT_ReleaseInstance(int32_t instance);
RT_API void RT_SetInstanceVisible(int32_t instance, int32_t visible); // hidden instances keep their resources
enum RT_InstanceFlags : uint32_t {
    RT_INSTANCE_EMITTER = 1, // a light's or particle's proxy: lights the scene, casts no shadows
};
RT_API void RT_SetInstanceFlags(int32_t instance, uint32_t flags); // RT_InstanceFlags

// Emitter sets: particles and other small lights that change every frame. Each emitter lights the scene like a sphere
// of the given radius and radiance and shows in reflections; camera and shadow rays pass through it (the host draws
// its particles). An update replaces the whole set.
struct RT_Emitter {
    float position[3]; // world
    float radius;      // m
    float radiance[3];
    float pad;
};
RT_API int32_t RT_CreateEmitterSet(uint32_t capacity);
RT_API int32_t RT_UpdateEmitterSet(int32_t set, const RT_Emitter* emitters, uint32_t count); // 1 on success
RT_API void RT_ReleaseEmitterSet(int32_t set);
RT_API const char* RT_GetStats(); // one line: what the stream holds

enum RT_Quality : int32_t { // DLSS-RR input resolution
    RT_QUALITY_DLAA = 0,      // native
    RT_QUALITY_QUALITY = 1,
    RT_QUALITY_BALANCED = 2,
    RT_QUALITY_PERFORMANCE = 3,
    RT_QUALITY_ULTRA_PERFORMANCE = 4,
};
RT_API void RT_SetQuality(int32_t quality); // RT_Quality; takes effect on the next rendered frame

RT_API void RT_SetInstanceTransforms(int32_t count, const int32_t* instances, const float* rowMajor3x4s);
RT_API int32_t RT_SubmitFrame(const RT_FrameDesc* desc, RT_FrameTicket* ticket);
RT_API void* RT_GetRenderEventFunc(); // UnityRenderingEventAndData; data = ticket index

// Textures the host writes for the renderer (water simulation data, for example): created on the renderer's device and
// shared with the host's. The host fills them on its graphics queue before the RT_EVENT_RENDER of the frame that reads
// them; that event orders the renderer's work after those writes. hostTexture receives the host device's
// ID3D12Resource for the same memory. Returns an id, -1 on failure.
RT_API int32_t RT_CreateHostTexture(uint32_t width, uint32_t height, int32_t dxgiFormat, void** hostTexture);
RT_API void RT_ReleaseHostTexture(int32_t texture); // freed once no queued frame reads it

// The host's ocean (an HDRP water surface, for example) drives the renderer's: up to four wave bands, each a host
// texture (RGBA16F, height in x, wrapping). A band's height at world (x, z) of the host's left-handed frame is
// amplitude * texture((x, z) * invPatchSize - offset).x, faded with distance d to the camera by
// smoothstep(saturate(d * fadeA + fadeB)^2). The surface is the sum of the bands above seaLevel. bandCount 0 hides the
// renderer's ocean. Applies from the next submitted frame; send it every frame (the offsets move with time).
struct RT_WaterBand {
    int32_t texture; // RT_CreateHostTexture id
    float invPatchSize;
    float offset[2];
    float amplitude;
    float fadeA, fadeB;
    float pad;
};
struct RT_WaterDesc {
    uint32_t bandCount;
    uint32_t resolution;  // band texture size
    float seaLevel;       // world y of the still surface
    float maxWaveHeight;  // bound on |height|, for culling and the underwater test
    RT_WaterBand bands[4];
    float absorption[3];  // water body, 1/m per RGB; all zero (with scattering) keeps the renderer's own water
    float scattering[3];
    float windSpeed;      // m/s at 10 m: whitecap cover and the unresolved slopes' roughness; 0 keeps the renderer's
    float foamCoverage;   // whitecap cover gain, 0 = no foam
};
RT_API int32_t RT_SetWater(const RT_WaterDesc* desc); // 1 on success

// The host's sky in place of the renderer's atmosphere model, for everything that sees or is lit by it (the sun
// stays the renderer's): a host texture (RGBA16F) of radiance over a latitude-longitude map of the host's directions,
// u = atan2(-z, x) / 2 pi + 1/2 and v = acos(y) / pi at the host's left-handed (x, y, z), times scale. Its row 0 is
// straight up. texture -1 brings back the atmosphere model. Read at each frame; the host writes it before the event.
RT_API int32_t RT_SetSky(int32_t texture, float scale); // 1 on success

// Frame generation (DLSS-G): the renderer presents the host's finished frames itself, through a child window over
// the host's window, on its own device where DLSS frame generation runs. Player builds only (the host's swap chain
// gives the window). Main thread: RT_PresenterEnable, then RT_PresenterUpdate each frame (follows the window's size).
// Each frame the host copies its finished frame (UI included) and the same frame before its UI (both RGBA8 sRGB, top
// row first) into the RT_PresenterSource host textures, and issues RT_EVENT_PRESENT with the frame's ticket.
// RT_PresenterEnable: 1 on success; 0 and RT_GetLastError when it cannot run.
RT_API int32_t RT_PresenterEnable(int32_t enable);
RT_API void RT_PresenterUpdate();
RT_API int32_t RT_PresenterSource(int32_t finished, int32_t hudless); // hudless -1: the finished frame
RT_API void RT_PresenterOptions(int32_t framesToGenerate, int32_t vsync); // 1..5 generated per rendered frame
// Most frames generated per rendered frame on this GPU: 0 none, -1 supported but not known until it has run.
RT_API int32_t RT_FrameGenerationMax();
RT_API const char* RT_PresenterStats(); // one line: presents, frame generation's state

// Overlays on the water (ship wakes, water deformers): rectangles of the host's world adding height and foam over the
// waves, each a texel region of two atlases (host textures). The height atlas holds height in x (m, times heightScale);
// the foam atlas whitecaps in x and bubbles in y as optical thickness (cover 1 - exp(-value * foamScale)). A region
// (x, y, w, h) spans the rectangle from its corner origin to origin + size along the host's world x and z; w 0 leaves
// that layer out. For the given frame only: send them for each frame after RT_SubmitFrame, before its render event.
struct RT_WaterOverlay {
    float origin[2]; // host world x, z
    float size[2];   // m
    float heightRegion[4];
    float foamRegion[4];
    float heightScale;
    float foamScale;
    float maxHeight; // bound on |height| after the scale, for the underwater test
    float foamMax;   // cap on the whitecap cover it adds, 0..1 (0: none)
};
#define RT_MAX_WATER_OVERLAYS 32
RT_API int32_t RT_SetWaterOverlays(uint64_t ticket, int32_t heightAtlas, int32_t foamAtlas, uint32_t count,
                                   const RT_WaterOverlay* overlays); // 1 on success

// Tests without a host: render the ticket on the calling thread and read its outputs back.
RT_API int32_t RT_RenderTicketForTest(uint64_t index, float* rgbaOut, float* depthOut);
