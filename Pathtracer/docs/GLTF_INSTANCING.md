# glTF / GLB instancing

`ObjLoader::loadGlbFile` reads both JSON glTF (including external buffers) and
binary GLB. Mesh vertices remain in mesh-local space. Multiple nodes referencing
the same glTF mesh share one loaded geometry and one renderer BLAS, regardless of
mesh size or instance count. Single-use meshes can still be merged to reduce BLAS
count; spatial splitting preserves references to every shared piece.

The loader supports `EXT_mesh_gpu_instancing` translation, rotation and scale
accessors, including strided buffers, sparse overrides and normalized BYTE/SHORT
quaternions. Missing TRS attributes use identity defaults. Custom `_` attributes
contribute their accessor count but do not add custom shader behavior. The batch
replaces the node's ordinary mesh placement; children retain the node transform
and are traversed once. Instance transforms compose as
`instance TRS * node * ancestors * model` in DirectXMath's row-vector convention.

Only the default scene is imported (the first scene when no default is specified).
For files without a scenes array, root nodes are imported as a forest. Unreferenced
meshes and primitives without triangle geometry do not allocate renderer meshes.
Malformed instance accessor counts, formats and data spans fail the import with
an error instead of creating a fallback copy at the node origin.

The renderer shares geometry and material ranges while retaining separate instance
transforms, previous transforms, triangle-to-light ranges and emissive triangles.
Surface and shadow normals use the inverse transpose for nonuniform scaling.
Reflected instances keep their outward light normals and normal-map handedness;
reflections baked into single-use geometry also reverse triangle winding.

This is static mesh instancing. Skinning, morph-target deformation, animation and
custom per-instance material/color attributes are not implemented by this change.
Repeated model-file entries are separate imports; sharing is within each import.

## Verification

In a Visual Studio Developer PowerShell, run:

```powershell
./tests/run_gltf_instancing_tests.ps1
```

The runner uses the existing Release/RelWithDebInfo `DirectXTex.lib`; pass
`-DirectXTexLibrary` if it is built elsewhere. As in the application build, the
linker permits duplicate STB symbols provided by the OMM SDK.

Tests generate small glTF/GLB fixtures and exercise the production loader,
`AssetLoader::LoadModels`, D3D12 buffer allocation, material partitioning, instance
properties, emissive light indexing, model movement, mirrored geometry and OBJ
compatibility. A GPU dispatch checks the production surface, shadow and light
sampling normals on scaled and reflected instances. The runner also compiles the
production path-tracing and shading shaders.
Outputs and generated fixtures are written to `out/gltf-instancing-tests`.
