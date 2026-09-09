// Exercises the production file loader, asset upload, and scene bookkeeping.
#include "../rdn/stdafx.h"
#include <filesystem>
#include <fstream>
#include <source_location>
#include "../rdn/Scene/AssetLoader.h"
#include "../third_party/tiny-cuda-nn/dependencies/json/json.hpp"

using Json = nlohmann::json;
namespace fs = std::filesystem;

static void Require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static void Near(float actual, float expected, const char* message) {
    Require(std::abs(actual - expected) < 0.0002f, message);
}
static void Position(const XMMATRIX& m, XMFLOAT3 p, XMFLOAT3 expected) {
    XMFLOAT3 actual;
    XMStoreFloat3(&actual, XMVector3TransformCoord(XMLoadFloat3(&p), m));
    Near(actual.x, expected.x, "Wrong transformed X");
    Near(actual.y, expected.y, "Wrong transformed Y");
    Near(actual.z, expected.z, "Wrong transformed Z");
}

struct Fixture {
    Json doc = {{"asset", {{"version", "2.0"}}}};
    std::vector<uint8_t> bytes;

    template<class T> int View(std::initializer_list<T> values, int stride = 0) {
        while (bytes.size() % 4) bytes.push_back(0);
        const size_t offset = bytes.size();
        const auto* data = reinterpret_cast<const uint8_t*>(values.begin());
        bytes.insert(bytes.end(), data, data + values.size() * sizeof(T));
        Json view = {{"buffer", 0}, {"byteOffset", offset}, {"byteLength", values.size() * sizeof(T)}};
        if (stride) view["byteStride"] = stride;
        doc["bufferViews"].push_back(view);
        return (int)doc["bufferViews"].size() - 1;
    }
    int Accessor(int view, int component, const char* type, int count, bool normalized = false) {
        Json a = {{"componentType", component}, {"type", type}, {"count", count}};
        if (view >= 0) a["bufferView"] = view;
        if (normalized) a["normalized"] = true;
        doc["accessors"].push_back(a);
        return (int)doc["accessors"].size() - 1;
    }
    Fixture() {
        const int positions = Accessor(View<float>({0,0,0, 1,0,0, 0,1,1}), 5126, "VEC3", 3);
        const int indices = Accessor(View<uint16_t>({0,1,2}), 5123, "SCALAR", 3);
        doc["accessors"][positions]["min"] = {0,0,0};
        doc["accessors"][positions]["max"] = {1,1,1};
        doc["materials"] = Json::array({{{"emissiveFactor", {1,2,3}}}, {{"alphaMode", "MASK"}}});
        const Json prim = {{"attributes", {{"POSITION", positions}}}, {"indices", indices}, {"material", 0}};
        doc["meshes"] = Json::array({{{"primitives", Json::array({prim})}}});
        doc["nodes"] = Json::array({{{"mesh", 0}}});
        doc["scenes"] = Json::array({{{"nodes", {0}}}});
        doc["scene"] = 0;
    }
    void Instances(const Json& attributes) {
        doc["extensionsUsed"] = {"EXT_mesh_gpu_instancing"};
        doc["extensionsRequired"] = {"EXT_mesh_gpu_instancing"};
        doc["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]["attributes"] = attributes;
    }
    fs::path Write(const fs::path& dir, const std::string& name, bool glb) const {
        Json json = doc;
        json["buffers"] = Json::array({{{"byteLength", bytes.size()}}});
        const fs::path path = dir / (name + (glb ? ".glb" : ".gltf"));
        if (!glb) {
            json["buffers"][0]["uri"] = name + ".bin";
            std::ofstream bin(dir / (name + ".bin"), std::ios::binary);
            bin.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            std::ofstream(path) << json.dump();
        } else {
            std::string text = json.dump();
            while (text.size() % 4) text.push_back(' ');
            std::vector<uint8_t> bin = bytes;
            while (bin.size() % 4) bin.push_back(0);
            std::ofstream file(path, std::ios::binary);
            auto word = [&](uint32_t v) { file.write(reinterpret_cast<const char*>(&v), 4); };
            word(0x46546C67); word(2); word((uint32_t)(28 + text.size() + bin.size()));
            word((uint32_t)text.size()); word(0x4E4F534A); file.write(text.data(), text.size());
            word((uint32_t)bin.size()); word(0x004E4942);
            file.write(reinterpret_cast<const char*>(bin.data()), bin.size());
        }
        return path;
    }
};

static LoadedScene Load(const fs::path& path) {
    std::map<std::string, uint32_t> textures;
    std::vector<TextureData> a, n, r;
    return ObjLoader::loadGlbFile(path.string(), textures, a, n, r, path.parent_path().string() + "/");
}

static void TestGpuNormals(Scene& scene, ID3D12Device* device, const fs::path& shaderPath) {
    auto check = [](HRESULT hr, std::source_location where = std::source_location::current()) {
        if (FAILED(hr)) throw std::runtime_error("D3D12 normal regression failed at line " +
            std::to_string(where.line()) + ", HRESULT=" + std::to_string((uint32_t)hr));
    };
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    D3D12_COMMAND_QUEUE_DESC q{};
    check(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&commands)));

    scene.BuildGlobalMeshBuffers(device, commands.Get());
    scene.UploadMaterials(device);
    scene.CreateInstancePropertiesBuffer(device);
    scene.PrepareInstanceProperties();
    scene.UploadInstanceProperties();

    auto buffer = [&](size_t bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                      D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        ComPtr<ID3D12Resource> resource;
        CD3DX12_HEAP_PROPERTIES heap(type);
        auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);
        check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&resource)));
        return resource;
    };
    auto upload = [&](const void* source, size_t bytes) {
        auto resource = buffer(bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped;
        check(resource->Map(0, nullptr, &mapped));
        memcpy(mapped, source, bytes);
        resource->Unmap(0, nullptr);
        return resource;
    };
    scene.CreateEmissiveTrianglesBuffer(device, commands.Get(), queue.Get(), allocator.Get());
    scene.CreateTriToLightIdBuffer(device, commands.Get());
    Require(scene.pendingLightUploads.size() == 2, "Light copy sources were released before GPU execution");
    const uint32_t pushDefaults[256] = {};
    auto push = upload(pushDefaults, sizeof(pushDefaults));
    auto output = buffer(6 * sizeof(XMFLOAT4), D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = buffer(6 * sizeof(XMFLOAT4), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    const UINT registers[] = {1,2,3,4,5,6,15};
    CD3DX12_ROOT_PARAMETER params[9];
    for (UINT i = 0; i < 7; ++i) params[i].InitAsShaderResourceView(registers[i]);
    params[7].InitAsUnorderedAccessView(0, 1);
    params[8].InitAsConstantBufferView(1);
    CD3DX12_ROOT_SIGNATURE_DESC desc(9, params, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED |
        D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED);
    ComPtr<ID3DBlob> blob, errors;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors));
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)));
    std::ifstream file(shaderPath, std::ios::binary);
    Require(bool(file), "Missing GPU normal regression shader");
    std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.CS = {code.data(), code.size()};
    ComPtr<ID3D12PipelineState> pso;
    check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso)));
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 1;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)));
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    ComPtr<ID3D12DescriptorHeap> samplers;
    check(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&samplers)));
    D3D12_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    device->CreateSampler(&sampler, samplers->GetCPUDescriptorHandleForHeapStart());
    ID3D12DescriptorHeap* heaps[] = {heap.Get(), samplers.Get()};
    commands->SetDescriptorHeaps(2, heaps);
    commands->SetComputeRootSignature(root.Get());
    commands->SetPipelineState(pso.Get());
    ID3D12Resource* inputs[] = {scene.indexGlobal.Get(), scene.vertexGlobal.Get(), scene.instanceProperties.Get(),
        scene.materialIndexBuffer.Get(), scene.materialBuffer.Get(), scene.emissiveTrianglesBuffer.Get(), scene.triToLightIdBuffer.Get()};
    for (UINT i = 0; i < 7; ++i) commands->SetComputeRootShaderResourceView(i, inputs[i]->GetGPUVirtualAddress());
    commands->SetComputeRootUnorderedAccessView(7, output->GetGPUVirtualAddress());
    commands->SetComputeRootConstantBufferView(8, push->GetGPUVirtualAddress());
    commands->Dispatch(1, 1, 1);
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands->ResourceBarrier(1, &barrier);
    commands->CopyResource(readback.Get(), output.Get());
    check(commands->Close());
    ID3D12CommandList* lists[] = {commands.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    check(queue->Signal(fence.Get(), 1));
    HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    Require(event != nullptr, "Could not create GPU completion event");
    HRESULT eventStatus = fence->SetEventOnCompletion(1, event);
    DWORD wait = SUCCEEDED(eventStatus) ? WaitForSingleObject(event, 30000) : WAIT_FAILED;
    CloseHandle(event);
    Require(wait == WAIT_OBJECT_0, "GPU normal regression timed out");
    scene.ReleaseLightUploadStaging();
    Require(scene.pendingLightUploads.empty(), "Completed light staging was retained");
    XMFLOAT4 values[6];
    void* mapped;
    check(readback->Map(0, nullptr, &mapped));
    memcpy(values, mapped, sizeof(values));
    readback->Unmap(0, nullptr);
    for (int instance = 0; instance < 2; ++instance) {
        const float y = instance == 0 ? -0.8f : -3.0f / std::sqrt(13.0f);
        const float z = instance == 0 ? 0.6f : 2.0f / std::sqrt(13.0f);
        for (int kind = 0; kind < 3; ++kind) {
            const auto& v = values[3 * instance + kind];
            Near(v.x, 0, "GPU surface/shadow/light normal X differs");
            Near(v.y, y, "GPU surface/shadow/light normal Y differs");
            Near(v.z, z, "GPU surface/shadow/light normal Z differs");
        }
        Near(values[3 * instance].w, (float)instance, "GPU light ID differs between instances");
        Near(values[3 * instance + 2].w, (float)instance, "GPU light instance mapping differs");
    }
    std::cout << "PASS: GPU surface, shadow and emissive normals for scaled and reflected instances.\n";
}

static void TestFiles(const fs::path& dir, bool glb) {
    Fixture f;
    // Two nodes reference the same mesh. Matrix and TRS forms must agree,
    // including noncommuting parent rotation/scale and node translation.
    f.doc["nodes"] = Json::array({
        {{"translation", {10,20,30}}, {"rotation", {0,0,0.7071067811865476,0.7071067811865476}},
         {"scale", {2,3,4}}, {"children", {1,2}}},
        {{"mesh", 0}, {"translation", {1,2,3}}, {"scale", {2,1,1}}},
        {{"mesh", 0}, {"matrix", {2,0,0,0, 0,1,0,0, 0,0,1,0, 1,2,3,1}}},
        {{"mesh", 1}, {"translation", {999,0,0}}}});
    f.doc["meshes"].push_back(f.doc["meshes"][0]);
    f.doc["scenes"].push_back({{"nodes", {3}}});
    auto scene = Load(f.Write(dir, "shared", glb));
    Require(scene.meshes.size() == 1 && scene.instances.size() == 2, "Shared/selected scene geometry duplicated");
    for (auto& [mesh, transform] : scene.instances) {
        Require(mesh == 0, "Mesh not reused");
        Position(transform, {1,0,0}, {4,26,42});
    }
    Position(XMMatrixIdentity(), scene.meshes[0].vertices[1].position, {1,0,0});

    f.doc["scene"] = 1;
    scene = Load(f.Write(dir, "selected", glb));
    Require(scene.instances.size() == 1 && scene.meshes.size() == 1, "Default scene ignored");
    Position(scene.instances[0].second, {0,0,0}, {999,0,0});
    f.doc.erase("scene");
    scene = Load(f.Write(dir, "first_scene", glb));
    Require(scene.instances.size() == 2, "First scene fallback failed");
    f.doc.erase("scenes");
    scene = Load(f.Write(dir, "forest", glb));
    Require(scene.instances.size() == 3, "Root forest fallback duplicated children");

    Fixture batch;
    const int t = batch.Accessor(batch.View<float>({123,1,2,3, 456,4,5,6}, 16), 5126, "VEC3", 2);
    batch.doc["accessors"][t]["byteOffset"] = 4;
    const int r = batch.Accessor(batch.View<int16_t>({0,0,0,32767, 0,0,32767,0}), 5122, "VEC4", 2, true);
    const int s = batch.Accessor(batch.View<float>({2,3,4, -1,2,3}), 5126, "VEC3", 2);
    batch.Instances({{"TRANSLATION", t}, {"ROTATION", r}, {"SCALE", s}});
    batch.doc["nodes"][0]["translation"] = {10,0,0};
    batch.doc["nodes"][0]["children"] = {1};
    batch.doc["nodes"].push_back({{"mesh", 0}, {"translation", {0,10,0}}});
    auto path = batch.Write(dir, "batch", glb);
    scene = Load(path);
    Require(scene.meshes.size() == 1 && scene.instances.size() == 3, "Extension count/fallback/children incorrect");
    Position(scene.instances[0].second, {1,0,0}, {13,2,3});
    Position(scene.instances[1].second, {1,0,0}, {15,5,6});
    Position(scene.instances[2].second, {0,0,0}, {10,10,0});

    batch.doc["nodes"].push_back({{"translation", {100,200,300}}, {"scale", {2,3,4}},
        {"rotation", {0,0,0.7071067811865476,0.7071067811865476}}, {"children", {0}}});
    batch.doc["scenes"][0]["nodes"] = {2};
    scene = Load(batch.Write(dir, "nested_batch", glb));
    Require(scene.instances.size() == 3, "Nested extension instances missing");
    Position(scene.instances[0].second, {1,0,0}, {94,226,312});
    Position(scene.instances[1].second, {1,0,0}, {85,230,324});
    Position(scene.instances[2].second, {0,0,0}, {70,220,300});

    Fixture packed;
    int q = packed.Accessor(packed.View<int8_t>({0,0,-128,0, 0,0,0,127}), 5120, "VEC4", 2, true);
    packed.Instances({{"ROTATION", q}});
    scene = Load(packed.Write(dir, "packed_rotation", glb));
    Require(scene.instances.size() == 2, "Packed rotations missing");
    Position(scene.instances[0].second, {1,0,0}, {-1,0,0});
    Position(scene.instances[1].second, {1,0,0}, {1,0,0});

    Fixture sparse;
    const int sparseT = sparse.Accessor(-1, 5126, "VEC3", 2);
    const int sparseI = sparse.View<uint8_t>({1});
    const int sparseV = sparse.View<float>({7,8,9});
    sparse.doc["accessors"][sparseT]["sparse"] = {{"count", 1},
        {"indices", {{"bufferView", sparseI}, {"componentType", 5121}}},
        {"values", {{"bufferView", sparseV}}}};
    sparse.Instances({{"TRANSLATION", sparseT}});
    scene = Load(sparse.Write(dir, "sparse", glb));
    Require(scene.instances.size() == 2, "Sparse instances missing");
    Position(scene.instances[0].second, {0,0,0}, {0,0,0});
    Position(scene.instances[1].second, {0,0,0}, {7,8,9});
    // Sparse overrides must also work over a populated base accessor.
    sparse.doc["accessors"][sparseT]["bufferView"] = sparse.View<float>({1,2,3, 4,5,6});
    scene = Load(sparse.Write(dir, "sparse_base", glb));
    Position(scene.instances[0].second, {0,0,0}, {1,2,3});
    Position(scene.instances[1].second, {0,0,0}, {7,8,9});

    Fixture custom;
    const int id = custom.Accessor(custom.View<uint16_t>({1,2,3}), 5123, "SCALAR", 3);
    custom.Instances({{"_ID", id}});
    scene = Load(custom.Write(dir, "custom_only", glb));
    Require(scene.instances.size() == 3, "Custom-only attributes must supply instance count");
    for (auto& inst : scene.instances) Position(inst.second, {1,2,3}, {1,2,3});

    batch.doc["accessors"][r]["count"] = 1;
    Require(Load(batch.Write(dir, "bad_count", glb)).instances.empty(), "Mismatched counts accepted");
    batch.doc["accessors"][r]["count"] = 2;
    batch.doc["accessors"][t]["byteOffset"] = 100000;
    Require(Load(batch.Write(dir, "bad_span", glb)).instances.empty(), "Out-of-bounds accessor accepted");
    batch.doc["accessors"][t]["byteOffset"] = 4;
    batch.doc["accessors"][r]["normalized"] = false;
    Require(Load(batch.Write(dir, "bad_rotation", glb)).instances.empty(), "Unnormalized integer rotation accepted");
    sparse.doc["bufferViews"][sparseI]["byteLength"] = 0;
    Require(Load(sparse.Write(dir, "bad_sparse", glb)).instances.empty(), "Invalid sparse span accepted");

    Fixture empty;
    empty.doc["meshes"][0]["primitives"][0]["mode"] = 1;
    scene = Load(empty.Write(dir, "lines", glb));
    Require(scene.meshes.empty() && scene.instances.empty(), "Empty triangle mesh retained");
}

static void TestScene(const fs::path& dir, ID3D12Device* device) {
    Fixture f;
    Json alpha = f.doc["meshes"][0]["primitives"][0];
    alpha["material"] = 1;
    f.doc["meshes"][0]["primitives"].push_back(alpha);
    const int t = f.Accessor(f.View<float>({0,0,0, 5,0,0}), 5126, "VEC3", 2);
    const int s = f.Accessor(f.View<float>({2,3,4, -1,2,3}), 5126, "VEC3", 2);
    f.Instances({{"TRANSLATION", t}, {"SCALE", s}});
    const auto path = f.Write(dir, "scene_pipeline", true);
    Scene scene;
    // No textures: the production upload path needs only a D3D12 device.
    AssetLoader::LoadModels({{path.string(), XMMatrixTranslation(10,20,30), "fixture"}},
        scene, device, nullptr, [] {});
    Require(scene.meshes.size() == 1 && scene.instances.size() == 2, "Asset stage expanded shared geometry");
    Require(scene.models[0].instanceCount == 2 && scene.models[0].meshCount == 1, "Incorrect model ranges");
    auto& mesh = scene.meshes[0];
    Require(!mesh.vertexBuffer && !mesh.indexBuffer, "Loader eagerly allocated BLAS build inputs");
    mesh.CreateBlasBuildInputs(device);
    Require(mesh.vertexBuffer->GetDesc().Width == mesh.cpuVertices.size() * sizeof(Vertex) &&
        mesh.indexBuffer->GetDesc().Width == mesh.cpuIndices.size() * sizeof(UINT), "Deferred BLAS inputs have wrong sizes");
    mesh.vertexBuffer.Reset();
    mesh.indexBuffer.Reset();
    Require(mesh.vertexCount == 6 && mesh.indexCount == 6, "Primitives/geometry duplicated or dropped");
    Require(mesh.opaqueTriCount == 1 && mesh.alphaTriCount == 1, "Opaque/alpha partition broken");
    Require(mesh.cpuMaterialIDs[0] != mesh.cpuMaterialIDs[1], "Primitive materials lost");
    Require(scene.instances[0].meshIndex == scene.instances[1].meshIndex, "Instances do not share GPU mesh");
    Position(scene.instances[1].worldTransform, {0,0,0}, {15,20,30});
    scene.CollectEmissiveTriangles();
    Require(scene.emissiveTriangles.size() == 2, "Emissive geometry not repeated per instance");
    Require(scene.instTriOffset[0] == 0 && scene.instTriOffset[1] == 2, "Triangle offsets overlap");
    Require(scene.triToLightId[0] == 0 && scene.triToLightId[2] == 1, "Instanced light IDs overlap");
    Require(scene.triToLightId[1] == UINT_MAX && scene.triToLightId[3] == UINT_MAX, "Non-emissive triangle became a light");
    Near(scene.emissiveTriangles[0].weight / scene.emissiveTriangles[1].weight,
        10.0f / std::sqrt(13.0f), "Scaled instance light area incorrect");
    scene.RebuildTLASInstanceList();
    Require(scene.tlasInstances.size() == 2 && scene.tlasInstances[1].hitGroupContribution == 2, "TLAS instance mapping broken");
    scene.PrepareInstanceProperties();
    Require(scene.cpuInstanceProps[0].materialBase == scene.cpuInstanceProps[1].materialBase,
        "Shared material ranges not reused");
    XMFLOAT3 n;
    XMStoreFloat3(&n, XMVector3Normalize(XMVector3TransformNormal(
        XMVectorSet(0,-1,1,0), scene.cpuInstanceProps[0].objectToWorldNormal)));
    Near(n.y, -0.8f, "Wrong normal matrix under nonuniform scaling");
    Near(n.z, 0.6f, "Wrong normal matrix under nonuniform scaling");
    TestGpuNormals(scene, device, dir.parent_path() / "gltf-normals.dxil");
    scene.models[0].position = {100,200,300};
    scene.MarkModelMoved(0);
    Position(scene.instances[1].worldTransform, {0,0,0}, {105,200,300});
    scene.PrepareInstanceProperties();
    Position(scene.tlasInstances[1].transform, {0,0,0}, {105,200,300});

    Fixture single;
    single.doc["nodes"][0]["scale"] = {-1,2,3};
    Scene merged;
    AssetLoader::LoadModels({{single.Write(dir, "mirrored_single", false).string()}},
        merged, device, nullptr, [] {});
    const auto& m = merged.meshes[0];
    const auto& a = m.cpuVertices[m.cpuIndices[0]];
    const auto& b = m.cpuVertices[m.cpuIndices[1]];
    const auto& c = m.cpuVertices[m.cpuIndices[2]];
    const XMVECTOR normal = XMVector3Cross(XMLoadFloat3(&b.position) - XMLoadFloat3(&a.position),
                                         XMLoadFloat3(&c.position) - XMLoadFloat3(&a.position));
    Require(XMVectorGetX(XMVector3Dot(normal, XMLoadFloat4(&a.normal_material))) > 0,
        "Baked mirrored winding disagrees with normals");

    // OBJ still has its ordinary single mesh, identity instance contract.
    const auto obj = dir / "triangle.obj";
    std::ofstream(obj) << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    Scene objScene;
    AssetLoader::LoadModels({{obj.string()}}, objScene, device, nullptr, [] {});
    Require(objScene.meshes.size() == 1 && objScene.instances.size() == 1 &&
        objScene.meshes[0].indexCount == 3, "OBJ loading regressed");
}

int main(int argc, char** argv) {
    try {
        const fs::path dir = argc > 1 ? argv[1] : "gltf-fixtures";
        fs::create_directories(dir);
        TestFiles(dir, false);
        TestFiles(dir, true);
        ComPtr<ID3D12Device> device;
        Require(SUCCEEDED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device))),
            "Could not create D3D12 device");
        TestScene(dir, device.Get());
        std::cout << "PASS: glTF + GLB instances, transforms, accessors, asset uploads, materials, lights, model updates, and OBJ.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
