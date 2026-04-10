#pragma once
#include <string>
#include <vector>
#include <DirectXMath.h>
using namespace DirectX;

class SceneManager;
class FlyCamController;
class Renderer;

struct MeshDefinition {
    std::string path;
    XMMATRIX    transform = XMMatrixIdentity();
    std::string name      = "";
};

class SceneDefinition {
public:
    virtual ~SceneDefinition() = default;
    virtual std::vector<MeshDefinition> GetMeshes() = 0;
    virtual void Init(SceneManager& sceneManager, Renderer& renderer) {}
    virtual void Update(float dt, SceneManager& sceneManager, FlyCamController& flyCam) {}
};
