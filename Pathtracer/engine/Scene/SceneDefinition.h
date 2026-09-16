#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <DirectXMath.h>
#include "../../rdn/minecraft/mc_config.h"
using namespace DirectX;

class SceneManager;
class FlyCamController;
class Renderer;

struct MeshDefinition {
    std::string path;
    XMMATRIX transform = XMMatrixIdentity();
    std::string name = "";
    std::shared_ptr<mc::MinecraftWorldConfig> minecraft;
};

inline MeshDefinition MinecraftWorld(const std::string& worldDir, std::vector<std::string> resourcePacks = {},
                                     const XMMATRIX& transform = XMMatrixIdentity(), std::string name = "") {
    MeshDefinition d;
    d.path = worldDir;
    d.transform = transform;
    d.name = std::move(name);
    d.minecraft = std::make_shared<mc::MinecraftWorldConfig>();
    d.minecraft->worldDir = worldDir;
    d.minecraft->resourcePacks = std::move(resourcePacks);
    return d;
}

class SceneDefinition {
  public:
    virtual ~SceneDefinition() = default;
    virtual std::vector<MeshDefinition> GetMeshes() = 0;
    virtual void Init(SceneManager& sceneManager, Renderer& renderer) {}
    virtual void Update(float dt, SceneManager& sceneManager, FlyCamController& flyCam) {}
};
