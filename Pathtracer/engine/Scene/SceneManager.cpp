#include "SceneManager.h"
#include "../../rdn/Renderer.h"

uint32_t SceneManager::Instantiate(UINT meshIndex, const Transform& transform) {
    uint32_t id = m_nextId++;
    GameObject go;
    go.id = id;
    go.transform = transform;
    go.meshIndex = meshIndex;
    const uint32_t idx = static_cast<uint32_t>(m_objects.size());
    m_objects.push_back(go);
    m_dirty.push_back(1);
    m_idToIndex[id] = idx;
    m_structuralChange = true;
    return id;
}

GameObject* SceneManager::Get(uint32_t id) {
    auto it = m_idToIndex.find(id);
    return it == m_idToIndex.end() ? nullptr : &m_objects[it->second];
}

void SceneManager::SetDirty(uint32_t id) {
    auto it = m_idToIndex.find(id);
    if (it != m_idToIndex.end()) {
        m_dirty[it->second] = 1;
        m_anyDirty = true;
    }
}

static void AppendEngineInstances(const std::vector<GameObject>& objects, Scene& scene) {
    for (const auto& obj : objects) {
        SceneInstance si;
        si.meshIndex = obj.meshIndex;
        si.modelIndex = 0;
        si.localTransform = XMMatrixIdentity();
        si.worldTransform = obj.transform.GetMatrix();
        si.prevWorldTransform = si.worldTransform;
        si.name = "EngineObject_" + std::to_string(obj.id);
        scene.instances.push_back(si);
    }
}

void SceneManager::SyncToRendererInitial(Scene& scene) {
    m_engineInstanceBase = scene.instances.size();
    m_firstSync = false;
    AppendEngineInstances(m_objects, scene);
    m_structuralChange = false;
    m_anyDirty = false;
    std::fill(m_dirty.begin(), m_dirty.end(), 0);
}

void SceneManager::SyncToRenderer(Renderer& renderer) {
    Scene& scene = renderer.GetScene();
    if (m_firstSync) {
        m_engineInstanceBase = scene.instances.size();
        m_firstSync = false;
    }

    if (m_structuralChange) {
        if (scene.instances.size() > m_engineInstanceBase)
            scene.instances.resize(m_engineInstanceBase);
        AppendEngineInstances(m_objects, scene);
        renderer.HandleSceneStructuralChange();
        m_structuralChange = false;
        m_anyDirty = false;
        std::fill(m_dirty.begin(), m_dirty.end(), 0);
        return;
    }

    if (m_anyDirty) {
        size_t engineIdx = m_engineInstanceBase;
        for (size_t i = 0; i < m_objects.size(); ++i) {
            const auto& obj = m_objects[i];
            if (m_dirty[i] && engineIdx < scene.instances.size()) {
                scene.instances[engineIdx].worldTransform = obj.transform.GetMatrix();
                scene.MarkInstanceDirty(static_cast<UINT>(engineIdx));
                m_dirty[i] = 0;
            }
            ++engineIdx;
        }
        scene.tlasDirty = true;
        scene.lightTreeDirty = true;
        m_anyDirty = false;
    }
}
