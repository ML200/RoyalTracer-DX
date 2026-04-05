#include "SceneManager.h"
#include "../../rdn/Renderer.h"

uint32_t SceneManager::Instantiate(UINT meshIndex, const Transform& transform) {
    uint32_t id = m_nextId++;
    GameObject go; go.id = id; go.transform = transform; go.meshIndex = meshIndex; go.active = true;
    uint32_t idx;
    if (!m_freeList.empty()) {
        idx = m_freeList.back(); m_freeList.pop_back();
        m_objects[idx] = go;
        m_dirty[idx] = 1;
    } else {
        idx = static_cast<uint32_t>(m_objects.size());
        m_objects.push_back(go);
        m_dirty.push_back(1);
    }
    m_idToIndex[id] = idx;
    m_structuralChange = true;
    return id;
}

void SceneManager::Destroy(uint32_t id) {
    auto it = m_idToIndex.find(id);
    if (it == m_idToIndex.end()) return;
    uint32_t idx = it->second;
    m_objects[idx].active = false;
    m_dirty[idx] = 0;
    m_freeList.push_back(idx);
    m_idToIndex.erase(it);
    m_structuralChange = true;
}

GameObject* SceneManager::Get(uint32_t id) {
    auto it = m_idToIndex.find(id);
    if (it == m_idToIndex.end()) return nullptr;
    auto& obj = m_objects[it->second];
    return obj.active ? &obj : nullptr;
}

void SceneManager::SetDirty(uint32_t id) {
    auto it = m_idToIndex.find(id);
    if (it != m_idToIndex.end()) {
        m_dirty[it->second] = 1;
        m_anyDirty = true;
    }
}

void SceneManager::SetAllDirty() {
    std::fill(m_dirty.begin(), m_dirty.end(), 1);
    m_anyDirty = true;
}

size_t SceneManager::ActiveCount() const {
    return m_idToIndex.size();
}

static void AppendEngineInstances(const std::vector<GameObject>& objects, Scene& scene) {
    for (const auto& obj : objects) {
        if (!obj.active) continue;
        SceneInstance si;
        si.meshIndex = obj.meshIndex; si.modelIndex = 0;
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
    if (m_firstSync) { m_engineInstanceBase = scene.instances.size(); m_firstSync = false; }

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
            if (!obj.active) continue;
            if (m_dirty[i] && engineIdx < scene.instances.size()) {
                scene.instances[engineIdx].worldTransform = obj.transform.GetMatrix();
                scene.MarkInstanceDirty(static_cast<UINT>(engineIdx));
                m_dirty[i] = 0;
            }
            ++engineIdx;
        }
        scene.tlasDirty      = true;
        scene.lightTreeDirty = true;
        m_anyDirty = false;
    }
}
