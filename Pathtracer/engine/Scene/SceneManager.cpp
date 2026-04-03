#include "SceneManager.h"
#include "../../rdn/Renderer.h"

uint32_t SceneManager::Instantiate(UINT meshIndex, const Transform& transform) {
    uint32_t id = m_nextId++;
    GameObject go; go.id = id; go.transform = transform; go.meshIndex = meshIndex; go.active = true;
    if (!m_freeList.empty()) { uint32_t idx = m_freeList.back(); m_freeList.pop_back(); m_objects[idx] = go; }
    else m_objects.push_back(go);
    m_structuralChange = true;
    return id;
}

void SceneManager::Destroy(uint32_t id) {
    for (size_t i = 0; i < m_objects.size(); ++i)
        if (m_objects[i].id == id && m_objects[i].active) {
            m_objects[i].active = false; m_freeList.push_back((uint32_t)i);
            m_structuralChange = true; m_dirtyTransforms.erase(id); return;
        }
}

GameObject* SceneManager::Get(uint32_t id) {
    for (auto& obj : m_objects) if (obj.id == id && obj.active) return &obj;
    return nullptr;
}

void SceneManager::SetDirty(uint32_t id) { m_dirtyTransforms.insert(id); }

void SceneManager::SetAllDirty() {
    for (const auto& obj : m_objects) if (obj.active) m_dirtyTransforms.insert(obj.id);
}

size_t SceneManager::ActiveCount() const {
    size_t c = 0; for (const auto& o : m_objects) if (o.active) ++c; return c;
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
    m_dirtyTransforms.clear();
}

void SceneManager::SyncToRenderer(Renderer& renderer) {
    Scene& scene = renderer.GetScene();
    if (m_firstSync) { m_engineInstanceBase = scene.instances.size(); m_firstSync = false; }

    if (m_structuralChange) {
        if (scene.instances.size() > m_engineInstanceBase)
            scene.instances.resize(m_engineInstanceBase);
        AppendEngineInstances(m_objects, scene);
        renderer.HandleSceneStructuralChange();
        m_structuralChange = false; m_dirtyTransforms.clear();
        return;
    }

    if (!m_dirtyTransforms.empty()) {
        size_t engineIdx = m_engineInstanceBase;
        for (const auto& obj : m_objects) {
            if (!obj.active) continue;
            if (m_dirtyTransforms.count(obj.id) && engineIdx < scene.instances.size())
                scene.instances[engineIdx].worldTransform = obj.transform.GetMatrix();
            ++engineIdx;
        }
        scene.tlasDirty = true;
        m_dirtyTransforms.clear();
    }
}
