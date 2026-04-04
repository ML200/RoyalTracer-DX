#pragma once
#include "GameObject.h"
#include "../../rdn/Scene/Scene.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

class Renderer;

class SceneManager {
public:
    uint32_t Instantiate(UINT meshIndex, const Transform& transform);
    void     Destroy(uint32_t id);
    GameObject* Get(uint32_t id);
    void     SetDirty(uint32_t id);
    void     SetAllDirty();
    size_t   ActiveCount() const;

    void SyncToRenderer(Renderer& renderer);
    void SyncToRendererInitial(Scene& scene);

private:
    std::vector<GameObject>                m_objects;
    std::vector<uint32_t>                  m_freeList;
    std::unordered_map<uint32_t, uint32_t> m_idToIndex;   // id → m_objects index
    std::vector<uint8_t>                   m_dirty;       // flat dirty flags (parallel to m_objects)
    bool                                   m_anyDirty = false;
    uint32_t                               m_nextId = 1;
    bool                                   m_structuralChange = false;
    size_t m_engineInstanceBase = 0;
    bool   m_firstSync = true;
};
