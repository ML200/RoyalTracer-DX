#pragma once
#include "GameObject.h"
#include "../../rdn/Scene/Scene.h"
#include <vector>
#include <unordered_set>
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
    std::vector<GameObject>      m_objects;
    std::vector<uint32_t>        m_freeList;
    uint32_t                     m_nextId = 1;
    bool                         m_structuralChange = false;
    std::unordered_set<uint32_t> m_dirtyTransforms;
    size_t m_engineInstanceBase = 0;
    bool   m_firstSync = true;
};
