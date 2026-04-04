#pragma once
#include "SceneManager.h"
#include "../../rdn/Renderer.h"
#include <vector>
#include <random>

struct CubeInstance {
    uint32_t objectId;
    XMFLOAT3 velocity, targetVelocity;
    float    changeTimer;
};

class EmissiveCubes {
public:
    struct Params {
        int count = 100; float cubeSize = 0.12f;
        float emissiveFraction = 1.0f;  // 0..1: fraction of cubes that are emissive
        float emissionMin = 2.0f, emissionMax = 15.0f;
        float speedMin = 0.3f, speedMax = 2.0f;
        XMFLOAT3 spawnMin = {-10,0.2f,-10}, spawnMax = {10,5,10};
        uint32_t seed = 42;
    };
    void Init(const Params& params, SceneManager& sm, Renderer& renderer);
    void Update(float dt, SceneManager& sm);
private:
    std::vector<CubeInstance> m_cubes;
    Params m_params;
    std::mt19937 m_rng;
    static void GenCube(float h, UINT matIdx, std::vector<Vertex>& V, std::vector<UINT>& I);
};
