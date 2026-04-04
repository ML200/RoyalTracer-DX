#include "../../rdn/stdafx.h"
#include "EmissiveCubes.h"

void EmissiveCubes::GenCube(float h, UINT matIdx, std::vector<Vertex>& V, std::vector<UINT>& I) {
    auto quad = [&](XMFLOAT3 p0,XMFLOAT3 p1,XMFLOAT3 p2,XMFLOAT3 p3,XMFLOAT3 n){
        UINT b=(UINT)V.size(); float m=(float)matIdx;
        V.push_back({p0,{n.x,n.y,n.z,m},{0,0}}); V.push_back({p1,{n.x,n.y,n.z,m},{1,0}});
        V.push_back({p2,{n.x,n.y,n.z,m},{1,1}}); V.push_back({p3,{n.x,n.y,n.z,m},{0,1}});
        I.push_back(b); I.push_back(b+1); I.push_back(b+2);
        I.push_back(b); I.push_back(b+2); I.push_back(b+3);
    };
    quad({-h,-h,h},{h,-h,h},{h,h,h},{-h,h,h},{0,0,1});
    quad({h,-h,-h},{-h,-h,-h},{-h,h,-h},{h,h,-h},{0,0,-1});
    quad({h,-h,h},{h,-h,-h},{h,h,-h},{h,h,h},{1,0,0});
    quad({-h,-h,-h},{-h,-h,h},{-h,h,h},{-h,h,-h},{-1,0,0});
    quad({-h,h,h},{h,h,h},{h,h,-h},{-h,h,-h},{0,1,0});
    quad({-h,-h,-h},{h,-h,-h},{h,-h,h},{-h,-h,h},{0,-1,0});
}

void EmissiveCubes::Init(const Params& params, SceneManager& sm, Renderer& renderer) {
    m_params = params; m_rng.seed(params.seed);
    std::uniform_real_distribution<float> dEm(params.emissionMin,params.emissionMax);
    std::uniform_real_distribution<float> dH(0,1);
    std::uniform_real_distribution<float> dX(params.spawnMin.x,params.spawnMax.x);
    std::uniform_real_distribution<float> dY(params.spawnMin.y,params.spawnMax.y);
    std::uniform_real_distribution<float> dZ(params.spawnMin.z,params.spawnMax.z);
    std::uniform_real_distribution<float> dS(params.speedMin,params.speedMax);
    std::uniform_real_distribution<float> dD(-1,1);

    // Create ONE shared mesh+BLAS for the cube geometry
    std::vector<Vertex> sharedVerts; std::vector<UINT> sharedIndices;
    GenCube(params.cubeSize*0.5f, 0, sharedVerts, sharedIndices);

    Material firstMat; firstMat.Kd={1,1,1,1}; firstMat.Ke={1,1,1}; firstMat.Pr_Pm_Ps_Pc={0.5f,0,0,0}; firstMat.Ni=1;
    UINT baseMeshIdx = renderer.CreateProceduralMesh(sharedVerts, sharedIndices, firstMat);

    m_cubes.reserve(params.count);
    for (int i = 0; i < params.count; ++i) {
        float str = dEm(m_rng); float r=dH(m_rng),g=dH(m_rng),b=dH(m_rng);
        float mx = std::max({r,g,b}); if(mx<0.3f){r+=0.5f; mx=std::max({r,g,b});}
        r=(r/mx)*str; g=(g/mx)*str; b=(b/mx)*str;

        Material mat; mat.Kd={1,1,1,1}; mat.Ke={r,g,b}; mat.Pr_Pm_Ps_Pc={0.5f,0,0,0}; mat.Ni=1;

        // Share BLAS with first cube — only creates a new material entry, no GPU work
        UINT meshIdx = (i == 0) ? baseMeshIdx : renderer.CreateMeshInstance(baseMeshIdx, mat);

        Transform t; t.position = {dX(m_rng),dY(m_rng),dZ(m_rng)};
        uint32_t id = sm.Instantiate(meshIdx, t);

        float spd = dS(m_rng); XMFLOAT3 dir={dD(m_rng),dD(m_rng)*0.3f,dD(m_rng)};
        float len=sqrtf(dir.x*dir.x+dir.y*dir.y+dir.z*dir.z);
        if(len>0.001f){dir.x/=len;dir.y/=len;dir.z/=len;}
        CubeInstance ci; ci.objectId=id;
        ci.velocity={dir.x*spd,dir.y*spd,dir.z*spd}; ci.targetVelocity=ci.velocity;
        ci.changeTimer=dH(m_rng)*3; m_cubes.push_back(ci);
    }
    LOG(L"[EmissiveCubes] Spawned " << params.count << L" cubes");
}

void EmissiveCubes::Update(float dt, SceneManager& sm) {
    std::uniform_real_distribution<float> dD(-1,1),dS(m_params.speedMin,m_params.speedMax),dT(1.5f,4);
    for (auto& c : m_cubes) {
        auto* obj = sm.Get(c.objectId); if(!obj) continue;
        c.changeTimer -= dt;
        if (c.changeTimer <= 0) {
            float spd=dS(m_rng); XMFLOAT3 d={dD(m_rng),dD(m_rng)*0.3f,dD(m_rng)};
            float l=sqrtf(d.x*d.x+d.y*d.y+d.z*d.z); if(l>0.001f){d.x/=l;d.y/=l;d.z/=l;}
            c.targetVelocity={d.x*spd,d.y*spd,d.z*spd}; c.changeTimer=dT(m_rng);
        }
        float lr = std::min(1.0f, dt*2);
        c.velocity.x+=(c.targetVelocity.x-c.velocity.x)*lr;
        c.velocity.y+=(c.targetVelocity.y-c.velocity.y)*lr;
        c.velocity.z+=(c.targetVelocity.z-c.velocity.z)*lr;
        auto& p=obj->transform.position;
        p.x+=c.velocity.x*dt; p.y+=c.velocity.y*dt; p.z+=c.velocity.z*dt;
        if(p.x<m_params.spawnMin.x){p.x=m_params.spawnMin.x;c.velocity.x=fabsf(c.velocity.x);}
        if(p.x>m_params.spawnMax.x){p.x=m_params.spawnMax.x;c.velocity.x=-fabsf(c.velocity.x);}
        if(p.y<m_params.spawnMin.y){p.y=m_params.spawnMin.y;c.velocity.y=fabsf(c.velocity.y);}
        if(p.y>m_params.spawnMax.y){p.y=m_params.spawnMax.y;c.velocity.y=-fabsf(c.velocity.y);}
        if(p.z<m_params.spawnMin.z){p.z=m_params.spawnMin.z;c.velocity.z=fabsf(c.velocity.z);}
        if(p.z>m_params.spawnMax.z){p.z=m_params.spawnMax.z;c.velocity.z=-fabsf(c.velocity.z);}
        obj->transform.rotation.y += dt*30;
        sm.SetDirty(c.objectId);
    }
}
