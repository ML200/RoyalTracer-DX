//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_SCENEOBJECT_H
#define PATHTRACER_SCENEOBJECT_H


#include <vector>
#include "../Components/Transform.h"
#include "../Components/Material.h"
#include "../Components/Mesh.h"

class SceneObject {
public:
    Mesh mesh;
    Transform transform;
};


#endif //PATHTRACER_SCENEOBJECT_H
