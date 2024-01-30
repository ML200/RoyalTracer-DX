//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_TRANSFORM_H
#define PATHTRACER_TRANSFORM_H


#include "../Math/Vector3.h"
#include "../Math/Quaternion.h"

class Transform {
public:
    Vector3 position;
    Quaternion rotation;
    Vector3 scale;

    void Translate(Vector3 translation);
    void Rotate(Quaternion angle);
    void RotateEuler(Vector3 angle);
    void Scale(Vector3 factor);
};


#endif //PATHTRACER_TRANSFORM_H
