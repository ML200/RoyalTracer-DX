//
// Created by m on 30.01.2024.
//

#include "Transform.h"

void Transform::Translate(Vector3 translation) {
    position.x += translation.x;
    position.y += translation.y;
    position.z += translation.z;
}

void Transform::Rotate(Quaternion angle) {
    rotation = Quaternion::Multiply(rotation, angle).Normalized();
}

void Transform::RotateEuler(Vector3 angle) {
    Quaternion rotationAngle = Quaternion::FromEulerAngles(angle);
    Rotate(rotationAngle);
}

void Transform::Scale(Vector3 factor) {
    scale.x *= factor.x;
    scale.y *= factor.y;
    scale.z *= factor.z;
}
