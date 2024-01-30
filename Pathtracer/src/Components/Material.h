//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_MATERIAL_H
#define PATHTRACER_MATERIAL_H


#include "../Math/Vector4.h"

struct Material {
    Vector4 diffuseColor; // RGBA color
    Vector4 specularColor; // RGBA color
    float specularPower;
    // Add other material properties as needed

    Material(const Vector4& diffuse, const Vector4& specular, float specPower)
            : diffuseColor(diffuse), specularColor(specular), specularPower(specPower) {}
};


#endif //PATHTRACER_MATERIAL_H
