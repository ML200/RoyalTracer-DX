//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VECTOR4_H
#define PATHTRACER_VECTOR4_H


class Vector4 {
public:

    float x,y,z,w;
    // Default constructor
    Vector4() : x(0), y(0), z(0), w(0) {}

    // Constructor with three floats
    Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
};


#endif //PATHTRACER_VECTOR4_H
