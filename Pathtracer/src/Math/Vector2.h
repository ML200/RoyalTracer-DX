//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VECTOR2_H
#define PATHTRACER_VECTOR2_H

#include <functional>
#include "Helpers.h"

class Vector2 {
public:
    float x,y;
    // Default constructor
    Vector2() : x(0), y(0) {}

    // Constructor with three floats
    Vector2(float x, float y) : x(x), y(y) {}

    // Equality comparison operator
    // Equality comparison operator
    bool operator==(const Vector2& other) const {
        return isNearlyEqual(x, other.x) && isNearlyEqual(y, other.y);
    }

};

// Specialize the std::hash template for the Vector2 class
namespace std {
    template <>
    struct hash<Vector2> {
        size_t operator()(const Vector2& v) const {
            // Use a combination of the hash values of x and y
            return ((hash<float>()(v.x) ^
                     (hash<float>()(v.y) << 1)) >> 1);
        }
    };
}


#endif //PATHTRACER_VECTOR2_H
