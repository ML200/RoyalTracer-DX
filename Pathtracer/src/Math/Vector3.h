//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_VECTOR3_H
#define PATHTRACER_VECTOR3_H

#include <functional>
#include <cmath>
#include "Helpers.h"

class Vector3 {
public:
    float x,y,z;

    // Default constructor
    Vector3() : x(0), y(0), z(0) {}

    // Constructor with three floats
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}

    // Operator overloading for vector addition
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    // Operator overloading for vector subtraction
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    // Operator overloading for scalar multiplication
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    // Operator overloading for scalar division
    Vector3 operator/(float scalar) const {
        return Vector3(x / scalar, y / scalar, z / scalar);
    }

    // Dot product of two vectors
    static float Dot(const Vector3& a, const Vector3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    // Cross product of two vectors
    static Vector3 Cross(const Vector3& a, const Vector3& b) {
        return Vector3(
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x
        );
    }

    // Equality comparison operator
    bool operator==(const Vector3& other) const {
        return isNearlyEqual(x, other.x) && isNearlyEqual(y, other.y) && isNearlyEqual(z, other.z);
    }

};

// Specialize the std::hash template for the Vector3 class
namespace std {
    template <>
    struct hash<Vector3> {
        size_t operator()(const Vector3& v) const {
            // Use a combination of the hash values of x, y, and z
            return ((hash<float>()(v.x) ^
                     (hash<float>()(v.y) << 1)) >> 1) ^
                   (hash<float>()(v.z) << 1);
        }
    };
}


#endif //PATHTRACER_VECTOR3_H
