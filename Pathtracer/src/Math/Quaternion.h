//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_QUATERNION_H
#define PATHTRACER_QUATERNION_H


#include <cmath>

class Quaternion {
public:
    float x,y,z,w;

    // Constructors
    Quaternion() : x(0), y(0), z(0), w(1) {} // Identity quaternion
    Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    // Normalize the quaternion
    void Normalize() {
        float norm = std::sqrt(x * x + y * y + z * z + w * w);
        x /= norm;
        y /= norm;
        z /= norm;
        w /= norm;
    }

    // Static function for quaternion multiplication
    static Quaternion Multiply(const Quaternion& a, const Quaternion& b) {
        return Quaternion(
                a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
        );
    }

    // Create a quaternion from an axis and an angle (in radians)
    static Quaternion FromAxisAngle(const Vector3& axis, float angle) {
        float halfAngle = angle * 0.5f;
        float s = std::sin(halfAngle);
        return Quaternion(
                axis.x * s,
                axis.y * s,
                axis.z * s,
                std::cos(halfAngle)
        ).Normalized();
    }

    // Create a quaternion from Euler angles (pitch, yaw, roll)
    static Quaternion FromEulerAngles(const Vector3& eulerAngles) {
        // Extract individual angles (assumed to be in radians)
        float pitch = eulerAngles.x;
        float yaw = eulerAngles.y;
        float roll = eulerAngles.z;

        // Compute half angles
        float halfPitch = pitch * 0.5f;
        float halfYaw = yaw * 0.5f;
        float halfRoll = roll * 0.5f;

        // Calculate sin/cos for each half angle
        float sinPitch = std::sin(halfPitch);
        float cosPitch = std::cos(halfPitch);
        float sinYaw = std::sin(halfYaw);
        float cosYaw = std::cos(halfYaw);
        float sinRoll = std::sin(halfRoll);
        float cosRoll = std::cos(halfRoll);

        // Compute quaternion
        return Quaternion(
                cosYaw * sinPitch * cosRoll + sinYaw * cosPitch * sinRoll,
                sinYaw * cosPitch * cosRoll - cosYaw * sinPitch * sinRoll,
                cosYaw * cosPitch * sinRoll - sinYaw * sinPitch * cosRoll,
                cosYaw * cosPitch * cosRoll + sinYaw * sinPitch * sinRoll
        ).Normalized();
    }

    // Conjugate of the quaternion
    Quaternion Conjugate() const {
        return Quaternion(-x, -y, -z, w);
    }

    // Inverse of the quaternion
    Quaternion Inverse() const {
        Quaternion conj = Conjugate();
        float norm = x * x + y * y + z * z + w * w;
        return Quaternion(conj.x / norm, conj.y / norm, conj.z / norm, conj.w / norm);
    }

    // Rotate a vector by this quaternion
    Vector3 Rotate(const Vector3& vec) const {
        Quaternion vecQuat(vec.x, vec.y, vec.z, 0);
        Quaternion resQuat = Multiply(Multiply(*this, vecQuat), Inverse());
        return Vector3(resQuat.x, resQuat.y, resQuat.z);
    }

    // Normalized copy of the quaternion
    Quaternion Normalized() const {
        Quaternion q = *this;
        q.Normalize();
        return q;
    }
};


#endif //PATHTRACER_QUATERNION_H
