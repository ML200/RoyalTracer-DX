/******************************************************************************
 * Copyright 1986, 2017 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#pragma once

#include "./glm/glm.hpp"

namespace nv_helpers_dx12 {

class Manipulator {
  public:
    enum Modes { Examine, Fly, Walk, Trackball };
    enum Actions { None, Orbit, Dolly, Pan, LookAround };
    struct Inputs {
        bool lmb = false;
        bool mmb = false;
        bool rmb = false;
        bool shift = false;
        bool ctrl = false;
        bool alt = false;
    };

    Actions mouseMove(int x, int y, const Inputs& inputs);

    void setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up);

    void setWindowSize(int w, int h);

    void setMousePosition(int x, int y);

    static Manipulator& Singleton() {
        static Manipulator manipulator;
        return manipulator;
    }

    void getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const;

    void setMode(Modes mode);

    Modes getMode() const;

    void setRoll(float roll);

    float getRoll() const;

    const glm::mat4& getMatrix() const;

    void setSpeed(float speed);

    float getSpeed();

    void getMousePosition(int& x, int& y);

    void motion(int x, int y, int action = 0);

    void wheel(int value);

    int getWidth() const;

    int getHeight() const;

  protected:
    Manipulator();

  private:
    void update();

    void pan(float dx, float dy);

    void orbit(float dx, float dy, bool invert = false);

    void dolly(float dx, float dy);

    void trackball(int x, int y);

    double projectOntoTBSphere(const glm::vec2& p);

  protected:
    glm::vec3 m_pos = glm::vec3(10, 10, 10);
    glm::vec3 m_int = glm::vec3(0, 0, 0);
    glm::vec3 m_up = glm::vec3(0, 1, 0);
    float m_roll = 0;
    glm::mat4 m_matrix = glm::mat4(1);

    int m_width = 1;
    int m_height = 1;

    float m_speed = 30;
    glm::vec2 m_mouse = glm::vec2(0, 0);

    bool m_button = false;
    bool m_moving = false;
    float m_tbsize = 0.8f;

    Modes m_mode = Examine;
};

#define CameraManip Manipulator::Singleton()

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type areEqual(T x, T y, int ulp = 2) {

    return std::abs(x - y) < std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp

           || std::abs(x - y) < std::numeric_limits<T>::min();
}

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type areDifferent(T x, T y, int ulp = 2) {
    return !areEqual(x, y, ulp);
}

template <typename T> bool isZero(const T& _a) {
    return fabs(_a) < std::numeric_limits<T>::epsilon();
}
template <typename T> bool isOne(const T& _a) {
    return areEqual(_a, (T)1);
}

inline float sign(float s) {
    return (s < 0.f) ? -1.f : 1.f;
}
inline double sign(double s) {
    return (s < 0.0) ? -1.0 : 1.0;
}

} // namespace nv_helpers_dx12
