/******************************************************************************
 * Copyright 1986, 2017 NVIDIA Corporation. All rights reserved.
 ******************************************************************************/
#include "stdafx.h"

#include "manipulator.h"

#include "./glm/glm.hpp"
#include "./glm/gtx/rotate_vector.hpp"

namespace nv_helpers_dx12 {

Manipulator::Manipulator() {
    update();
}

void Manipulator::setLookat(const glm::vec3& eye, const glm::vec3& center, const glm::vec3& up) {
    m_pos = eye;
    m_int = center;
    m_up = up;
    update();
}

void Manipulator::getLookat(glm::vec3& eye, glm::vec3& center, glm::vec3& up) const {
    eye = m_pos;
    center = m_int;
    up = m_up;
}

void Manipulator::setMode(Modes mode) {
    m_mode = mode;
}

Manipulator::Modes Manipulator::getMode() const {
    return m_mode;
}

void Manipulator::setRoll(float roll) {
    m_roll = roll;
    update();
}

float Manipulator::getRoll() const {
    return m_roll;
}

const glm::mat4& Manipulator::getMatrix() const {
    return m_matrix;
}

void Manipulator::setSpeed(float speed) {
    m_speed = speed;
}

float Manipulator::getSpeed() {
    return m_speed;
}

void Manipulator::setMousePosition(int x, int y) {
    m_mouse[0] = static_cast<float>(x);
    m_mouse[1] = static_cast<float>(y);
}

void Manipulator::getMousePosition(int& x, int& y) {
    x = static_cast<int>(m_mouse[0]);
    y = static_cast<int>(m_mouse[1]);
}

void Manipulator::setWindowSize(int w, int h) {
    m_width = w;
    m_height = h;
}

void Manipulator::motion(int x, int y, int action) {
    float dx = float(x - m_mouse[0]) / float(m_width);
    float dy = float(y - m_mouse[1]) / float(m_height);

    switch (action) {
    case Manipulator::Orbit:
        if (m_mode == Trackball)
            orbit(dx, dy, true);
        else
            orbit(dx, dy, false);
        break;
    case Manipulator::Dolly:
        dolly(dx, dy);
        break;
    case Manipulator::Pan:
        pan(dx, dy);
        break;
    case Manipulator::LookAround:
        if (m_mode == Trackball)
            trackball(x, y);
        else
            orbit(dx, -dy, true);
        break;
    }

    update();

    m_mouse[0] = static_cast<float>(x);
    m_mouse[1] = static_cast<float>(y);
}

Manipulator::Actions Manipulator::mouseMove(int x, int y, const Inputs& inputs) {
    Actions curAction = None;
    if (inputs.lmb) {
        if (((inputs.ctrl) && (inputs.shift)) || inputs.alt)
            curAction = m_mode == Examine ? LookAround : Orbit;
        else if (inputs.shift)
            curAction = Dolly;
        else if (inputs.ctrl)
            curAction = Pan;
        else
            curAction = m_mode == Examine ? Orbit : LookAround;
    } else if (inputs.mmb)
        curAction = Pan;
    else if (inputs.rmb)
        curAction = Dolly;

    if (curAction != None)
        motion(x, y, curAction);

    return curAction;
}

void Manipulator::wheel(int value) {
    float fval(static_cast<float>(value));
    float dx = (fval * fabs(fval)) / static_cast<float>(m_width);

    glm::vec3 z(m_pos - m_int);
    float length = z.length() * 0.1f;
    length = length < 0.001f ? 0.001f : length;

    dolly(dx * m_speed, dx * m_speed);
    update();
}

int Manipulator::getWidth() const {
    return m_width;
}

int Manipulator::getHeight() const {
    return m_height;
}

void Manipulator::trackball(int x, int y) {
    glm::vec2 p0(2 * (m_mouse[0] - m_width / 2) / double(m_width), 2 * (m_height / 2 - m_mouse[1]) / double(m_height));
    glm::vec2 p1(2 * (x - m_width / 2) / double(m_width), 2 * (m_height / 2 - y) / double(m_height));

    glm::vec3 pTB0(p0[0], p0[1], projectOntoTBSphere(p0));
    glm::vec3 pTB1(p1[0], p1[1], projectOntoTBSphere(p1));

    glm::vec3 axis = glm::cross(pTB0, pTB1);
    axis = glm::normalize(axis);

    double t = glm::length(pTB0 - pTB1) / (2.f * m_tbsize);

    if (t > 1.0)
        t = 1.0;
    else if (t < -1.0)
        t = -1.0;

    float rad = (float)(2.0 * asin(t));

    {
        glm::vec4 rot_axis = m_matrix * glm::vec4(axis, 0);
        glm::mat4 rot_mat = glm::rotate(rad, glm::vec3(rot_axis.x, rot_axis.y, rot_axis.z));

        glm::vec3 pnt = m_pos - m_int;
        glm::vec4 pnt2 = rot_mat * glm::vec4(pnt.x, pnt.y, pnt.z, 1);
        m_pos = m_int + glm::vec3(pnt2.x, pnt2.y, pnt2.z);
        glm::vec4 up2 = rot_mat * glm::vec4(m_up.x, m_up.y, m_up.z, 0);
        m_up = glm::vec3(up2.x, up2.y, up2.z);
    }
}

double Manipulator::projectOntoTBSphere(const glm::vec2& p) {
    double z;
    double d = length(p);
    if (d < m_tbsize * 0.70710678118654752440) {

        z = sqrt(m_tbsize * m_tbsize - d * d);
    } else {

        double t = m_tbsize / 1.41421356237309504880;
        z = t * t / d;
    }

    return z;
}

void Manipulator::update() {
    m_matrix = glm::lookAt(m_pos, m_int, m_up);

    if (!isZero(m_roll)) {
        glm::mat4 rot = glm::rotate(m_roll, glm::vec3(0, 0, 1));
        m_matrix = m_matrix * rot;
    }
}

void Manipulator::pan(float dx, float dy) {
    if (m_mode == Fly) {
        dx *= -1;
        dy *= -1;
    }

    glm::vec3 z(m_pos - m_int);
    float length = static_cast<float>(glm::length(z)) / 0.785f;
    z = glm::normalize(z);
    glm::vec3 x = glm::cross(m_up, z);
    x = glm::normalize(x);
    glm::vec3 y = glm::cross(z, x);
    y = glm::normalize(y);
    x *= -dx * length;
    y *= dy * length;

    m_pos += x + y;
    m_int += x + y;
}

void Manipulator::orbit(float dx, float dy, bool invert) {
    if (isZero(dx) && isZero(dy))
        return;

    dx *= float(glm::two_pi<float>());
    dy *= float(glm::two_pi<float>());

    glm::vec3 origin(invert ? m_pos : m_int);
    glm::vec3 position(invert ? m_int : m_pos);

    glm::vec3 centerToEye(position - origin);
    float radius = glm::length(centerToEye);
    centerToEye = glm::normalize(centerToEye);

    glm::mat4 rot_x, rot_y;

    glm::vec3 axe_z(glm::normalize(centerToEye));
    rot_y = glm::rotate(dx, m_up);

    glm::vec4 vect_tmp = rot_y * glm::vec4(centerToEye.x, centerToEye.y, centerToEye.z, 0);
    centerToEye = glm::vec3(vect_tmp.x, vect_tmp.y, vect_tmp.z);

    glm::vec3 axe_x = glm::cross(m_up, axe_z);
    axe_x = glm::normalize(axe_x);
    rot_x = glm::rotate(dy, axe_x);

    vect_tmp = rot_x * glm::vec4(centerToEye.x, centerToEye.y, centerToEye.z, 0);
    glm::vec3 vect_rot(vect_tmp.x, vect_tmp.y, vect_tmp.z);
    if (sign(vect_rot.x) == sign(centerToEye.x))
        centerToEye = vect_rot;

    centerToEye *= radius;

    glm::vec3 newPosition = centerToEye + origin;

    if (!invert) {
        m_pos = newPosition;
    } else {
        m_int = newPosition;
    }
}

void Manipulator::dolly(float dx, float dy) {
    glm::vec3 z = m_int - m_pos;
    float length = static_cast<float>(glm::length(z));

    if (isZero(length))
        return;

    float dd;
    if (m_mode != Examine)
        dd = -dy;
    else
        dd = fabs(dx) > fabs(dy) ? dx : -dy;
    float factor = m_speed * dd / length;

    length /= 10;
    length = length < 0.001f ? 0.001f : length;
    factor *= length;

    if (factor >= 1.0f)
        return;

    z *= factor;

    if (m_mode == Walk) {
        if (m_up.y > m_up.z)
            z.y = 0;
        else
            z.z = 0;
    }

    m_pos += z;

    if (m_mode != Examine)
        m_int += z;
}

} // namespace nv_helpers_dx12
