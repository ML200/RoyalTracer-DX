#pragma once

#include <cmath>

namespace mc {

struct Placement {
    double a[9]  = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
    double t[3]  = { 0, 0, 0 };
    double ai[9] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
    bool   identity = true;

    // m: block-to-scene affine transform.
    void set(const float m[16]) {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) a[c * 3 + r] = (double)m[r * 4 + c];
        for (int c = 0; c < 3; ++c) t[c] = (double)m[12 + c];
        const double det = determinant();
        if (std::fabs(det) > 1e-30) {
            const double id = 1.0 / det;
            ai[0] =  (a[4] * a[8] - a[5] * a[7]) * id;
            ai[1] = -(a[1] * a[8] - a[2] * a[7]) * id;
            ai[2] =  (a[1] * a[5] - a[2] * a[4]) * id;
            ai[3] = -(a[3] * a[8] - a[5] * a[6]) * id;
            ai[4] =  (a[0] * a[8] - a[2] * a[6]) * id;
            ai[5] = -(a[0] * a[5] - a[2] * a[3]) * id;
            ai[6] =  (a[3] * a[7] - a[4] * a[6]) * id;
            ai[7] = -(a[0] * a[7] - a[1] * a[6]) * id;
            ai[8] =  (a[0] * a[4] - a[1] * a[3]) * id;
        }
        identity = true;
        for (int i = 0; i < 9; ++i) if (std::fabs(a[i] - ((i % 4 == 0) ? 1.0 : 0.0)) > 1e-12) identity = false;
        for (int i = 0; i < 3; ++i) if (std::fabs(t[i]) > 1e-12) identity = false;
    }
    void to_scene(const double b[3], double s[3]) const {
        for (int r = 0; r < 3; ++r) s[r] = a[r * 3] * b[0] + a[r * 3 + 1] * b[1] + a[r * 3 + 2] * b[2] + t[r];
    }
    void to_blocks(const double s[3], double b[3]) const {
        const double d[3] = { s[0] - t[0], s[1] - t[1], s[2] - t[2] };
        for (int r = 0; r < 3; ++r) b[r] = ai[r * 3] * d[0] + ai[r * 3 + 1] * d[1] + ai[r * 3 + 2] * d[2];
    }
    void dir_to_scene(const double b[3], double s[3]) const {
        for (int r = 0; r < 3; ++r) s[r] = a[r * 3] * b[0] + a[r * 3 + 1] * b[1] + a[r * 3 + 2] * b[2];
    }
    double determinant() const {
        return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
    }
    double area_scale() const { return std::pow(std::fabs(determinant()), 2.0 / 3.0); }
};

}
