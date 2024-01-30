//
// Created by m on 30.01.2024.
//

#ifndef PATHTRACER_HELPERS_H
#define PATHTRACER_HELPERS_H


#include <cmath>

static bool isNearlyEqual(float a, float b, float epsilon = 1e-6f) {
    return std::fabs(a - b) < epsilon;
}

#endif //PATHTRACER_HELPERS_H
