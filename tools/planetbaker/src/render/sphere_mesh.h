#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace pb {

struct SphereMesh {
    std::vector<glm::vec3> positions;
    std::vector<uint32_t>  indices;
};

SphereMesh make_icosphere(int subdivisions);

}
