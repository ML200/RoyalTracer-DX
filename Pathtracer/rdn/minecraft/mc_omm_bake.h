#pragma once

#include <string>
#include <vector>
#include "mc_omm.h"
#include "../Scene/OmmBuilder.h"

namespace mc {

class BlockRegistry;

// One deduplicated table for all cutouts.
bool bake_omm_table(const BlockRegistry& reg, const std::vector<OmmBakeTri>& tris,
                    OmmTable& table, OmmBakeResult& out, std::string* err);

}
