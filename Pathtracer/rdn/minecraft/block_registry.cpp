#include "block_registry.h"
#include "nbt.h"
#include <algorithm>
#include <stdexcept>

namespace mc {

BlockRegistry::BlockRegistry() {
    BlockStateDesc air;
    air.name = "minecraft:air";
    air.canonical = air.name;
    m_states.push_back(air);
    m_info.push_back(BlockInfo{});
    m_index.emplace(air.canonical, AIR_ID);
}

bool BlockRegistry::is_air_name(std::string_view name) {
    return name == "minecraft:air" || name == "minecraft:cave_air" || name == "minecraft:void_air"
        || name == "air" || name == "cave_air" || name == "void_air";
}

std::string BlockRegistry::make_canonical(const std::string& name,
                                          const std::vector<std::pair<std::string, std::string>>& props) {
    if (props.empty()) return name;
    std::string s = name;
    s += '[';
    for (size_t i = 0; i < props.size(); ++i) {
        if (i) s += ',';
        s += props[i].first; s += '='; s += props[i].second;
    }
    s += ']';
    return s;
}

BlockId BlockRegistry::intern(const std::string& canonical, const std::string& name,
                              const std::vector<std::pair<std::string, std::string>>& props) {
    if (is_air_name(name)) return AIR_ID;
    std::lock_guard<std::mutex> lk(m_mutex);
    const auto it = m_index.find(canonical);
    if (it != m_index.end()) return it->second;
    if (m_states.size() >= INVALID_BLOCK)
        throw std::runtime_error("mc: more than 65534 distinct block states");
    const BlockId id = (BlockId)m_states.size();
    BlockStateDesc d;
    d.name = name;
    if (d.name.find(':') == std::string::npos) d.name = "minecraft:" + d.name;
    d.props = props;
    d.canonical = canonical;
    m_states.push_back(std::move(d));
    m_info.push_back(BlockInfo{});
    m_index.emplace(canonical, id);
    return id;
}

BlockId BlockRegistry::intern(const std::string& name,
                              std::vector<std::pair<std::string, std::string>> props) {
    std::sort(props.begin(), props.end());
    std::string full = name;
    if (full.find(':') == std::string::npos) full = "minecraft:" + full;
    return intern(make_canonical(full, props), full, props);
}

BlockId BlockRegistry::find(const std::string& canonical) const {
    std::lock_guard<std::mutex> lk(m_mutex);
    const auto it = m_index.find(canonical);
    return it == m_index.end() ? INVALID_BLOCK : it->second;
}

BlockId InternCache::intern(std::string_view name, const NbtValue* props) {
    m_props.clear();
    if (props && props->is_compound()) {
        for (const NbtValue& p : props->children) {
            if (p.type != NbtType::String) continue;
            m_props.emplace_back(std::string(p.name), std::string(p.str));
        }
        std::sort(m_props.begin(), m_props.end());
    }
    m_scratch.assign(name.data(), name.size());
    if (m_scratch.find(':') == std::string::npos) m_scratch.insert(0, "minecraft:");
    const std::string fullName = m_scratch;
    if (!m_props.empty()) {
        m_scratch += '[';
        for (size_t i = 0; i < m_props.size(); ++i) {
            if (i) m_scratch += ',';
            m_scratch += m_props[i].first; m_scratch += '='; m_scratch += m_props[i].second;
        }
        m_scratch += ']';
    }
    const auto it = m_local.find(m_scratch);
    if (it != m_local.end()) return it->second;
    const BlockId id = m_reg.intern(m_scratch, fullName, m_props);
    m_local.emplace(m_scratch, id);
    return id;
}

}
