#pragma once

#include <cstdint>
#include <string_view>

namespace pb {

class ShaderProgram {
public:
    ShaderProgram(std::string_view vs, std::string_view fs);
    ~ShaderProgram();

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    void use() const;
    int  uniform(const char* name) const;
    uint32_t id() const { return program_; }

private:
    uint32_t program_ = 0;
};

}
