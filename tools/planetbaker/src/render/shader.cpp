#include "render/shader.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <GL/glew.h>

namespace pb {

static uint32_t compile(GLenum stage, std::string_view src) {
    GLuint id = glCreateShader(stage);
    const char* p = src.data();
    GLint len = static_cast<GLint>(src.size());
    glShaderSource(id, 1, &p, &len);
    glCompileShader(id);

    GLint ok = GL_FALSE;
    glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint log_len = 0;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &log_len);
        std::vector<char> log(static_cast<size_t>(log_len) + 1, 0);
        glGetShaderInfoLog(id, log_len, nullptr, log.data());
        std::fprintf(stderr, "Shader compile failed:\n%s\n", log.data());
        std::abort();
    }
    return id;
}

ShaderProgram::ShaderProgram(std::string_view vs, std::string_view fs) {
    GLuint v = compile(GL_VERTEX_SHADER, vs);
    GLuint f = compile(GL_FRAGMENT_SHADER, fs);

    program_ = glCreateProgram();
    glAttachShader(program_, v);
    glAttachShader(program_, f);
    glLinkProgram(program_);

    GLint ok = GL_FALSE;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint log_len = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &log_len);
        std::vector<char> log(static_cast<size_t>(log_len) + 1, 0);
        glGetProgramInfoLog(program_, log_len, nullptr, log.data());
        std::fprintf(stderr, "Shader link failed:\n%s\n", log.data());
        std::abort();
    }

    glDetachShader(program_, v);
    glDetachShader(program_, f);
    glDeleteShader(v);
    glDeleteShader(f);
}

ShaderProgram::~ShaderProgram() {
    if (program_) glDeleteProgram(program_);
}

void ShaderProgram::use() const {
    glUseProgram(program_);
}

int ShaderProgram::uniform(const char* name) const {
    return glGetUniformLocation(program_, name);
}

}
