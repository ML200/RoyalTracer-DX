#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace pb {

enum class LogLevel : std::uint8_t {
    Info,
    Warn,
    Error,
};

struct LogEntry {
    LogLevel    level;
    std::string category;
    std::string message;
};

//====================================
//Process-wide logger. Every write goes to stderr AND into a fixed-capacity
//ring buffer that the viewer's Log panel reads each frame. CUDA kernels
//never log directly; only host code.
//
//Use the macros for normal logging. printf-style format strings, snprintf
//under the hood, single allocation per call. The category tag is short, the
//message can be longer (~256 chars before truncation).
//====================================

class Log {
public:
    static void write(LogLevel lvl, std::string_view category, std::string message);

    //Visit recently-written entries newest-last. Visitor is called under the
    //internal mutex; keep it short. Returns the number of entries visited.
    using Visitor = std::function<void(const LogEntry&)>;
    static std::size_t drain(const Visitor& visit);

    //For "Clear" buttons.
    static void reset();

    static void writef(LogLevel lvl, const char* category, const char* fmt, ...);
};

}

#define PB_LOG_INFO(category, ...)  ::pb::Log::writef(::pb::LogLevel::Info,  (category), __VA_ARGS__)
#define PB_LOG_WARN(category, ...)  ::pb::Log::writef(::pb::LogLevel::Warn,  (category), __VA_ARGS__)
#define PB_LOG_ERROR(category, ...) ::pb::Log::writef(::pb::LogLevel::Error, (category), __VA_ARGS__)
