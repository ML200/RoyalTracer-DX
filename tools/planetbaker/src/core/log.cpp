#include "core/log.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <vector>

namespace pb {

namespace {

constexpr std::size_t kCapacity = 4096;

struct LogState {
    std::mutex            mu;
    std::array<LogEntry, kCapacity> ring;
    std::size_t           head  = 0;   //next write index
    std::size_t           count = 0;   //number of valid entries (<= kCapacity)
};

LogState& g_state() {
    static LogState s;
    return s;
}

const char* level_tag(LogLevel l) {
    switch (l) {
        case LogLevel::Info:  return "INFO";
        case LogLevel::Warn:  return "WARN";
        case LogLevel::Error: return "ERR ";
    }
    return "????";
}

}

void Log::write(LogLevel lvl, std::string_view category, std::string message) {
    std::fprintf(stderr, "[%s][%.*s] %s\n",
                 level_tag(lvl),
                 static_cast<int>(category.size()), category.data(),
                 message.c_str());

    auto& s = g_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.ring[s.head] = LogEntry{lvl, std::string(category), std::move(message)};
    s.head = (s.head + 1) % kCapacity;
    if (s.count < kCapacity) ++s.count;
}

std::size_t Log::drain(const Visitor& visit) {
    auto& s = g_state();
    std::lock_guard<std::mutex> lk(s.mu);
    std::size_t start = (s.head + kCapacity - s.count) % kCapacity;
    for (std::size_t i = 0; i < s.count; ++i) {
        visit(s.ring[(start + i) % kCapacity]);
    }
    return s.count;
}

void Log::reset() {
    auto& s = g_state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.head  = 0;
    s.count = 0;
}

void Log::writef(LogLevel lvl, const char* category, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n < 0) n = 0;
    std::size_t len = static_cast<std::size_t>(n);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    write(lvl, category, std::string(buf, len));
}

}
