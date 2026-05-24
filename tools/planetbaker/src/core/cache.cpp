#include "core/cache.h"

#include <cstdio>
#include <fstream>
#include <system_error>

namespace pb {

std::string format_key(std::uint64_t key) {
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(key));
    return std::string(buf, 16);
}

std::filesystem::path entry_dir(const std::filesystem::path& root,
                                std::string_view pass_name,
                                std::uint64_t key) {
    return root / std::string(pass_name) / format_key(key);
}

std::filesystem::path file_path(const std::filesystem::path& root,
                                std::string_view pass_name,
                                std::uint64_t key,
                                std::string_view field_name) {
    return entry_dir(root, pass_name, key) / (std::string(field_name) + ".bin");
}

bool entry_exists(const std::filesystem::path& root,
                  std::string_view pass_name,
                  std::uint64_t key,
                  std::string_view field_name) {
    std::error_code ec;
    return std::filesystem::exists(file_path(root, pass_name, key, field_name), ec);
}

namespace detail {

bool write_blob(const std::filesystem::path& dest,
                const FieldFileHeader& hdr,
                const void* payload,
                std::size_t payload_bytes) {
    std::error_code ec;
    std::filesystem::create_directories(dest.parent_path(), ec);
    if (ec) {
        PB_LOG_ERROR("cache", "mkdir failed for %s: %s",
                     dest.parent_path().string().c_str(), ec.message().c_str());
        return false;
    }

    auto tmp = dest;
    tmp += ".tmp";

    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) {
            PB_LOG_ERROR("cache", "open tmp failed for %s", tmp.string().c_str());
            return false;
        }
        os.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        if (payload_bytes > 0) {
            os.write(reinterpret_cast<const char*>(payload), static_cast<std::streamsize>(payload_bytes));
        }
        if (!os) {
            PB_LOG_ERROR("cache", "write failed for %s", tmp.string().c_str());
            return false;
        }
    }

    std::filesystem::rename(tmp, dest, ec);
    if (ec) {
        //On Windows, rename fails if dest exists. Fall back to remove + rename.
        std::filesystem::remove(dest, ec);
        std::filesystem::rename(tmp, dest, ec);
        if (ec) {
            PB_LOG_ERROR("cache", "rename failed for %s -> %s: %s",
                         tmp.string().c_str(), dest.string().c_str(), ec.message().c_str());
            return false;
        }
    }
    return true;
}

bool read_blob(const std::filesystem::path& src,
               FieldFileHeader& hdr,
               std::vector<std::uint8_t>& payload) {
    std::ifstream is(src, std::ios::binary);
    if (!is) return false;

    is.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!is || is.gcount() != static_cast<std::streamsize>(sizeof(hdr))) return false;

    if (hdr.magic[0] != 'P' || hdr.magic[1] != 'B' || hdr.magic[2] != 'F' || hdr.magic[3] != 'D') {
        PB_LOG_ERROR("cache", "bad magic in %s", src.string().c_str());
        return false;
    }
    if (hdr.version != kFieldFileVersion) {
        PB_LOG_ERROR("cache", "version %u != %u in %s",
                     hdr.version, kFieldFileVersion, src.string().c_str());
        return false;
    }

    is.seekg(0, std::ios::end);
    auto end = is.tellg();
    if (end < 0 || static_cast<std::size_t>(end) < sizeof(hdr)) return false;
    is.seekg(static_cast<std::streamoff>(sizeof(hdr)), std::ios::beg);
    std::size_t payload_bytes = static_cast<std::size_t>(end) - sizeof(hdr);

    payload.resize(payload_bytes);
    if (payload_bytes > 0) {
        is.read(reinterpret_cast<char*>(payload.data()),
                static_cast<std::streamsize>(payload_bytes));
        if (!is) return false;
    }
    return true;
}

}

}
