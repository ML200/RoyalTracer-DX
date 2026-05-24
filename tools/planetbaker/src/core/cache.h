#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "core/field.h"
#include "core/hash.h"
#include "core/log.h"

namespace pb {

//====================================
//Per-blob header for cached field files. payload_hash is FNV-1a of the
//device bytes; M5+ uses it to derive input-field hashes for downstream
//cache keys without re-reading the whole blob.
//====================================

#pragma pack(push, 1)
struct FieldFileHeader {
    char           magic[4];      //"PBFD"
    std::uint32_t  version;       //1
    std::uint32_t  type_id;       //see FieldTypeId
    std::uint32_t  n;             //face resolution
    std::uint32_t  halo;          //halo size
    std::uint32_t  face_count;    //6
    std::uint64_t  total_cells;   //matches buffer length in cells
    std::uint64_t  payload_hash;  //FNV-1a of payload bytes
};
#pragma pack(pop)

constexpr std::uint32_t kFieldFileVersion = 1u;

//Order MUST be stable across releases; reading an existing cache depends on it.
template <typename T> struct FieldTypeId;
template <> struct FieldTypeId<float>          { static constexpr std::uint32_t value = 0; };
template <> struct FieldTypeId<float2>         { static constexpr std::uint32_t value = 1; };
template <> struct FieldTypeId<float4>         { static constexpr std::uint32_t value = 2; };
template <> struct FieldTypeId<std::uint8_t>   { static constexpr std::uint32_t value = 3; };
template <> struct FieldTypeId<std::uint32_t>  { static constexpr std::uint32_t value = 4; };

//====================================
//Cache layout. One directory per (pass_name, key); one file per field.
//Keys are formatted as 16-char hex strings. Use entry_dir() to assemble
//the path, then file_path() to add a field name.
//====================================

std::string                  format_key(std::uint64_t key);
std::filesystem::path        entry_dir(const std::filesystem::path& root,
                                       std::string_view pass_name,
                                       std::uint64_t key);
std::filesystem::path        file_path(const std::filesystem::path& root,
                                       std::string_view pass_name,
                                       std::uint64_t key,
                                       std::string_view field_name);

bool entry_exists(const std::filesystem::path& root,
                  std::string_view pass_name,
                  std::uint64_t key,
                  std::string_view field_name);

namespace detail {

bool write_blob(const std::filesystem::path& dest,
                const FieldFileHeader& hdr,
                const void* payload,
                std::size_t payload_bytes);

bool read_blob(const std::filesystem::path& src,
               FieldFileHeader& hdr,
               std::vector<std::uint8_t>& payload);

}

//====================================
//Templated field save/load. Round-trip is bit-identical for any T listed in
//FieldTypeId. The Field's existing grid is consulted for n/halo/face_count;
//on load, those values are validated against the header.
//====================================

template <typename T>
bool save_field(const std::filesystem::path& root,
                std::string_view pass_name,
                std::uint64_t key,
                std::string_view field_name,
                const Field<T>& field) {
    std::vector<T> host;
    field.download(host);
    const std::size_t payload_bytes = host.size() * sizeof(T);

    FieldFileHeader hdr{};
    hdr.magic[0] = 'P'; hdr.magic[1] = 'B'; hdr.magic[2] = 'F'; hdr.magic[3] = 'D';
    hdr.version      = kFieldFileVersion;
    hdr.type_id      = FieldTypeId<T>::value;
    hdr.n            = static_cast<std::uint32_t>(field.grid().n);
    hdr.halo         = static_cast<std::uint32_t>(CubedSphereGrid::HALO);
    hdr.face_count   = 6u;
    hdr.total_cells  = static_cast<std::uint64_t>(host.size());
    hdr.payload_hash = fnv1a64(host.data(), payload_bytes);

    auto dest = file_path(root, pass_name, key, field_name);
    if (!detail::write_blob(dest, hdr, host.data(), payload_bytes)) {
        PB_LOG_ERROR("cache", "save_field failed for %s", dest.string().c_str());
        return false;
    }
    return true;
}

template <typename T>
bool load_field(const std::filesystem::path& root,
                std::string_view pass_name,
                std::uint64_t key,
                std::string_view field_name,
                Field<T>& field) {
    auto src = file_path(root, pass_name, key, field_name);
    FieldFileHeader hdr{};
    std::vector<std::uint8_t> raw;
    if (!detail::read_blob(src, hdr, raw)) return false;

    if (hdr.type_id != FieldTypeId<T>::value) {
        PB_LOG_ERROR("cache", "load_field type mismatch on %s (got %u expected %u)",
                     src.string().c_str(), hdr.type_id, FieldTypeId<T>::value);
        return false;
    }
    if (hdr.n != static_cast<std::uint32_t>(field.grid().n)
     || hdr.halo != static_cast<std::uint32_t>(CubedSphereGrid::HALO)
     || hdr.face_count != 6u
     || hdr.total_cells != static_cast<std::uint64_t>(field.total_cells())) {
        PB_LOG_ERROR("cache", "load_field shape mismatch on %s", src.string().c_str());
        return false;
    }
    const std::size_t expected_bytes = hdr.total_cells * sizeof(T);
    if (raw.size() != expected_bytes) {
        PB_LOG_ERROR("cache", "load_field byte count mismatch on %s", src.string().c_str());
        return false;
    }
    std::uint64_t actual_hash = fnv1a64(raw.data(), raw.size());
    if (actual_hash != hdr.payload_hash) {
        PB_LOG_ERROR("cache", "load_field payload hash mismatch on %s", src.string().c_str());
        return false;
    }

    std::vector<T> host(hdr.total_cells);
    std::memcpy(host.data(), raw.data(), raw.size());
    field.upload(host);
    return true;
}

}
