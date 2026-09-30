#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mc {

enum class NbtType : uint8_t {
    End = 0, Byte = 1, Short = 2, Int = 3, Long = 4, Float = 5, Double = 6,
    ByteArray = 7, String = 8, List = 9, Compound = 10, IntArray = 11, LongArray = 12,
};

struct NbtValue {
    NbtType type     = NbtType::End;
    NbtType listType = NbtType::End;
    std::string_view name;
    int64_t  i = 0;
    double   d = 0.0;
    std::string_view str;
    std::vector<int32_t> ints;
    std::vector<int64_t> longs;
    std::vector<NbtValue> children;

    bool is_compound() const { return type == NbtType::Compound; }
    bool is_list()     const { return type == NbtType::List; }
    bool is_number()   const { return type >= NbtType::Byte && type <= NbtType::Double; }

    const NbtValue* find(std::string_view key) const;
    int64_t          get_int   (std::string_view key, int64_t def = 0) const;
    double           get_double(std::string_view key, double def = 0.0) const;
    std::string_view get_string(std::string_view key) const;
    const NbtValue*  get_list  (std::string_view key) const;
    const NbtValue*  get_compound(std::string_view key) const;
};

// Big-endian; root keeps string views into data.
bool nbt_parse(const uint8_t* data, size_t size, NbtValue& root, std::string* err = nullptr);

class NbtWriter {
public:
    std::vector<uint8_t> bytes;
    void begin_compound(std::string_view name);
    void end_compound();
    void write_byte  (std::string_view name, int8_t v);
    void write_short (std::string_view name, int16_t v);
    void write_int   (std::string_view name, int32_t v);
    void write_long  (std::string_view name, int64_t v);
    void write_float (std::string_view name, float v);
    void write_double(std::string_view name, double v);
    void write_string(std::string_view name, std::string_view v);
    void write_byte_array(std::string_view name, const std::vector<int8_t>& v);
    void write_int_array (std::string_view name, const std::vector<int32_t>& v);
    void write_long_array(std::string_view name, const std::vector<int64_t>& v);
    void begin_list(std::string_view name, NbtType elem, int32_t count);
    void write_string_item(std::string_view v);
private:
    void tag_header(NbtType t, std::string_view name);
    void put_u16(uint16_t v); void put_u32(uint32_t v); void put_u64(uint64_t v);
};

}
