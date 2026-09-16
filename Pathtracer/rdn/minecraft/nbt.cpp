#include "nbt.h"
#include <cstring>

namespace mc {

namespace {

struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    std::string*   err;
    int            depth = 0;

    bool fail(const char* msg) { if (err && err->empty()) *err = msg; return false; }
    bool need(size_t n) const { return (size_t)(end - p) >= n; }
    uint8_t  u8()  { return *p++; }
    uint16_t u16() { const uint16_t v = (uint16_t)((p[0] << 8) | p[1]); p += 2; return v; }
    uint32_t u32() { const uint32_t v = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]; p += 4; return v; }
    uint64_t u64() { const uint64_t hi = u32(); const uint64_t lo = u32(); return (hi << 32) | lo; }
};

bool read_string(Cursor& c, std::string_view& out) {
    if (!c.need(2)) return c.fail("truncated string length");
    const uint16_t len = c.u16();
    if (!c.need(len)) return c.fail("truncated string");
    out = std::string_view((const char*)c.p, len);
    c.p += len;
    return true;
}

bool read_payload(Cursor& c, NbtType type, NbtValue& v);

bool read_named(Cursor& c, NbtValue& v) {
    if (!c.need(1)) return c.fail("truncated tag type");
    const NbtType t = (NbtType)c.u8();
    v.type = t;
    if (t == NbtType::End) return true;
    if ((uint8_t)t > (uint8_t)NbtType::LongArray) return c.fail("bad tag type");
    if (!read_string(c, v.name)) return false;
    return read_payload(c, t, v);
}

// Recursively decodes one payload while validating cursor bounds.
bool read_payload(Cursor& c, NbtType type, NbtValue& v) {
    switch (type) {
    case NbtType::Byte:   if (!c.need(1)) return c.fail("truncated byte");   v.i = (int8_t)c.u8();   return true;
    case NbtType::Short:  if (!c.need(2)) return c.fail("truncated short");  v.i = (int16_t)c.u16(); return true;
    case NbtType::Int:    if (!c.need(4)) return c.fail("truncated int");    v.i = (int32_t)c.u32(); return true;
    case NbtType::Long:   if (!c.need(8)) return c.fail("truncated long");   v.i = (int64_t)c.u64(); return true;
    case NbtType::Float: {
        if (!c.need(4)) return c.fail("truncated float");
        const uint32_t bits = c.u32(); float f; std::memcpy(&f, &bits, 4); v.d = f; return true;
    }
    case NbtType::Double: {
        if (!c.need(8)) return c.fail("truncated double");
        const uint64_t bits = c.u64(); double d; std::memcpy(&d, &bits, 8); v.d = d; return true;
    }
    case NbtType::ByteArray: {
        if (!c.need(4)) return c.fail("truncated byte array length");
        const int32_t n = (int32_t)c.u32();
        if (n < 0 || !c.need((size_t)n)) return c.fail("truncated byte array");
        v.str = std::string_view((const char*)c.p, (size_t)n);
        c.p += n;
        return true;
    }
    case NbtType::String: return read_string(c, v.str);
    case NbtType::List: {
        if (!c.need(5)) return c.fail("truncated list header");
        const NbtType et = (NbtType)c.u8();
        const int32_t n  = (int32_t)c.u32();
        if (n < 0) return c.fail("negative list length");
        if ((uint8_t)et > (uint8_t)NbtType::LongArray) return c.fail("bad list element type");
        if (n > 0 && et == NbtType::End) return c.fail("non-empty list of TAG_End");
        if (++c.depth > 512) return c.fail("nesting too deep");
        v.listType = et;
        v.children.reserve((size_t)(n < 4096 ? n : 4096));
        for (int32_t k = 0; k < n; ++k) {
            NbtValue item;
            item.type = et;
            if (!read_payload(c, et, item)) return false;
            v.children.push_back(std::move(item));
        }
        --c.depth;
        return true;
    }
    case NbtType::Compound: {
        if (++c.depth > 512) return c.fail("nesting too deep");
        for (;;) {
            NbtValue m;
            if (!read_named(c, m)) return false;
            if (m.type == NbtType::End) break;
            v.children.push_back(std::move(m));
        }
        --c.depth;
        return true;
    }
    case NbtType::IntArray: {
        if (!c.need(4)) return c.fail("truncated int array length");
        const int32_t n = (int32_t)c.u32();
        if (n < 0 || !c.need((size_t)n * 4)) return c.fail("truncated int array");
        v.ints.resize((size_t)n);
        for (int32_t k = 0; k < n; ++k) v.ints[(size_t)k] = (int32_t)c.u32();
        return true;
    }
    case NbtType::LongArray: {
        if (!c.need(4)) return c.fail("truncated long array length");
        const int32_t n = (int32_t)c.u32();
        if (n < 0 || !c.need((size_t)n * 8)) return c.fail("truncated long array");
        v.longs.resize((size_t)n);
        for (int32_t k = 0; k < n; ++k) v.longs[(size_t)k] = (int64_t)c.u64();
        return true;
    }
    default: return c.fail("unknown tag");
    }
}

}

const NbtValue* NbtValue::find(std::string_view key) const {
    if (type != NbtType::Compound) return nullptr;
    for (const NbtValue& c : children)
        if (c.name == key) return &c;
    return nullptr;
}
int64_t NbtValue::get_int(std::string_view key, int64_t def) const {
    const NbtValue* v = find(key);
    if (!v) return def;
    if (v->type == NbtType::Float || v->type == NbtType::Double) return (int64_t)v->d;
    if (!v->is_number()) return def;
    return v->i;
}
double NbtValue::get_double(std::string_view key, double def) const {
    const NbtValue* v = find(key);
    if (!v || !v->is_number()) return def;
    if (v->type == NbtType::Float || v->type == NbtType::Double) return v->d;
    return (double)v->i;
}
std::string_view NbtValue::get_string(std::string_view key) const {
    const NbtValue* v = find(key);
    return (v && v->type == NbtType::String) ? v->str : std::string_view();
}
const NbtValue* NbtValue::get_list(std::string_view key) const {
    const NbtValue* v = find(key);
    return (v && v->type == NbtType::List) ? v : nullptr;
}
const NbtValue* NbtValue::get_compound(std::string_view key) const {
    const NbtValue* v = find(key);
    return (v && v->type == NbtType::Compound) ? v : nullptr;
}

// Parses one named root tag from a bounded input buffer.
bool nbt_parse(const uint8_t* data, size_t size, NbtValue& root, std::string* err) {
    if (err) err->clear();
    root = NbtValue{};
    Cursor c{ data, data + size, err };
    if (!data || size == 0) return c.fail("empty input");
    if (!read_named(c, root)) return false;
    if (root.type == NbtType::End) return c.fail("empty root");
    return true;
}

void NbtWriter::put_u16(uint16_t v) { bytes.push_back((uint8_t)(v >> 8)); bytes.push_back((uint8_t)v); }
void NbtWriter::put_u32(uint32_t v) { put_u16((uint16_t)(v >> 16)); put_u16((uint16_t)v); }
void NbtWriter::put_u64(uint64_t v) { put_u32((uint32_t)(v >> 32)); put_u32((uint32_t)v); }
void NbtWriter::tag_header(NbtType t, std::string_view name) {
    bytes.push_back((uint8_t)t);
    put_u16((uint16_t)name.size());
    bytes.insert(bytes.end(), name.begin(), name.end());
}
void NbtWriter::begin_compound(std::string_view name) { tag_header(NbtType::Compound, name); }
void NbtWriter::end_compound() { bytes.push_back(0); }
void NbtWriter::write_byte  (std::string_view n, int8_t v)  { tag_header(NbtType::Byte, n);  bytes.push_back((uint8_t)v); }
void NbtWriter::write_short (std::string_view n, int16_t v) { tag_header(NbtType::Short, n); put_u16((uint16_t)v); }
void NbtWriter::write_int   (std::string_view n, int32_t v) { tag_header(NbtType::Int, n);   put_u32((uint32_t)v); }
void NbtWriter::write_long  (std::string_view n, int64_t v) { tag_header(NbtType::Long, n);  put_u64((uint64_t)v); }
void NbtWriter::write_float (std::string_view n, float v)   { tag_header(NbtType::Float, n); uint32_t b; std::memcpy(&b, &v, 4); put_u32(b); }
void NbtWriter::write_double(std::string_view n, double v)  { tag_header(NbtType::Double, n); uint64_t b; std::memcpy(&b, &v, 8); put_u64(b); }
void NbtWriter::write_string(std::string_view n, std::string_view v) {
    tag_header(NbtType::String, n); put_u16((uint16_t)v.size()); bytes.insert(bytes.end(), v.begin(), v.end());
}
void NbtWriter::write_string_item(std::string_view v) { put_u16((uint16_t)v.size()); bytes.insert(bytes.end(), v.begin(), v.end()); }
void NbtWriter::write_byte_array(std::string_view n, const std::vector<int8_t>& v) {
    tag_header(NbtType::ByteArray, n); put_u32((uint32_t)v.size());
    for (int8_t b : v) bytes.push_back((uint8_t)b);
}
void NbtWriter::write_int_array(std::string_view n, const std::vector<int32_t>& v) {
    tag_header(NbtType::IntArray, n); put_u32((uint32_t)v.size());
    for (int32_t x : v) put_u32((uint32_t)x);
}
void NbtWriter::write_long_array(std::string_view n, const std::vector<int64_t>& v) {
    tag_header(NbtType::LongArray, n); put_u32((uint32_t)v.size());
    for (int64_t x : v) put_u64((uint64_t)x);
}
void NbtWriter::begin_list(std::string_view n, NbtType elem, int32_t count) {
    tag_header(NbtType::List, n); bytes.push_back((uint8_t)elem); put_u32((uint32_t)count);
}

}
