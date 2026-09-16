#include "mc_inflate.h"
#include <cstdlib>
#include <cstring>

#include "../../src/Util/stb_image.h"

namespace mc {

bool inflate_buffer(const uint8_t* data, size_t size, Compression kind,
                    std::vector<uint8_t>& out, size_t expected) {
    out.clear();
    if (!data) return false;

    if (kind == Compression::None) {
        out.assign(data, data + size);
        return true;
    }

    const uint8_t* p = data;
    size_t n = size;
    int parseHeader = 1;
    if (kind == Compression::Gzip) {
        if (n < 18 || p[0] != 0x1F || p[1] != 0x8B || p[2] != 8) return false;
        const uint8_t flg = p[3];
        size_t off = 10;
        if (flg & 0x04) { if (off + 2 > n) return false; const size_t xlen = p[off] | (p[off + 1] << 8); off += 2 + xlen; }
        if (flg & 0x08) { while (off < n && p[off]) ++off; ++off; }
        if (flg & 0x10) { while (off < n && p[off]) ++off; ++off; }
        if (flg & 0x02) off += 2;
        if (off + 8 > n) return false;
        const uint32_t isize = (uint32_t)p[n - 4] | ((uint32_t)p[n - 3] << 8) | ((uint32_t)p[n - 2] << 16) | ((uint32_t)p[n - 1] << 24);
        if (expected == 0) expected = isize;
        p += off; n -= off + 8;
        parseHeader = 0;
    } else if (kind == Compression::RawDeflate) {
        parseHeader = 0;
    }

    if (n > 0x7FFFFFFF) return false;
    int outLen = 0;
    const size_t guessSz = expected ? expected + 16 : (n * 4 + 65536);
    const int guess = (int)(guessSz > 0x7FFFFFFF ? 0x7FFFFFFF : guessSz);
    char* buf = stbi_zlib_decode_malloc_guesssize_headerflag((const char*)p, (int)n, guess, &outLen, parseHeader);
    if (!buf) return false;
    out.assign((const uint8_t*)buf, (const uint8_t*)buf + outLen);
    std::free(buf);
    return true;
}

}
