#include "ReuseTextureGen.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

namespace
{
    inline int WrapMod(int a, int m)
    {
        const int r = a % m;
        return (r < 0) ? r + m : r;
    }

    // Paper Eq. 3: n_sigma = floor((sigma/sqrt(2))^2 + 1.46/sigma + 1.76/sigma^2
    //                              + 0.656/sigma^3 + 0.5)
    int ComputeSigmaIterations(float sigma)
    {
        const float s = std::max(sigma, 0.8f);
        const float s2 = s * s;
        const float s3 = s2 * s;
        const float v = 0.5f * s2
                      + 1.46f  / s
                      + 1.76f  / s2
                      + 0.656f / s3
                      + 0.5f;
        return std::max(1, static_cast<int>(std::floor(v)));
    }
}

void GenerateReuseTexture(int size, float sigma, uint32_t seed,
                          std::vector<int16_t>& outRG)
{
    assert((size & 1) == 0 && "reuse texture size must be even");
    assert(size > 0);

    const int N = size * size;

    // Link indices: adjacent horizontal pairs share an index in the
    // initial configuration (pixels 2k and 2k+1 both hold link k).
    std::vector<uint32_t> link(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        link[i] = static_cast<uint32_t>(i / 2);

    std::mt19937 rng(seed);
    const int iters = ComputeSigmaIterations(sigma);

    // Tiled 2×2 shuffles with alternating diagonal offset of 1.
    for (int it = 0; it < iters; ++it)
    {
        const int off = (it & 1) ? 1 : 0;

        for (int by = 0; by < size; by += 2)
        {
            for (int bx = 0; bx < size; bx += 2)
            {
                const int x0 = WrapMod(bx + off,     size);
                const int x1 = WrapMod(bx + off + 1, size);
                const int y0 = WrapMod(by + off,     size);
                const int y1 = WrapMod(by + off + 1, size);

                const int p0 = y0 * size + x0;
                const int p1 = y0 * size + x1;
                const int p2 = y1 * size + x0;
                const int p3 = y1 * size + x1;

                uint32_t tmp[4] = { link[p0], link[p1], link[p2], link[p3] };

                // Fisher–Yates on 4 entries.
                for (int i = 3; i > 0; --i)
                {
                    std::uniform_int_distribution<int> d(0, i);
                    const int j = d(rng);
                    std::swap(tmp[i], tmp[j]);
                }

                link[p0] = tmp[0]; link[p1] = tmp[1];
                link[p2] = tmp[2]; link[p3] = tmp[3];
            }
        }
    }

    // Each link index now appears in exactly two positions. Build a
    // (link -> {firstPos, secondPos}) table.
    const int halfLinks = N / 2;
    std::vector<int32_t> firstPos(static_cast<size_t>(halfLinks), -1);
    std::vector<int32_t> secondPos(static_cast<size_t>(halfLinks), -1);

    for (int i = 0; i < N; ++i)
    {
        const uint32_t l = link[i];
        if (firstPos[l] < 0) firstPos[l]  = i;
        else                 secondPos[l] = i;
    }

    // Emit tileable deltas.
    outRG.resize(static_cast<size_t>(N) * 2u);
    const int halfW = size / 2;

    for (int i = 0; i < N; ++i)
    {
        const uint32_t l = link[i];
        const int partner = (firstPos[l] == i) ? secondPos[l] : firstPos[l];

        const int mx = i       % size;
        const int my = i       / size;
        const int px = partner % size;
        const int py = partner / size;

        int dx = px - mx;
        int dy = py - my;

        // Canonicalize to the short-way-around delta so the texture
        // tiles under wrap. dx, dy end up in [-halfW, halfW]; both
        // +halfW and -halfW are valid and consistent for the pair.
        if (dx >  halfW) dx -= size;
        if (dx < -halfW) dx += size;
        if (dy >  halfW) dy -= size;
        if (dy < -halfW) dy += size;

        outRG[static_cast<size_t>(i) * 2 + 0] = static_cast<int16_t>(dx);
        outRG[static_cast<size_t>(i) * 2 + 1] = static_cast<int16_t>(dy);
    }
}

bool ValidateReuseTexture(int size, const std::vector<int16_t>& rg,
                          int* outFirstBadTexel)
{
    if (size <= 0 || (size & 1)) return false;
    if (rg.size() != static_cast<size_t>(size) * size * 2) return false;

    for (int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            const int i   = y * size + x;
            const int dx  = rg[static_cast<size_t>(i) * 2 + 0];
            const int dy  = rg[static_cast<size_t>(i) * 2 + 1];

            const int px  = WrapMod(x + dx, size);
            const int py  = WrapMod(y + dy, size);
            const int pi  = py * size + px;

            const int pdx = rg[static_cast<size_t>(pi) * 2 + 0];
            const int pdy = rg[static_cast<size_t>(pi) * 2 + 1];

            // Applying partner's delta should land back at (x, y)
            // under wrap. This formulation is robust to the ±halfW
            // canonicalization ambiguity.
            const int bx = WrapMod(px + pdx, size);
            const int by = WrapMod(py + pdy, size);

            if (bx != x || by != y)
            {
                if (outFirstBadTexel) *outFirstBadTexel = i;
                return false;
            }
        }
    }
    return true;
}
