// =====================================================================
//  Pass_cloudnoise_bake_v8.hlsl  —  One-shot cloud noise texture bake
// =====================================================================
//
//  Run ONCE during renderer init. Fills a 256³ RGBA8 3D texture with
//  the four noise channels the runtime cloud shader needs. After this
//  pass the cloud shader replaces its analytical Perlin / Worley / FBM
//  evaluations with a single Texture3D.SampleLevel() per density tap —
//  ~50–200× fewer ALU ops per sample, which is the dominant cost in
//  the cloud march per NSight (compute-bound, not bandwidth-bound).
//
//  Channel layout (all bake-side periods are integer for clean tiling):
//    R: Perlin-Worley FBM      (Schneider HZD base shape, 32 cells / tile)
//    G: inv-Worley FBM 3-oct   (cauliflower lobe + HF erosion,  8 cells / tile)
//    B: Value noise            (smooth fields: domain warp + top, 48 / tile)
//    A: Worley single octave   (billow seam carve,             16 cells / tile)
//
//  PERIOD vs CRISPNESS (2026-06-13 shape overhaul): a channel's finest
//  resolvable detail is BAKE_RES/period texels per noise cell. The detail
//  channels (G cauliflower, A billow) were the blocky bottleneck — their
//  top FBM octaves landed at only ~4 texels/cell and got low-pass-filtered
//  into "potato" blobs. Halving G's period (16->8) and A's (32->16) doubles
//  texels/cell on every octave (G-oct2 4->8 tx, A 8->16 tx) so the cells
//  keep crisp cauliflower edges. The cost is faster WRAP tiling, but only
//  on HIGH-frequency channels (G tiles ~3 km, A ~17 km in world space) where
//  the domain warp + the 100 km-tiling R base + wind drift hide it. The
//  base R and smooth B keep their periods — tiling a 3 km macro blob or a
//  240 km top-variation field at higher frequency WOULD be visible.
//  Cell size is set by the runtime SAMPLE frequency, NOT the period, so
//  these changes are crispness-only — feature scales are unchanged.
//
//  The bake-side noise functions are tiled by wrapping the hash input
//  modulo the channel's `period`, so sampling the texture with WRAP
//  addressing produces a seamless infinite-domain noise field.
//
//  Runtime sample mapping (see Clouds_v8.hlsli):
//    uvw = world_pos / period  →  texture wraps every `period` units
//                                 in the original analytical input space.
//
//  Dispatch: (256/8)³ = 32³ = 32768 thread groups, one voxel per thread.
//  One-time cost is a few ms on the target hardware; runs synchronously
//  in the init command list before the path-tracing pipeline starts.

RWTexture3D<float4> g_noise : register(u0);

#define BAKE_RES        256
#define R_PERIOD        32      // Perlin-Worley grid (base shape)
#define G_PERIOD        8       // inv-Worley FBM grid (cauliflower + erosion)
#define B_PERIOD        48      // Value-noise grid (warp + top, smooth)
#define A_PERIOD        16      // Single-octave Worley grid (billow seams)

//----------------------------------------------------------------------
//  TILEABLE HASH PRIMITIVES
//----------------------------------------------------------------------
//  Mirrors CloudHash3D / CloudHashFloat / CloudHashVec3 from
//  Clouds_v8.hlsli but takes a `period` parameter that wraps the cell
//  coordinate before hashing. The wrap is what makes the noise tile.

uint BakeHash3D(int3 p, int period)
{
    // ((p % period) + period) % period — wraps negatives correctly.
    int3 m = (p % period + period) % period;
    uint h = uint(m.x) * 73856093u
           ^ uint(m.y) * 19349663u
           ^ uint(m.z) * 83492791u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float BakeHashFloat(int3 p, int period)
{
    return float(BakeHash3D(p, period)) * (1.0f / 4294967296.0f);
}

float3 BakeHashVec3(int3 p, int period)
{
    uint h = BakeHash3D(p, period);
    return float3((h        & 1023u),
                  ((h >> 10) & 1023u),
                  ((h >> 20) & 1023u)) * (1.0f / 1023.0f);
}

//----------------------------------------------------------------------
//  VALUE NOISE  (matches CloudValueNoise — quintic interpolation)
//----------------------------------------------------------------------

float BakeValueNoise(float3 p, int period)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float c000 = BakeHashFloat(i + int3(0, 0, 0), period);
    float c100 = BakeHashFloat(i + int3(1, 0, 0), period);
    float c010 = BakeHashFloat(i + int3(0, 1, 0), period);
    float c110 = BakeHashFloat(i + int3(1, 1, 0), period);
    float c001 = BakeHashFloat(i + int3(0, 0, 1), period);
    float c101 = BakeHashFloat(i + int3(1, 0, 1), period);
    float c011 = BakeHashFloat(i + int3(0, 1, 1), period);
    float c111 = BakeHashFloat(i + int3(1, 1, 1), period);

    float x00 = lerp(c000, c100, u.x);
    float x10 = lerp(c010, c110, u.x);
    float x01 = lerp(c001, c101, u.x);
    float x11 = lerp(c011, c111, u.x);
    float y0  = lerp(x00,  x10,  u.y);
    float y1  = lerp(x01,  x11,  u.y);
    return lerp(y0, y1, u.z);
}

//----------------------------------------------------------------------
//  WORLEY  (single octave, 27-neighbour search, matches CloudWorley)
//----------------------------------------------------------------------

float BakeWorley(float3 p, int period)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float  minD2 = 1.0e10f;

    [unroll] for (int x = -1; x <= 1; ++x)
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int z = -1; z <= 1; ++z)
    {
        int3   cell    = i + int3(x, y, z);
        float3 feature = float3(x, y, z) + BakeHashVec3(cell, period) - f;
        float  d2      = dot(feature, feature);
        minD2          = min(minD2, d2);
    }
    return saturate(sqrt(minD2) * 1.15f);
}

//----------------------------------------------------------------------
//  WORLEY FBM  (3-octave, matches the analytical CloudWorleyFBM at q=0)
//----------------------------------------------------------------------
//  Higher octaves run at 2× and 4× frequency on correspondingly larger
//  period grids so the whole FBM still tiles with the texture. Three
//  octaves are required — earlier two-octave versions produced visibly
//  blobby/potato-shaped cell boundaries because single-scale Worley
//  cells are too smooth on their own. The 3rd octave provides the
//  cauliflower edge detail that distinguishes cumulus from smooth ovals.
//  Weights (0.55/0.30/0.15) keep the result amplitude-normalised in [0,1].

float BakeWorleyFBM(float3 p, int basePeriod)
{
    float w0 = 1.0f - BakeWorley(p,                                          basePeriod);
    float w1 = 1.0f - BakeWorley(p * 2.0f + float3(13.31f, -7.13f, 19.77f),  basePeriod * 2);
    float w2 = 1.0f - BakeWorley(p * 4.0f + float3(-3.47f, 21.97f,  5.13f),  basePeriod * 4);
    return saturate(w0 * 0.55f + w1 * 0.30f + w2 * 0.15f);
}

//----------------------------------------------------------------------
//  PERLIN  (gradient noise, matches CloudPerlin)
//----------------------------------------------------------------------

float3 BakePerlinGrad(int3 p, int period)
{
    uint h = BakeHash3D(p, period) & 15u;
    float3 g;
    g.x = (h <  8u) ? 1.0f : -1.0f;
    g.y = (h <  4u || h == 12u || h == 14u) ? 1.0f : -1.0f;
    g.z = (h <  2u || h == 12u || h == 13u) ?  0.0f : 1.0f;
    return g;
}

float BakePerlin(float3 p, int period)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float3 g000 = BakePerlinGrad(i + int3(0, 0, 0), period);
    float3 g100 = BakePerlinGrad(i + int3(1, 0, 0), period);
    float3 g010 = BakePerlinGrad(i + int3(0, 1, 0), period);
    float3 g110 = BakePerlinGrad(i + int3(1, 1, 0), period);
    float3 g001 = BakePerlinGrad(i + int3(0, 0, 1), period);
    float3 g101 = BakePerlinGrad(i + int3(1, 0, 1), period);
    float3 g011 = BakePerlinGrad(i + int3(0, 1, 1), period);
    float3 g111 = BakePerlinGrad(i + int3(1, 1, 1), period);

    float c000 = dot(g000, f - float3(0, 0, 0));
    float c100 = dot(g100, f - float3(1, 0, 0));
    float c010 = dot(g010, f - float3(0, 1, 0));
    float c110 = dot(g110, f - float3(1, 1, 0));
    float c001 = dot(g001, f - float3(0, 0, 1));
    float c101 = dot(g101, f - float3(1, 0, 1));
    float c011 = dot(g011, f - float3(0, 1, 1));
    float c111 = dot(g111, f - float3(1, 1, 1));

    float x00 = lerp(c000, c100, u.x);
    float x10 = lerp(c010, c110, u.x);
    float x01 = lerp(c001, c101, u.x);
    float x11 = lerp(c011, c111, u.x);
    float y0  = lerp(x00,  x10,  u.y);
    float y1  = lerp(x01,  x11,  u.y);
    return lerp(y0, y1, u.z);
}

// 3-octave Perlin FBM. The third octave restores the high-freq jitter
// the analytical version had — without it, the Perlin lift in
// BakePerlinWorley over-smooths the result and contributes to the
// blobby low-freq look. Weights are 4:2:1 (analytical convention).
float BakePerlinFBM(float3 p, int period)
{
    float v = BakePerlin(p,                                       period)     * 0.50f
            + BakePerlin(p * 2.0f + float3(5.7f, -2.3f, 9.1f),    period * 2) * 0.25f
            + BakePerlin(p * 4.0f + float3(-1.3f, 11.7f, -4.2f),  period * 4) * 0.125f;
    return saturate(v * (1.0f / 1.75f) + 0.5f);
}

//----------------------------------------------------------------------
//  PERLIN-WORLEY  (Schneider 2015 — Perlin lift on inverted Worley FBM)
//----------------------------------------------------------------------
//  Identical to the analytical CloudPerlinWorley: the worleyInv input
//  is the FULL 3-octave Worley FBM, not a single-octave Worley. The
//  earlier bake used single-octave Worley here, which collapsed cell
//  boundaries to smooth circular ovals (the "blobs / potatoes" look).
//  Feeding the 3-octave FBM restores the high-freq jitter at cell
//  edges, which is what gives cumulus their cauliflower silhouette.

float BakePerlinWorley(float3 p, int period)
{
    float perlin    = BakePerlinFBM(p, period);
    float worleyInv = BakeWorleyFBM(p, period);   // 3-octave; cauliflower edges
    return saturate((perlin - (1.0f - worleyInv)) / max(worleyInv, 1e-3f));
}

//----------------------------------------------------------------------
//  ENTRY POINT
//----------------------------------------------------------------------
//  One thread per voxel. uvw is the voxel-centred normalised position
//  in [0,1)³. Each channel scales uvw by its per-channel period to span
//  that many tileable noise cells across the texture.

[numthreads(8, 8, 8)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (any(dtid >= uint3(BAKE_RES, BAKE_RES, BAKE_RES))) return;

    float3 uvw = (float3(dtid) + 0.5f) / float(BAKE_RES);

    float r = BakePerlinWorley(uvw * float(R_PERIOD), R_PERIOD);
    float g = BakeWorleyFBM   (uvw * float(G_PERIOD), G_PERIOD);
    float b = BakeValueNoise  (uvw * float(B_PERIOD), B_PERIOD);
    // A channel keeps Worley in RAW (non-inverted) form because the
    // CloudWorley() call site in Clouds_v8.hlsli inverts inline:
    //   modulation = 2*((1-midW)-0.5) = 1 - 2*midW
    // Storing the raw form preserves the original sign of `modulation`
    // so cumulus mid-frequency bumps remain additive at Worley feature
    // points (midW≈0) and subtractive at cell interiors (midW≈1).
    float a = BakeWorley(uvw * float(A_PERIOD), A_PERIOD);

    g_noise[dtid] = float4(saturate(r), saturate(g), saturate(b), saturate(a));
}
