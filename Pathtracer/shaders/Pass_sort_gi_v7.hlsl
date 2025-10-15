#include "Includes_v7.hlsli"

//--------------------------------------------------------------------------------------
// Tunables
//--------------------------------------------------------------------------------------
#define TILE_W 16
#define TILE_H 16
#define TILE_N (TILE_W * TILE_H)

#define NORM_BITS_PER_AXIS 3     // 8x8 octa bins -> 64 normal buckets
#define OBJ_HASH_BITS      2     // coarse object split (collisions ok)
#define BUCKET_BITS        8     // total bucket bits (<= 8 keeps 32-bit key simple)
#define PROB_BITS          24    // remaining bits for -p_hat ordering

// Output: one slice (z=12) carries the per-tile sorted list (contiguous sublists by bucket)
//   gScratchPing[uint3(x,y,12)].x = asfloat(pixelIdx)
//   gScratchPing[uint3(x,y,12)].y = asfloat(bucketId)
//   gScratchPing[uint3(x,y,12)].z = pix_p_hat

//--------------------------------------------------------------------------------------
// Hash & helpers
//--------------------------------------------------------------------------------------
uint Hash32(uint x) {
    x ^= 61u; x ^= x >> 16; x *= 9u; x ^= x >> 4; x *= 0x27d4eb2d; x ^= x >> 15;
    return x;
}

// Octahedral normal binning (quantized)
uint OctaBin(float3 n) {
    n = normalize(n);
    float3 a = abs(n);
    float invL1 = 1.0 / max(a.x + a.y + a.z, 1e-6);
    float2 p = n.xy * invL1;
    if (n.z < 0.0) {
        float2 s = float2((p.x >= 0.0) ? 1.0 : -1.0, (p.y >= 0.0) ? 1.0 : -1.0);
        p = (1.0 - float2(abs(p.y), abs(p.x))) * s;
    }
    const float qSteps = float(1 << NORM_BITS_PER_AXIS); // e.g., 8
    float2 uv = (p * 0.5 + 0.5) * qSteps;
    uint ux = (uint) clamp(floor(uv.x), 0.0, qSteps - 1.0);
    uint uy = (uint) clamp(floor(uv.y), 0.0, qSteps - 1.0);
    return (ux << NORM_BITS_PER_AXIS) | uy; // [0..(8*8-1)]
}

uint MakeBucketId(float3 n, uint objId) {
    uint nbin   = OctaBin(n);                               // 6 bits (with NORM_BITS_PER_AXIS=3)
    uint ohash  = Hash32(objId) & ((1u << OBJ_HASH_BITS) - 1u);  // 2 bits
    uint merged = (nbin << OBJ_HASH_BITS) | ohash;          // up to 256
    if ((2 * NORM_BITS_PER_AXIS + OBJ_HASH_BITS) > BUCKET_BITS)
        merged = Hash32(merged) & ((1u << BUCKET_BITS) - 1u);
    return merged & ((1u << BUCKET_BITS) - 1u);
}

// Quantize p in [0,1] to PROB_BITS and invert so larger p sorts first
uint ProbDesc(float p) {
    float q = saturate(p);
    uint  u = (uint)round(q * ((1u << PROB_BITS) - 1u));
    return ((1u << PROB_BITS) - 1u) - u; // invert for descending probability
}

// Composite key: [bucket (BUCKET_BITS) | prob (PROB_BITS)]
uint MakeKey(uint bucket, float p) {
    return (bucket << PROB_BITS) | ProbDesc(p); // ascending sort on this key
}

uint FlatLocal(uint2 l) { return l.y * TILE_W + l.x; }

//--------------------------------------------------------------------------------------
// Shared memory for 256-way bitonic sort
//--------------------------------------------------------------------------------------
groupshared uint sKey[TILE_N];
groupshared uint sPix[TILE_N];

inline void CompareSwap(uint i, uint j, bool dirAscending) {
    uint ki = sKey[i], kj = sKey[j];
    bool swap = (ki > kj);
    if (!dirAscending) swap = !swap;
    if (swap) {
        sKey[i] = kj; sKey[j] = ki;
        uint pi = sPix[i]; sPix[i] = sPix[j]; sPix[j] = pi;
    }
}

//─────────────────────────────────────────────────────────────────────────────
//  SORT GI (16x16 tile)
//─────────────────────────────────────────────────────────────────────────────
[numthreads(TILE_W, TILE_H, 1)]
void main(uint3 tid  : SV_DispatchThreadID,
          uint3 ltid : SV_GroupThreadID,
          uint3 gid  : SV_GroupID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx = MapPixelID(dims, tid.xy);

    // Load inputs via your helpers
    float  pix_p_hat = load_F_gi(g_Reservoirs_current_gi, pixelIdx);
    float3 normal    = load_n1(   g_sample_current,        pixelIdx);
    uint   objID     = load_objID(g_sample_current,        pixelIdx);

    // Partition & composite key
    uint bucket = MakeBucketId(normal, objID);
    uint key    = MakeKey(bucket, pix_p_hat);

    // Stage to shared
    const uint local = FlatLocal(ltid.xy);
    sKey[local] = key;
    sPix[local] = pixelIdx;
    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort on 256 elements (ascending on composite key)
    [unroll] for (uint k = 2; k <= TILE_N; k <<= 1) {
        [unroll] for (uint j = k >> 1; j > 0; j >>= 1) {
            uint idx = local;
            uint ixj = idx ^ j;
            if (ixj > idx) {
                bool dirAscending = ((idx & k) == 0);
                CompareSwap(idx, ixj, dirAscending);
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write sorted, partitioned list to z=12 (one tile occupies its own 16x16 block)
    uint2 tileBase = gid.xy * uint2(TILE_W, TILE_H);
    uint2 outXY    = tileBase + ltid.xy;

    uint outBucket = (sKey[local] >> PROB_BITS);
    uint outPix    = sPix[local];

    // Store as float3: pixelId, bucketId, p_hat
    gScratchPing[uint3(outXY, 12)] = float4(asfloat(outPix), asfloat(outBucket), pix_p_hat, 0);
}