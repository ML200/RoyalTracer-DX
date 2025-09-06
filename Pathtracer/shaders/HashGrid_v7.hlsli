// ───────────────────────────── Tunables (range & budget) ─────────────────────────────
#define GRID_MAX_RANGE                         10000.0f
#define GRID_MAX_CELL_BUDGET                    2000
#define GRID_LMAX_CAP                              64

// Distance→level mapping controls
#define GRID_DISTANCE_SLOPE                        0.7f   // <1 flatter, >1 steeper
#define GRID_DISTANCE_PIVOT_MULT                  16.0f   // pivot at r = PIVOT_MULT * s0 (pick > 1 so l_min won't dominate)
#define GRID_DISTANCE_BIAS_LEVELS                  10.0f   // additive bias in "levels"

// Minimum selectable cell size (layout s0/L is unchanged; just enforces selection floor)
#define GRID_MIN_CELL_SIZE                          0.5f

#define GRID_LEVEL_EPS                            1e-6f
#define GRID_ALIGN_TO_CAMERA                         0
#define GRID_ANCHOR_TO_CAMERA                        1
#define GRID_QUANTIZE_ANCHOR                         1
#define GRID_DEBUG_TINT_LEVEL                        0
#define GRID_CULL_OUTSIDE_RANGE                      1

// ───────────────────────────── Helpers ─────────────────────────────
inline float3 GetCameraPosWS() { return mul(viewI, float4(0,0,0,1)).xyz; }
inline float  ChebRadius3(float3 v){ v = abs(v); return max(v.x, max(v.y, v.z)); }

inline float3 GridBasisSpace(float3 pWS)
{
#if GRID_ALIGN_TO_CAMERA
    return mul(view, float4(pWS,1)).xyz;     // camera/view aligned
#else
    return pWS;                               // world-aligned
#endif
}

// ───────────────────────────── Budget math ─────────────────────────────
// 3D Chebyshev rings: L levels → cells = 64 + 56*(L-1), L ≥ 1
inline int  GridLevelsFromBudget(uint budget)
{
    uint b = (budget < 64u) ? 64u : budget;
    uint extraRings = (b - 64u) / 56u;
    int  L = (int)(1u + extraRings);
    return min(L, GRID_LMAX_CAP);
}

// Derive (s0, L) purely from range + budget (NO min-size enforcement here)
inline void GridDerivedParams(out float s0, out int L)
{
    L  = GridLevelsFromBudget(GRID_MAX_CELL_BUDGET);
    s0 = GRID_MAX_RANGE * exp2(- (float)L);   // s0 = R / 2^L
}

// ───────────────────────────── Level selection by distance (slope + pivot + bias)
inline int SelectLevelCheb(float rCheb, float s0, int Lmax)
{
    const float EPS = 1e-30f;

    // Raw level: grows by 1 per doubling of distance (clamped to >= 0 when r < s0)
    float t_raw = log2( max(rCheb / max(s0, EPS), 1.0f) );

    // Pivot in "levels" (choose pivot > 1 so clamp-to-l_min doesn't dominate)
    float t0 = log2( max(GRID_DISTANCE_PIVOT_MULT, 1.000001f) );

    // Shape around the pivot, then bias
    float t_shaped = (t_raw - t0) * GRID_DISTANCE_SLOPE + t0 + GRID_DISTANCE_BIAS_LEVELS;

    int l = (int)floor(t_shaped + GRID_LEVEL_EPS);
    return clamp(l, 0, Lmax - 1);
}

inline float CellSizeAt(int l, float s0) { return s0 * exp2((float)l); }

// ───────────────────────────── FIXED: Minimum allowed level so that s(l) >= GRID_MIN_CELL_SIZE
inline int MinLevelFromCellSize(float s0)
{
    // l_min = ceil(log2(sMin / s0)) = ceil(log2(sMin) - log2(s0)) in log-domain for robustness
    const float EPS = 1e-30f;
    float sMin  = max(GRID_MIN_CELL_SIZE, EPS);
    float lmind = ceil( log2(sMin) - log2(max(s0, EPS)) );
    int   lmin  = (int)lmind;
    return max(lmin, 0);
}

// ───────────────────────────── Hashing/Color ─────────────────────────────
inline uint Hash32(uint x){ x^=x>>16; x*=0x7feb352du; x^=x>>15; x*=0x846ca68bu; x^=x>>16; return x; }

inline uint HashIndex(int level, int ix, int iy, int iz)
{
    uint h=2166136261u;
    h ^= (uint)level;      h *= 16777619u;
    h ^= (uint)ix*73856093u;
    h ^= (uint)iy*19349663u;
    h ^= (uint)iz*83492791u;
    return Hash32(h);
}

inline float3 HashColor(uint h){
    uint h1=Hash32(h^0x9e3779b9u), h2=Hash32(h1^0x85ebca6bu), h3=Hash32(h2^0xc2b2ae35u);
    const float k=1.0f/4294967295.0f; return float3(h1,h2,h3)*k;
}

inline float3 TintByLevel(int l)
{
#if GRID_DEBUG_TINT_LEVEL
    uint th = Hash32((uint)l * 2654435761u);
    float3 c = HashColor(th);
    return lerp(float3(1,1,1), c, 0.35f);
#else
    return 1.0.xxx;
#endif
}

// ───────────────────────────── Visualization ─────────────────────────────
inline float3 VisualizeRectGridAt(float3 pWS)
{
    float s0; int L; GridDerivedParams(s0, L);

    // Early-out if min-size exceeds the coarsest available level
    int l_min_raw = MinLevelFromCellSize(s0);
    if (l_min_raw > L - 1) return 0.0.xxx;

    const float3 camWS = GetCameraPosWS();
    float3 toCam = pWS - camWS;
    float  rCheb = ChebRadius3(toCam);

#if GRID_CULL_OUTSIDE_RANGE
    if (rCheb >= GRID_MAX_RANGE) return 0.0.xxx;
#endif

    // Distance-based level
    int l = SelectLevelCheb(rCheb, s0, L);

    // Clamp l_min and final l to [0, L-1]
    int l_min = min(l_min_raw, L - 1);
    l = clamp(max(l, l_min), 0, L - 1);

    float s = CellSizeAt(l, s0);

    // Basis coords
    float3 pGS   = GridBasisSpace(pWS);
    float3 camGS = GridBasisSpace(camWS);

    // Anchor (quantized per chosen level)
    float3 baseGS;
#if GRID_ANCHOR_TO_CAMERA
  #if GRID_QUANTIZE_ANCHOR
    baseGS = floor(camGS / s) * s;
  #else
    baseGS = camGS;
  #endif
#else
    baseGS = 0.0.xxx;
#endif

    // Integer cell index
    float3 q = (pGS - baseGS) / s;
    int ix = (int)floor(q.x);
    int iy = (int)floor(q.y);
    int iz = (int)floor(q.z);

    // Hash color
    float3 col = HashColor(HashIndex(l, ix, iy, iz)) * TintByLevel(l);
    return col;
}

inline float3 VisualizeRectGridAt_NormalAware(float3 pWS, float3 nWS)
{
    float len2 = dot(nWS, nWS);
    float3 nn  = (len2 > 1e-12f) ? nWS * rsqrt(len2) : 0.0.xxx;

    float s0; int L; GridDerivedParams(s0, L);

    // Early-out if min-size exceeds the coarsest available level
    int l_min_raw = MinLevelFromCellSize(s0);
    if (l_min_raw > L - 1) return 0.0.xxx;

    const float3 camWS = GetCameraPosWS();
    float3 toCam = pWS - camWS;
    float  rCheb = ChebRadius3(toCam);

#if GRID_CULL_OUTSIDE_RANGE
    if (rCheb >= GRID_MAX_RANGE) return 0.0.xxx;
#endif

    int l = SelectLevelCheb(rCheb, s0, L);

    // Clamp l_min and final l to [0, L-1]
    int l_min = min(l_min_raw, L - 1);
    l = clamp(max(l, l_min), 0, L - 1);

    float s = CellSizeAt(l, s0);

    float3 pEff  = pWS + nn * (0.5f * s); // push half a cell
    float3 pGS   = GridBasisSpace(pEff);
    float3 camGS = GridBasisSpace(camWS);

#if GRID_ANCHOR_TO_CAMERA
  #if GRID_QUANTIZE_ANCHOR
    float3 baseGS = floor(camGS / s) * s;
  #else
    float3 baseGS = camGS;
  #endif
#else
    float3 baseGS = 0.0.xxx;
#endif

    float3 q = (pGS - baseGS) / s;
    int ix = (int)floor(q.x);
    int iy = (int)floor(q.y);
    int iz = (int)floor(q.z);

    return HashColor(HashIndex(l, ix, iy, iz)) * TintByLevel(l);
}
