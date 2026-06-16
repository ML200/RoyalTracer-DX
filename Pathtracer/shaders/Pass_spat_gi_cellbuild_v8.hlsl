#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//CELL BUILD (Hedstrom et al. 2026, Algorithm 1 - tile-local form)
//====================================
//Builds the paper's screen-space cells WITHOUT a global Boissé hash grid + GPU
//sort. The paper's cells live inside an 8x8 tile, so one 8x8 threadgroup == one
//tile and reduces its cells in groupshared memory. We write a per-pixel cell
//record to the (unused in cell mode) path-state scratch buffer; the resample
//pass reads these records cross-tile to run the §5.1 cell search.
//
//Cell membership is keyed on instID (same surface) plus a SOFT normal CONE that
//the search/gather test against the QUERY pixel's normal - NOT a hard quantized
//normal bucket. A hard bucket creates a fixed boundary that, where normals vary
//fast (curved surfaces, edges), fragments cells, fails the cell search and falls
//back to the home tile -> a visible 8x8 grid on edges. A query-relative cone has
//no fixed boundary, so there is nothing for a grid to align to.
//
//Per-pixel record, 20 B at g_pathStateBuffer[px*20]:
//  +0  instID (uint)       same-surface key ; 0xFFFFFFFF = invalid (emitter/empty)
//  +4  normalPk (uint)     PackNormal(n) ; cone-tested at search/gather
//  +8  confSum (float)     sum of c_i over same-instID pixels in this 8x8 tile
//  +12 c_i (float)         this pixel's confidence min(cap, M)
//  +16 w_i (float)         selection weight c_i*pHat(F)*W (Eq.17) ; 0 if no sample
static const uint CELL_REC = 20u;
uint cell_addr(uint px) { return px * CELL_REC; }

static const uint TILE_PX = 64u;   // 8x8 threadgroup == paper's 8x8 tile
groupshared uint  gsInst[TILE_PX];
groupshared float gsC   [TILE_PX];

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint gi : SV_GroupIndex)
{
    if (!CELL_SPATIAL_MODE) return;

    const float2 dims = float2(IMG_W, IMG_H);
    const bool   inB  = (tid.x < IMG_W) && (tid.y < IMG_H);

    uint  inst = 0xFFFFFFFFu, nrmPk = 0u;
    float c_i = 0.0f, w_i = 0.0f;
    uint  px  = 0u;

    if (inB)
    {
        gDispatchIdx = tid;
        px = MapPixelID(dims, tid.xy);
        if (!load_isEmitter(g_sample_current, px))
        {
            const uint M = load_M(g_Reservoirs_current, px);
            if (M > 0u)
            {
                inst = load_instID(g_sample_current, px);
                const float3 n = load_n1_s_with_instID(g_sample_current, px, inst);
                nrmPk = PackNormal(n);

                const float  W     = load_W(g_Reservoirs_current, px);
                const float3 F     = load_F(g_Reservoirs_current, px);
                const float  pHatX = GetPHat(F) * (W > 0.0f ? 1.0f : 0.0f);
                c_i = min((float)rs_cellMcap, (float)M);
                w_i = c_i * pHatX * W;
            }
        }
    }

    gsInst[gi] = inst;
    gsC[gi]    = c_i;
    GroupMemoryBarrierWithGroupSync();

    if (!inB) return;

    //cell confidence sum over the tile (same surface); cone coherence is applied
    //later by the search/gather, so confSum here is the per-surface tile total.
    float confSum = 0.0f;
    if (inst != 0xFFFFFFFFu)
    {
        [loop]
        for (uint q = 0u; q < TILE_PX; ++q)
            if (gsInst[q] == inst) confSum += gsC[q];
    }

    g_pathStateBuffer.Store4(cell_addr(px), uint4(inst, nrmPk, asuint(confSum), asuint(c_i)));
    g_pathStateBuffer.Store (cell_addr(px) + 16u, asuint(w_i));
}
