cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx


#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);
RWTexture2DArray<float4> gScratchPing : register(u8); // Storage for denoiser

RWByteAddressBuffer g_sample_current : register(u6);
RWByteAddressBuffer g_sample_last : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi : register(u5);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);
StructuredBuffer<float> g_AliasProb  : register(t7);
StructuredBuffer<uint>  g_AliasIdx   : register(t8);

// Needs access to all structured/random buffers
#include "Sample_data.hlsli"
#include "GGX_v7.hlsli"
#include "Lambertian_v7.hlsli"
#include "BSDF_v7.hlsli"

cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float time;
}
// These includes need access to ALL previous buffers
#include "Camera_ray_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

// -----------------------------------------------------------------------------
//  Constants used by every node
// -----------------------------------------------------------------------------
static const uint kGroupSizeX = 16;
static const uint kGroupSizeY = 16;
static const uint kMaxWidth   = 1920;
static const uint kMaxHeight  = 1080;

static const uint kTilesX = (kMaxWidth  + kGroupSizeX - 1) / kGroupSizeX; // 120
static const uint kTilesY = (kMaxHeight + kGroupSizeY - 1) / kGroupSizeY; //  68

// -----------------------------------------------------------------------------
//  Records exchanged between nodes
// -----------------------------------------------------------------------------
struct TileJob { uint2 origin; };                 // upper-left pixel of a tile

// =============================================================================
//  Node C  –  TileWriter  (broadcasting 16×16)
//            Reads random value from gScratchPing and commits to gOutput
// =============================================================================
[Shader("node")]
[NodeLaunch("broadcasting")]
[NumThreads(kGroupSizeX, kGroupSizeY, 1)]
[NodeDispatchGrid(1, 1, 1)]                       // one group per record
void TileWriter( DispatchNodeInputRecord<TileJob> inRec,
                 uint3 ltid : SV_GroupThreadID )
{
    uint2 pix = inRec.Get().origin + ltid.xy;
    if (pix.x >= gImageWidth || pix.y >= gImageHeight) return;

    float4 randVal = gScratchPing[uint3(pix, 0)]; // produced by TileRandom
    gOutput[uint3(pix, 0)] = randVal;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stand-alone SDF helpers (must be at file / namespace scope in HLSL)
// ─────────────────────────────────────────────────────────────────────────────
static float sdCircle(float2 p, float r)
{
    return length(p) - r;
}

static float sdBox(float2 p, float2 b)
{
    float2 d = abs(p) - b;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

static float sceneSDF(float2 p)     // <- replaces the former lambda 'map'
{
    float d1 = sdCircle(p - float2(-0.30,  0.20), 0.25);
    float d2 = sdCircle(p - float2( 0.40, -0.10), 0.20);
    float d3 = sdBox   (p - float2( 0.00,  0.00), float2(0.15, 0.30));
    return min(min(d1, d2), d3);
}

// ============================================================================
//  Heavy, non-divergent TileRandom  – 16×16 broadcasting work-group
// ============================================================================
//
//  • One work-group per 16×16 tile (same NodeDispatchGrid as before)
//  • Queues one TileWriter record per tile (leader thread)
//
[Shader("node")]
[NodeLaunch("broadcasting")]
[NumThreads(16,16,1)]
[NodeDispatchGrid(1, 1, 1)]
void TileRandom( DispatchNodeInputRecord<TileJob> inRec,
                 uint3 ltid : SV_GroupThreadID,
                 [MaxRecords(1)] NodeOutput<TileJob> TileWriter )
{
    uint2 pix = inRec.Get().origin + ltid.xy;
    if (pix.x >= gImageWidth || pix.y >= gImageHeight) return;

    // ── deterministic RNG state (one per lane, will mutate each iter) ────────
    uint state = pix.x + pix.y * gImageWidth;

    // ── 12 accumulators  →  36 scalars  (helps hit ~50 VGPR/GRF per lane) ────
    float3 acc0  = 0, acc1  = 0, acc2  = 0, acc3  = 0;
    float3 acc4  = 0, acc5  = 0, acc6  = 0, acc7  = 0;
    float3 acc8  = 0, acc9  = 0, acc10 = 0, acc11 = 0;

    // ── 128 fixed iterations (uniform) – heavy ALU, no divergence ────────────
    [unroll(128)]
    for (int i = 0; i < 128; ++i)
    {
        // LCG - style RNG
        state  = state * 1664525u + 1013904223u;
        float r = (state & 0x00FFFFFFu) * (1.0 / 16777216.0);   // [0,1)

        // Some transcendental + nonlinear math to burn instructions
        float3 v = float3(r, r*r, sqrt(r + 1e-4));

        // Chain of dependent MADs – maximises live registers
        acc0  = mad(v, acc0  + 0.01, acc1  + 0.02);
        acc1  = mad(v, acc1  + 0.03, acc2  + 0.04);
        acc2  = mad(v, acc2  + 0.05, acc3  + 0.06);
        acc3  = mad(v, acc3  + 0.07, acc4  + 0.08);
        acc4  = mad(v, acc4  + 0.09, acc5  + 0.10);
        acc5  = mad(v, acc5  + 0.11, acc6  + 0.12);
        acc6  = mad(v, acc6  + 0.13, acc7  + 0.14);
        acc7  = mad(v, acc7  + 0.15, acc8  + 0.16);
        acc8  = mad(v, acc8  + 0.17, acc9  + 0.18);
        acc9  = mad(v, acc9  + 0.19, acc10 + 0.20);
        acc10 = mad(v, acc10 + 0.21, acc11 + 0.22);
        acc11 = mad(v, acc11 + 0.23, acc0  + 0.24);   // feedback for fun
    }

    // ── squeeze the mess into one colour – still uniform ─────────────────────
    float3 colour = normalize(
          acc0 + acc1 + acc2 + acc3 + acc4 + acc5
        + acc6 + acc7 + acc8 + acc9 + acc10 + acc11 );

    // Store to scratch UAV   (TileWriter will copy to gOutput)
    gScratchPing[uint3(pix, 0)] = float4(colour, 1);

    // ── Leader thread queues exactly one TileWriter record per tile ──────────
    bool leader = (ltid.x == 0 && ltid.y == 0);
    GroupNodeOutputRecords<TileJob> rec =
        TileWriter.GetGroupNodeOutputRecords(leader ? 1 : 0);

    if (leader)
        rec[0].origin = inRec.Get().origin;

    GroupMemoryBarrierWithGroupSync();
    if (leader) rec.OutputComplete();
}




// =============================================================================
//  Node A – TileProducer  (broadcasting 16×16)
//           • Loads 30 DI reservoirs per iteration
//           • Keeps 30 float4 accumulators live  → ≈ 120 scalar registers
//           • Only 16 iterations  → light ALU; stress is register pressure
//           • ONE TileJob record per tile (leader thread)
//           • Writes a dummy colour to gScratchPing slice 15
// =============================================================================
[Shader("node")]
[NodeLaunch("broadcasting")]
[NumThreads(kGroupSizeX, kGroupSizeY, 1)]
[NodeDispatchGrid(kTilesX, kTilesY, 1)]
void main( uint3 tid  : SV_DispatchThreadID,
           uint3 ltid : SV_GroupThreadID,
           [MaxRecords(1)] NodeOutput<TileJob> TileRandom )
{
    // ------------------------------------------------------------------
    //  1. Unique pixel index (for DI reservoir addressing)
    // ------------------------------------------------------------------
    uint pixelIdx = tid.x + tid.y * gImageWidth;

    // ------------------------------------------------------------------
    //  2. 30 float4 “register hogs”  → 120 scalar registers
    // ------------------------------------------------------------------
    float4 r00 = 0, r01 = 0, r02 = 0, r03 = 0, r04 = 0,
       r05 = 0, r06 = 0, r07 = 0, r08 = 0, r09 = 0,
       r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0,
       r15 = 0, r16 = 0, r17 = 0, r18 = 0, r19 = 0,
       r20 = 0, r21 = 0, r22 = 0, r23 = 0, r24 = 0,
       r25 = 0, r26 = 0, r27 = 0, r28 = 0, r29 = 0;    // 30 × float4

    // Only 16 iterations – tiny ALU load, keeps regs live
    [unroll(16)]
    for (uint i = 0; i < 16; ++i)
    {
        // 30 consecutive reservoir loads – observable memory traffic
        uint base = pixelIdx * 30 + i * 30;

        Reservoir_DI di0  = loadReservoirDI(g_Reservoirs_current_di, base +  0);
        Reservoir_DI di1  = loadReservoirDI(g_Reservoirs_current_di, base +  1);
        Reservoir_DI di2  = loadReservoirDI(g_Reservoirs_current_di, base +  2);


        // Convert the four fields you mentioned into float4s:
        // (L2_di, x2_di, W_di, M_di)
        r00 += float4(di0.L2_di,di0.M_di);
        r01 += float4(di1.L2_di,di1.M_di);
        r02 += float4(di2.L2_di,di2.M_di);
    }

    // ------------------------------------------------------------------
    // 3. Collapse to a single colour and store to gScratchPing[layer 15]
    // ------------------------------------------------------------------
    float4 sum0 = r00+r01+r02+r03;
    float4 sum1 = r10+r11+r12+r13;
    float4 sum2 = r20+r21+r22+r23;

    gScratchPing[uint3(tid.xy, 15)] = (sum0 + sum1 + sum2) * (1.0 / 480.0);

    // ------------------------------------------------------------------
    // 4. ONE TileJob record per tile (leader thread)
    // ------------------------------------------------------------------
    bool leader = (ltid.x == 0 && ltid.y == 0);

    GroupNodeOutputRecords<TileJob> rec =
        TileRandom.GetGroupNodeOutputRecords(leader ? 1 : 0);

    if (leader)
        rec[0].origin = tid.xy - ltid.xy;

    GroupMemoryBarrierWithGroupSync();
    if (leader) rec.OutputComplete();
}







