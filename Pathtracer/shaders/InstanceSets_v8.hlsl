// Instance sets: per cell and per instance, the bucket (a shared uniform cut's BLAS) within the error budget, and the
// partitioned TLAS operations for the instances whose bucket changed or that the TLAS does not hold yet. Assemblies'
// instances that need full detail expand into their parts, from a pool of part slots, and sway in the wind near the
// camera. Layouts mirror rdn/Clusters/InstanceSets.cpp.

#define SET_UNWRITTEN 0xFFFFFFFFu // instance and part slot state: not in the TLAS
#define SET_MIXED     0xFFFFFFFFu // cell state: its instances' buckets differ, or some wait for their operation

// An assembly instance's state, unless SET_UNWRITTEN: without ST_BLOCK its whole's bucket; with it, a pool block
// (ST_BLOCK_MASK) and, until ST_EXPANDED, the bucket its whole keeps meanwhile (ST_BUCKET_SHIFT).
#define ST_BLOCK        0x80000000u
#define ST_EXPANDED     0x40000000u // its parts are in the TLAS, its whole disabled
#define ST_WINDY        0x20000000u // it was last written swaying (whole or parts)
#define ST_BUCKET_MASK  0x1Fu       // a plain state's bucket
#define ST_FRESH        0x10000000u // a plain state written by an earlier pass this frame
#define ST_BUCKET_SHIFT 24u
#define ST_BLOCK_MASK   0x00FFFFFFu

struct SetDesc
{
    uint instanceBase, instanceCount, cellBase, cellCount;
    uint bucketBase, bucketCount, partBase, partCount;
    uint blockBase, blockCount, slotBase, partitionBase; // assemblies: pool blocks, part slots, their partitions
    uint mask;
    float wind;    // its sway's factor: 0 stands still
    uint pad1, pad2;
    float3 center; // the object's bounding sphere in set space
    float  radius;
};
struct SetBucket
{
    float  error; // set space; buckets coarsen with their index
    uint   record, hitGroup, flags;
    uint2  blas;
    uint2  pad;
    float4 base[3]; // its mesh to set space
};
struct SetPart
{
    uint   bucketBase, bucketCount;
    float  windWeight, pad;
    float3 pivot; // set space
    float  pad2;
};
struct SetCell
{
    float3 center;
    float  radius;
    uint   first, count, set;
    float  minScale, maxScale;
    uint   partition; // its region's, which all sets' instances there share
    uint2  pad;
};

cbuffer Sets : register(b0)
{
    float3 cCamera;         // in the sets' frame (the scene origin they were made at)
    float  cPixelsPerUnit;
    float  cThreshold;      // error budget in pixels
    float  cOffscreen;      // its factor off screen
    float  cZnear;
    uint   cCells;
    uint   cInstances;
    uint   cMaxWrites;
    uint   cMaxUpdates;
    uint   cTlasBase;       // the sets' first TLAS instance index
    uint   cPartitionBase;  // and partition
    uint   cWriteArgs;      // work offsets of the arguments
    uint   cUpdateArgs;
    uint   cWorkVaLo;
    uint   cWorkVaHi;
    uint   cTranslate;      // this frame: every set partition's translation becomes cDelta
    uint   cTranslateArgs;
    uint   cSlots;          // assemblies' part slots
    float3 cDelta;          // from the sets' frame to the scene origin's
    float  cTime;
    uint   cFrame;          // this frame's stamp in the motion records (never 0)
    uint   cRegions;        // the sets' partitions from cPartitionBase
    uint   cPoolTlasBase;   // the part slots' first TLAS instance index
    uint   cPoolPartitionBase;
    float  cWindDistance;
    float  cWindAmplitude;  // the bend: sideways sway per unit of height
    float  cWindFrequency;
    uint   cBlocks;
    uint   cSets;
    uint   cFreeLists;      // pool offsets
    uint   cPartStates;
    uint   cPoolPartitions;
    float  cWindFlutter;    // parts' own sway about their pivots, radians at full weight
    float3 cPad3;
    float4 cPlanes[5];      // in view, in the sets' frame: dot(xyz, p) + w >= 0
};

StructuredBuffer<SetDesc>   sets      : register(t0);
StructuredBuffer<SetBucket> buckets   : register(t1);
StructuredBuffer<SetCell>   cells     : register(t2);
// 32 bytes: position, largest axis scale, snorm16 rotation (x, y | z, w), snorm16 axis scales over it (x, y | z)
ByteAddressBuffer           instances : register(t3);
StructuredBuffer<SetPart>   parts     : register(t4);

RWStructuredBuffer<uint>  instanceState : register(u0);
RWStructuredBuffer<uint2> cellState     : register(u1); // the bucket all its instances hold (or SET_MIXED), visit
RWByteAddressBuffer       work          : register(u2); // counters, operations, the readback's copy, arguments
RWByteAddressBuffer       pool          : register(u3); // per set free blocks, free lists, part slots' buckets
// Per TLAS instance from cTlasBase: an instance's sway (shear now, shear before this frame's write, base height in the
// scene origin's frame, the frame stamp of that write), then a part slot's instance (x). The camera pass turns it into
// the surface's motion.
RWStructuredBuffer<float4> motion       : register(u4);

// work: counters (writes, updates, cells visited, instances in the TLAS, writes of expansions, collapses and sways,
// instances expanded), the operation count, the operations, the readback's copy. One write operation takes every
// write (a second one's writes did not reach the TLAS), and an instance is written at most once a frame, since a
// build applies its operations in no set order.
#define WORK_WRITES       0u
#define WORK_UPDATES      4u
#define WORK_CELLS        8u
#define WORK_WRITTEN      12u
#define WORK_BLOCK_WRITES 16u
#define WORK_EXPANDED     20u
#define WORK_OP_COUNT     32u
#define WORK_OPS          40u
#define WORK_FEEDBACK     160u
#define WRITE_BYTES       104u

// 0: outside the view, 1: partly inside, 2: inside.
uint SphereVisibility(float3 c, float r)
{
    uint v = 2u;
    [unroll] for (uint i = 0; i < 5u; ++i)
    {
        const float d = dot(cPlanes[i].xyz, c) + cPlanes[i].w;
        if (d < -r) return 0u;
        if (d < r) v = 1u;
    }
    return v;
}

// The coarsest of count buckets from first whose error, at distance d and scale, stays within threshold pixels.
uint BucketOf(uint first, uint count, float d, float scale, float threshold)
{
    const float allowed = threshold * d / (max(scale, 1e-6f) * cPixelsPerUnit);
    uint k = 0u;
    for (uint b = 1u; b < count; ++b)
        if (buckets[first + b].error <= allowed) k = b;
    return k;
}

float SnormOf(int v) { return max((float)v / 32767.0f, -1.0f); }

float4 RotationOf(uint2 q)
{
    const float4 r = float4(SnormOf(int(q.x << 16) >> 16), SnormOf(int(q.x) >> 16),
                            SnormOf(int(q.y << 16) >> 16), SnormOf(int(q.y) >> 16));
    return r * rsqrt(max(dot(r, r), 1e-12f));
}

float3x3 MatrixOf(float4 q)
{
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - w * z),        2.0f * (x * z + w * y),
                    2.0f * (x * y + w * z),        1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - w * x),
                    2.0f * (x * z - w * y),        2.0f * (y * z + w * x),        1.0f - 2.0f * (x * x + y * y));
}

uint Hash(uint v)
{
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}

// An instance as the per-instance passes see it.
struct SetInstance
{
    uint ci, i;      // cell, instance
    uint set;
    uint partition;  // its cell's
    SetDesc s;
    float4 ps;       // position, largest axis scale
    float3x3 RS;     // rotation after axis scale
    float d;         // distance to its bounding sphere
    float threshold; // its error budget: looser off screen
};

#define SET_CELL_ROWS 65535u // cells per dispatch slab (z), rows of groups (y) of a dispatch's most

bool LoadInstance(uint3 group, uint3 thread, out SetInstance o)
{
    o = (SetInstance)0;
    o.ci = group.y + group.z * SET_CELL_ROWS;
    if (o.ci >= cCells || cellState[o.ci].y == 0u) return false;
    const SetCell c = cells[o.ci];
    const uint k = group.x * 256u + thread.x;
    if (k >= c.count) return false;
    o.i = c.first + k;
    o.set = c.set;
    o.partition = cPartitionBase + c.partition;
    o.s = sets[c.set];
    o.ps = asfloat(instances.Load4(32u * o.i));
    const uint4 ra = instances.Load4(32u * o.i + 16u);
    const float3x3 R = MatrixOf(RotationOf(ra.xy));
    const float3 axes = o.ps.w * float3(SnormOf(int(ra.z << 16) >> 16), SnormOf(int(ra.z) >> 16),
                                        SnormOf(int(ra.w << 16) >> 16));
    o.RS = float3x3(R[0] * axes, R[1] * axes, R[2] * axes);
    const float3 center = o.ps.xyz + mul(o.RS, o.s.center);
    const float  radius = o.s.radius * o.ps.w;
    o.d = max(length(center - cCamera) - radius, cZnear);
    o.threshold = SphereVisibility(center, radius) != 0u ? cThreshold : cThreshold * cOffscreen;
    return true;
}

// An assembly instance wants its parts while its whole would need full detail; it gives them back a little farther
// than it takes them.
bool WantsParts(SetInstance o, bool expanded)
{
    return o.s.partCount != 0u &&
           BucketOf(o.s.bucketBase, o.s.bucketCount, expanded ? o.d * 0.8f : o.d, o.ps.w, o.threshold) == 0u;
}

// D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS at byte at: the transform's rows, then the rest.
void WriteArgs(uint at, float3x3 M, float3 t, uint record, uint mask, uint hitGroup, uint flags, uint2 blas, uint index,
               uint partition)
{
    work.Store4(at,       asuint(float4(M[0], t.x)));
    work.Store4(at + 16u, asuint(float4(M[1], t.y)));
    work.Store4(at + 32u, asuint(float4(M[2], t.z)));
    work.Store4(at + 48u, uint4(record, mask, hitGroup, flags));
    work.Store4(at + 64u, uint4(blas, index, partition));
    work.Store4(at + 80u, uint4(0, 0, 0, 0)); // no explicit AABB
    work.Store2(at + 96u, uint2(0, 0));
}

// The wind bends a whole instance alike, whole or parts, so its pieces never part: a shear about its base, sideways
// along the world's x in proportion to height, swinging with a phase of its own. 0 out of the wind's reach.
float WindShear(SetInstance o)
{
    if (o.d >= cWindDistance || o.s.wind == 0.0f) return 0.0f;
    const float phase = (float)(Hash(o.i) & 0xFFFFu) * (6.2831853f / 65536.0f);
    const float w = cTime * cWindFrequency;
    return o.s.wind * cWindAmplitude * (sin(w + phase) + 0.35f * sin(2.3f * w + 1.7f * phase));
}

float3x3 ShearOf(float shear) { return float3x3(1.0f, shear, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f); }

// The instance as bucket b: its mesh in set space, placed, bent by shear. visible false hides it behind an instance
// mask no ray matches, its BLAS kept, so that it may still be updated later.
void WriteBucket(uint at, SetInstance o, uint b, uint index, uint partition, float shear, bool visible = true)
{
    const SetBucket k = buckets[b];
    const float3x3 S = mul(ShearOf(shear), o.RS);
    const float3x3 M = mul(S, float3x3(k.base[0].xyz, k.base[1].xyz, k.base[2].xyz));
    const float3   t = o.ps.xyz + mul(S, float3(k.base[0].w, k.base[1].w, k.base[2].w));
    WriteArgs(at, M, t, k.record, visible ? o.s.mask : 0u, k.hitGroup, k.flags, k.blas, index, partition);
}

// Reserves n consecutive writes, all or none: an expansion or a sway is never written in part.
bool ReserveBlock(uint n, out uint first)
{
    first = 0u;
    uint cur = work.Load(WORK_WRITES);
    [loop] for (uint tries = 0u; tries < 64u; ++tries)
    {
        if (cur + n > cMaxWrites) return false;
        uint was;
        work.InterlockedCompareExchange(WORK_WRITES, cur, cur + n, was);
        if (was == cur)
        {
            first = cur;
            uint ignored;
            work.InterlockedAdd(WORK_BLOCK_WRITES, n, ignored);
            return true;
        }
        cur = was;
    }
    return false;
}

uint SlotOf(SetInstance o, uint block, uint p) { return o.s.slotBase + block * o.s.partCount + p; }
uint SlotPartition(SetInstance o, uint block, uint p)
{
    return cPoolPartitionBase + o.s.partitionBase + (block * o.s.partCount + p) / 1024u;
}

// Part p of an expanded instance, as bucket pb: bent with the instance by shear, and fluttering by its own angle about
// its pivot (in set space, about z) when it is a part that may: a twig, not a piece cut from a whole. visible false
// hides it, as WriteBucket.
void WritePart(uint at, SetInstance o, uint block, uint p, uint pb, float shear, bool visible = true)
{
    const SetPart   part = parts[o.s.partBase + p];
    const SetBucket k    = buckets[pb];
    float angle = 0.0f;
    if (shear != 0.0f && part.windWeight > 0.0f)
    {
        const float phase = (float)(Hash(o.i * 257u + p) & 0xFFFFu) * (6.2831853f / 65536.0f);
        angle = cWindFlutter * part.windWeight * sin(3.1f * cTime * cWindFrequency + phase);
    }
    float sn, cs;
    sincos(angle, sn, cs);
    const float3x3 F  = float3x3(cs, -sn, 0.0f, sn, cs, 0.0f, 0.0f, 0.0f, 1.0f);
    const float3x3 S  = mul(ShearOf(shear), o.RS);
    const float3x3 B  = float3x3(k.base[0].xyz, k.base[1].xyz, k.base[2].xyz);
    const float3   Bt = float3(k.base[0].w, k.base[1].w, k.base[2].w);
    const float3x3 M  = mul(S, mul(F, B));
    const float3   t  = mul(S, mul(F, Bt - part.pivot) + part.pivot) + o.ps.xyz;
    WriteArgs(at, M, t, k.record, visible ? o.s.mask : 0u, k.hitGroup, k.flags, k.blas,
              cPoolTlasBase + SlotOf(o, block, p), SlotPartition(o, block, p));
}

// The instance's transform was written this frame with shear: its sway for the motion vectors.
void RecordMotion(SetInstance o, float shear, bool fresh)
{
    const float before = fresh ? shear : motion[o.i].x;
    motion[o.i] = float4(shear, before, o.ps.y + cDelta.y, asfloat(cFrame));
}

uint PartBucket(SetInstance o, uint p)
{
    const SetPart part = parts[o.s.partBase + p];
    return part.bucketBase + BucketOf(part.bucketBase, part.bucketCount, o.d, o.ps.w, o.threshold);
}

// A new TLAS: no instance is in it, and every pool block is free.
[numthreads(64, 1, 1)]
void SetReset(uint3 id : SV_DispatchThreadID)
{
    const uint x = id.x;
    if (x < cInstances) instanceState[x] = SET_UNWRITTEN;
    if (x < cInstances) motion[x] = 0.0f;
    if (x < cCells) cellState[x] = uint2(SET_MIXED, 1u);
    if (x < cSlots) pool.Store(cPartStates + 4u * x, SET_UNWRITTEN);
    if (x < cSets) pool.Store(4u * x, sets[x].blockCount);
    if (x < cBlocks)
        for (uint s = 0u; s < cSets; ++s)
            if (x >= sets[s].blockBase && x < sets[s].blockBase + sets[s].blockCount)
                pool.Store(cFreeLists + 4u * x, x - sets[s].blockBase);
    if (x == 0u)
    {
        work.Store4(0, uint4(0, 0, 0, 0));
        work.Store4(16, uint4(0, 0, 0, 0));
    }
}

// Per cell: the bucket range of its instances, from its bounds; a cell whose instances all hold the one bucket that
// range allows is skipped. Cells in the wind's reach, and assemblies' near enough for parts, are always visited. Every
// set partition's translation, when it changes.
[numthreads(64, 1, 1)]
void SetCells(uint3 id : SV_DispatchThreadID)
{
    // D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS: the regions' partitions, then the slots'.
    if (cTranslate != 0u && id.x < cRegions + cPoolPartitions)
        work.Store4(cTranslateArgs + 16u * id.x, uint4(cPartitionBase + id.x, asuint(cDelta)));
    if (id.x >= cCells) return;
    const SetCell c = cells[id.x];
    const SetDesc s = sets[c.set];
    const float dist = length(c.center - cCamera);
    const uint  vis  = SphereVisibility(c.center, c.radius);
    const uint  fine   = BucketOf(s.bucketBase, s.bucketCount, max(dist - c.radius, cZnear), c.maxScale,
                                  vis != 0u ? cThreshold : cThreshold * cOffscreen);
    const uint  coarse = BucketOf(s.bucketBase, s.bucketCount, max(dist + c.radius, cZnear), c.minScale,
                                  vis == 2u ? cThreshold : cThreshold * cOffscreen);
    const uint  held   = fine == coarse ? fine : SET_MIXED;
    const bool  skip   = held != SET_MIXED && cellState[id.x].x == held && !(s.partCount != 0u && fine == 0u) &&
                         (dist - c.radius >= cWindDistance || s.wind == 0.0f);
    cellState[id.x] = uint2(held, skip ? 0u : 1u);
    if (skip) return;
    uint ignored;
    work.InterlockedAdd(WORK_CELLS, 1u, ignored);
}

// Assembly instances that no longer want their parts: their slots disabled, their whole back, their block freed (the
// only pass that frees, after the only one that takes, so a block freed is taken no sooner than the next frame).
[numthreads(256, 1, 1)]
void SetCollapse(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    SetInstance o;
    if (!LoadInstance(group, thread, o) || o.s.partCount == 0u) return;
    const uint st = instanceState[o.i];
    if (st == SET_UNWRITTEN || (st & ST_BLOCK) == 0u) return;
    const bool expanded = (st & ST_EXPANDED) != 0u;
    if (WantsParts(o, expanded)) return;
    const uint block = st & ST_BLOCK_MASK;
    cellState[o.ci].x = SET_MIXED;
    uint next = (st >> ST_BUCKET_SHIFT) & 0x1Fu;
    if (expanded)
    {
        uint first;
        if (!ReserveBlock(o.s.partCount + 1u, first)) return;
        for (uint p = 0u; p < o.s.partCount; ++p)
        {
            const uint at = cPartStates + 4u * SlotOf(o, block, p);
            WritePart(cWriteArgs + WRITE_BYTES * (first + p), o, block, p, pool.Load(at), 0.0f, false);
            pool.Store(at, SET_UNWRITTEN);
        }
        const float shear = WindShear(o);
        next = BucketOf(o.s.bucketBase, o.s.bucketCount, o.d, o.ps.w, o.threshold) | (shear != 0.0f ? ST_WINDY : 0u) |
               ST_FRESH;
        WriteBucket(cWriteArgs + WRITE_BYTES * (first + o.s.partCount), o, o.s.bucketBase + (next & ST_BUCKET_MASK),
                    cTlasBase + o.i, o.partition, shear);
        RecordMotion(o, shear, false);
    }
    instanceState[o.i] = next;
    uint k;
    pool.InterlockedAdd(4u * o.set, 1u, k);
    pool.Store(cFreeLists + 4u * (o.s.blockBase + k), block);
}

// Assembly instances that want their parts take a pool block (the only pass that takes, before the one that frees);
// they expand once their writes find room.
[numthreads(256, 1, 1)]
void SetAlloc(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    SetInstance o;
    if (!LoadInstance(group, thread, o) || o.s.partCount == 0u) return;
    const uint st = instanceState[o.i];
    if (st == SET_UNWRITTEN || (st & ST_BLOCK) != 0u || !WantsParts(o, false)) return;
    uint k;
    pool.InterlockedAdd(4u * o.set, 0xFFFFFFFFu, k);
    if (k == 0u || k > o.s.blockCount)
    {
        uint ignored;
        pool.InterlockedAdd(4u * o.set, 1u, ignored);
        return;
    }
    const uint block = pool.Load(cFreeLists + 4u * (o.s.blockBase + k - 1u));
    instanceState[o.i] = ST_BLOCK | ((st & ST_BUCKET_MASK) << ST_BUCKET_SHIFT) | block;
    cellState[o.ci].x = SET_MIXED;
}

// Per instance of a visited cell (CELL_INSTANCES / 256 groups per cell): its bucket, and its TLAS operation when that
// changed: a write when the TLAS lacks it or its record changes (the stand-in), an update of its BLAS otherwise. An
// assembly instance with a block expands into its parts; expanded ones sway near the camera and follow their parts'
// buckets. One that finds no room this frame waits, its cell marked so it is visited again.
[numthreads(256, 1, 1)]
void SetInstances(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID)
{
    SetInstance o;
    if (!LoadInstance(group, thread, o)) return;
    const uint st = instanceState[o.i];
    uint slot;

    if (st != SET_UNWRITTEN && (st & ST_BLOCK) != 0u)
    {
        cellState[o.ci].x = SET_MIXED;
        const uint block = st & ST_BLOCK_MASK;
        const float shear = WindShear(o);
        const bool windy = shear != 0.0f;
        uint next = st;
        if ((st & ST_EXPANDED) == 0u)
        {
            uint first;
            if (!ReserveBlock(o.s.partCount + 1u, first)) return;
            for (uint p = 0u; p < o.s.partCount; ++p)
            {
                const uint pb = PartBucket(o, p);
                WritePart(cWriteArgs + WRITE_BYTES * (first + p), o, block, p, pb, shear);
                pool.Store(cPartStates + 4u * SlotOf(o, block, p), pb);
                motion[cInstances + SlotOf(o, block, p)] = float4(asfloat(o.i), 0.0f, 0.0f, 0.0f);
            }
            RecordMotion(o, shear, false);
            WriteBucket(cWriteArgs + WRITE_BYTES * (first + o.s.partCount), o,
                        o.s.bucketBase + ((st >> ST_BUCKET_SHIFT) & ST_BUCKET_MASK), cTlasBase + o.i,
                        o.partition, 0.0f, false);
            next = ST_BLOCK | ST_EXPANDED | (windy ? ST_WINDY : 0u) | block;
        }
        else if (windy || (st & ST_WINDY) != 0u)
        {
            // Swaying, or settling back once out of the wind's reach.
            uint first;
            if (ReserveBlock(o.s.partCount, first))
            {
                for (uint p = 0u; p < o.s.partCount; ++p)
                {
                    const uint pb = PartBucket(o, p);
                    WritePart(cWriteArgs + WRITE_BYTES * (first + p), o, block, p, pb, shear);
                    pool.Store(cPartStates + 4u * SlotOf(o, block, p), pb);
                }
                RecordMotion(o, shear, false);
                next = (st & ~ST_WINDY) | (windy ? ST_WINDY : 0u);
            }
        }
        else
        {
            for (uint p = 0u; p < o.s.partCount; ++p)
            {
                const uint pb = PartBucket(o, p);
                const uint at = cPartStates + 4u * SlotOf(o, block, p);
                if (pool.Load(at) == pb) continue;
                work.InterlockedAdd(WORK_UPDATES, 1u, slot);
                if (slot >= cMaxUpdates) continue;
                // D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS
                work.Store4(cUpdateArgs + 16u * slot, uint4(cPoolTlasBase + SlotOf(o, block, p), buckets[pb].hitGroup,
                                                           buckets[pb].blas));
                pool.Store(at, pb);
            }
        }
        instanceState[o.i] = next;
        if ((next & ST_EXPANDED) != 0u)
        {
            uint ignored;
            work.InterlockedAdd(WORK_EXPANDED, 1u, ignored);
        }
        return;
    }

    // Written by the collapse this frame already.
    if (st != SET_UNWRITTEN && (st & ST_FRESH) != 0u)
    {
        instanceState[o.i] = st & ~ST_FRESH;
        cellState[o.ci].x = SET_MIXED;
        return;
    }
    // A whole in the wind's reach is written bent every frame, and once more straight when it leaves.
    const uint  b     = BucketOf(o.s.bucketBase, o.s.bucketCount, o.d, o.ps.w, o.threshold);
    const float shear = WindShear(o);
    const bool  windy = shear != 0.0f, wasWindy = st != SET_UNWRITTEN && (st & ST_WINDY) != 0u;
    if (windy) cellState[o.ci].x = SET_MIXED;
    if (st == b && !windy) return;
    const uint kb = o.s.bucketBase + b;
    if (st == SET_UNWRITTEN || windy || wasWindy ||
        buckets[o.s.bucketBase + (st & ST_BUCKET_MASK)].record != buckets[kb].record)
    {
        work.InterlockedAdd(WORK_WRITES, 1u, slot);
        if (slot >= cMaxWrites) { cellState[o.ci].x = SET_MIXED; return; }
        WriteBucket(cWriteArgs + WRITE_BYTES * slot, o, kb, cTlasBase + o.i, o.partition, shear);
        RecordMotion(o, shear, st == SET_UNWRITTEN);
        if (st == SET_UNWRITTEN)
        {
            uint ignored;
            work.InterlockedAdd(WORK_WRITTEN, 1u, ignored);
        }
    }
    else
    {
        work.InterlockedAdd(WORK_UPDATES, 1u, slot);
        if (slot >= cMaxUpdates) { cellState[o.ci].x = SET_MIXED; return; }
        // D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS
        work.Store4(cUpdateArgs + 16u * slot, uint4(cTlasBase + o.i, buckets[kb].hitGroup, buckets[kb].blas));
    }
    instanceState[o.i] = b | (windy ? ST_WINDY : 0u);
}

// D3D12_RTAS_PARTITIONED_TLAS_OPERATION k: type, argument count, argument address and stride.
void StoreOp(uint k, uint type, uint count, uint64_t va, uint stride)
{
    const uint at = WORK_OPS + 24u * k;
    work.Store2(at,       uint2(type, count));
    work.Store2(at + 8u,  uint2(uint(va), uint(va >> 32)));
    work.Store2(at + 16u, uint2(stride, 0u));
}

// The frame's operations over the arguments written, none empty; its counts for the readback; fresh counters.
[numthreads(1, 1, 1)]
void SetFinalize(uint3 id : SV_DispatchThreadID)
{
    const uint4 n = work.Load4(0);
    const uint2 m = work.Load2(16);
    const uint writes = min(n.x, cMaxWrites), updates = min(n.y, cMaxUpdates);
    const uint64_t va = (uint64_t(cWorkVaHi) << 32) | cWorkVaLo;
    uint ops = 0u;
    if (writes != 0u) StoreOp(ops++, 0u, writes, va + cWriteArgs, WRITE_BYTES);       // WRITE_INSTANCE
    if (updates != 0u) StoreOp(ops++, 1u, updates, va + cUpdateArgs, 16u);            // UPDATE_INSTANCE
    if (cTranslate != 0u) StoreOp(ops++, 2u, cRegions + cPoolPartitions, va + cTranslateArgs, 16u); // TRANSLATE_PARTITION
    work.Store(WORK_OP_COUNT, ops);
    work.Store4(WORK_FEEDBACK, uint4(writes, updates, n.z, n.w));
    work.Store4(WORK_FEEDBACK + 16u, uint4(m.x, m.y, 0u, 0u));
    work.Store3(0, uint3(0, 0, 0));
    work.Store2(16, uint2(0, 0));
}
