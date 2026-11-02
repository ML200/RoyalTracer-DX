// GPU cluster LOD: per instance, the cut of its mesh's hierarchy for its view, a cluster BLAS over it when it changed
// (batched per size class, counts written here), the partitioned TLAS updates, and feedback for the pager. Layouts
// mirror rdn/Clusters/LodGpu.h.

#define LOD_NONE 0xFFFFFFFFu
#define LOD_RING 4u
#define LOD_BASE 0xFFFFFFFFu // state.current: the mesh's own BLAS (its coarsest cut)
#define LOD_ROW (65535u * 64u) // threads per dispatch row (LodGpu.cpp's DISPATCH_ROW groups)

struct LodMesh
{
    uint groupBase, groupCount;     // global group range
    uint clusterBase, clusterCount; // global hierarchy cluster range
    uint clasBase;                  // into clasAddr: its fixed clusters, then its hierarchy's
    uint fixedCount, rootCount;     // root clusters lead the hierarchy
    uint pagerMesh;
    uint2 baseBlas;
    uint celled;    // its instances are cells, handed their clusters by target
    uint pad;
};
struct LodInstance
{
    uint mesh;
    uint cutAddr;   // byte offset in work: its CLAS address list (fixed first)
    uint cutIdx;    // into cutIdx: two halves of clusterCount
    uint tlasIndex; // its partitioned TLAS instance
    uint hitGroup;
    uint cls;       // unused
    uint groupWork, clusterWork; // its first work items; a cell's groupWork is its hierarchy's first cell's
    uint clusterFirst, clusterCount; // unused, and the capacity of its cut lists (a cell's clusters)
    uint fixedCount;                 // fixed clusters leading its cuts
    uint pad;
};
struct LodGroup
{
    float3 center;
    float radius;
    float error;
    uint depFirst, depCount, mesh;
};
struct LodState
{
    uint current;          // ring entry, or LOD_BASE
    uint retired[LOD_RING];
    uint hashA, hashB, count;
    uint half, currentCount;
    uint2 ring;            // the ring of BLAS entries it builds into (the CPU's allocation), their class, its handle
    uint ringClass, ringId;
    uint2 pending;         // a ring the CPU handed over, taken at the next build
    uint pendingClass, pendingId;
    uint drop;             // the CPU wants the ring back once the instance traces its mesh's own BLAS
    uint pad;
};
struct LodUpdate
{
    uint index, offset, value, pad; // table update: residency[index] = value, or clasAddr when offset == 1
};

cbuffer Lod : register(b0)
{
    uint  cFrame;
    uint  cInstances;
    uint  cGroupWork;
    uint  cClusterWork;
    float cPixelsPerUnit;
    float cThreshold;
    uint  cGroups;          // global groups
    uint  cMaxRequests;
    uint  cClasses;
    uint  cUpdates;         // table updates this frame
    uint  cClassArgs;       // work offsets: class args, class dests, class counts, class budget use
    uint  cClassDests;
    uint  cClassCounts;
    uint  cClassBudgets;
    uint  cTlasArgs;        // work offset of the TLAS update args; their count and the op follow
    uint  cWorkVaLo;        // work buffer address, for addresses written into arguments
    uint  cWorkVaHi;
};

StructuredBuffer<LodMesh>     meshes        : register(t0);
StructuredBuffer<LodInstance> instances     : register(t1);
StructuredBuffer<LodGroup>    groups        : register(t2);
StructuredBuffer<uint2>       clusters      : register(t3);  // global group, global refined group (or LOD_NONE)
StructuredBuffer<uint>        deps          : register(t4);  // global groups
StructuredBuffer<float4>      views         : register(t5);  // camera in mesh space, znear
StructuredBuffer<uint>        groupWorkOff  : register(t6);  // per instance, and the total
StructuredBuffer<uint>        clusterWorkOff: register(t7);
StructuredBuffer<uint>        rewritten     : register(t8);  // per instance: the TLAS instance was written this frame
StructuredBuffer<uint>        target        : register(t9);  // per cluster of every mesh: the instance tracing it
StructuredBuffer<uint>        classInfo     : register(t10); // per class: budget of CLAS references, arg base, size,
                                                               // ring entry bytes, argument capacity
StructuredBuffer<LodUpdate>   updates       : register(t11);

RWStructuredBuffer<LodState>  state         : register(u0);
RWStructuredBuffer<uint>      finePrime     : register(u1); // per group work item
RWStructuredBuffer<uint>      cutIdx        : register(u2);
RWStructuredBuffer<uint4>     counters      : register(u3); // per instance: count, non-root count, hash A, hash B
RWByteAddressBuffer           work          : register(u4);
RWByteAddressBuffer           feedback      : register(u5); // FEEDBACK_HEADER, then requests
RWStructuredBuffer<uint>      lastUsed      : register(u6); // per global group
RWStructuredBuffer<uint>      want          : register(u7); // per global group, float bits
RWStructuredBuffer<uint>      requested     : register(u8); // per global group: frame + 1 it was requested
RWStructuredBuffer<uint>      residency     : register(u9); // per global group
RWStructuredBuffer<uint2>     clasAddr      : register(u10);

#define FEEDBACK_HEADER 32u // request count, builds, their CLAS references, traced clusters, TLAS updates

uint FindInstance(StructuredBuffer<uint> offsets, uint item)
{
    uint lo = 0, hi = cInstances; // offsets[lo] <= item < offsets[hi]
    while (hi - lo > 1u)
    {
        const uint mid = (lo + hi) >> 1;
        if (offsets[mid] <= item) lo = mid; else hi = mid;
    }
    return lo;
}

// A work item across dispatch rows.
uint Item(uint3 id) { return id.y * LOD_ROW + id.x; }

uint ClassSize(uint cls) { return classInfo[2u * cClasses + cls]; }

uint2 EntryAddr(uint2 ring, uint cls, uint e)
{
    const uint64_t a = (uint64_t(ring.y) << 32 | ring.x) + uint64_t(classInfo[3u * cClasses + cls]) * e;
    return uint2(uint(a), uint(a >> 32));
}

// A ring the instance no longer builds into, for the CPU to free once no frame in flight traces it after frame last.
void ReleaseRing(uint id, uint last)
{
    uint slot;
    feedback.InterlockedAdd(20, 1u, slot);
    if (slot < cInstances) feedback.Store2(FEEDBACK_HEADER + 16u * cMaxRequests + 8u * slot, uint2(id, last));
}

uint Hash(uint v)
{
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return v;
}

// Residency and CLAS address changes, ordered in the list before the passes that read them.
[numthreads(64, 1, 1)]
void LodApply(uint3 tid : SV_DispatchThreadID)
{
    const uint3 id = uint3(Item(tid), 0, 0);
    if (id.x >= cUpdates) return;
    const LodUpdate u = updates[id.x];
    if (u.offset == 0u) residency[u.index] = u.value;
    else if (u.offset == 1u) clasAddr[u.index].x = u.value;
    else if (u.offset == 2u) clasAddr[u.index].y = u.value;
    else if (u.offset == 3u) state[u.index].pending.x = u.value;
    else if (u.offset == 4u) state[u.index].pending.y = u.value;
    else if (u.offset == 5u) state[u.index].pendingClass = u.value;
    else if (u.offset == 6u) state[u.index].pendingId = u.value;
    else state[u.index].drop = u.value;
}

[numthreads(64, 1, 1)]
void LodClear(uint3 tid : SV_DispatchThreadID)
{
    const uint3 id = uint3(Item(tid), 0, 0);
    const uint i = id.x;
    if (i < cInstances) counters[i] = uint4(0, 0, 0, 0);
    if (i < cGroups) want[i] = 0;
    if (i < cClasses)
    {
        work.Store(cClassCounts + 4u * i, 0u);
        work.Store(cClassBudgets + 4u * i, 0u);
    }
    if (i == 0u)
    {
        feedback.Store4(0, uint4(0, 0, 0, 0));
        feedback.Store4(16, uint4(0, 0, 0, 0)); // padding up to the requests
        work.Store(cTlasArgs + 16u * cInstances, 0u);
    }
}

// Per instance and group: is its simplification fine enough for the view (or is the group absent, so its coarser level
// stands in)? Wanted absent pages whose deps are resident are requested, once a frame.
[numthreads(64, 1, 1)]
void LodGroups(uint3 tid : SV_DispatchThreadID)
{
    const uint3 id = uint3(Item(tid), 0, 0);
    if (id.x >= cGroupWork) return;
    const uint i = FindInstance(groupWorkOff, id.x);
    const LodInstance inst = instances[i];
    const LodMesh m = meshes[inst.mesh];
    const uint gl = m.groupBase + (id.x - inst.groupWork);
    const LodGroup g = groups[gl];
    const float4 v = views[i];
    // The roots' simplification is never fine; they stay resident.
    const bool root = g.error >= 3.0e38f;
    bool fine = false;
    float over = 0.0f;
    if (!root)
    {
        const float d = max(length(g.center - v.xyz) - g.radius, v.w);
        over = g.error * cPixelsPerUnit / (cThreshold * d);
        fine = over <= 1.0f;
    }
    const bool resident = residency[gl] != 0u;
    if (!fine && !root)
    {
        InterlockedMax(want[gl], asuint(over));
        if (!resident)
        {
            bool ready = true;
            for (uint k = 0; k < g.depCount && ready; ++k) ready = residency[deps[g.depFirst + k]] != 0u;
            if (ready)
            {
                uint before;
                InterlockedMax(requested[gl], cFrame + 1u, before);
                if (before < cFrame + 1u)
                {
                    uint slot;
                    feedback.InterlockedAdd(0, 1u, slot);
                    if (slot < cMaxRequests)
                        feedback.Store4(FEEDBACK_HEADER + 16u * slot, uint4(m.pagerMesh, gl - m.groupBase, asuint(over), 0u));
                }
            }
        }
    }
    finePrime[id.x] = (fine || !resident) ? 1u : 0u;
}

// Per instance and hierarchy cluster: in the cut when its group's simplification is not fine enough and its own
// triangles are. Into the half of the tracing instance's list that is not traced (a cell's, for a celled hierarchy).
[numthreads(64, 1, 1)]
void LodClusters(uint3 tid : SV_DispatchThreadID)
{
    const uint3 id = uint3(Item(tid), 0, 0);
    if (id.x >= cClusterWork) return;
    const uint i = FindInstance(clusterWorkOff, id.x);
    const LodInstance inst = instances[i];
    const LodMesh m = meshes[inst.mesh];
    const uint c = id.x - inst.clusterWork;
    const uint2 cl = clusters[m.clusterBase + c];
    if (finePrime[inst.groupWork + (cl.x - m.groupBase)] != 0u) return;
    if (cl.y != LOD_NONE && finePrime[inst.groupWork + (cl.y - m.groupBase)] == 0u) return;
    const uint t = m.celled != 0u ? target[m.clusterBase + c] : i;
    if (t == LOD_NONE) return;
    const LodInstance to = instances[t];
    uint slot;
    InterlockedAdd(counters[t].x, 1u, slot);
    const uint half = 1u - state[t].half;
    cutIdx[to.cutIdx + half * to.clusterCount + slot] = c;
    work.Store2(to.cutAddr + 8u * (to.fixedCount + slot), clasAddr[m.clasBase + m.fixedCount + c]);
    InterlockedXor(counters[t].z, Hash(c * 2u + 1u));
    InterlockedAdd(counters[t].w, Hash(c * 2u + 2u));
    if (c >= m.rootCount) InterlockedAdd(counters[t].y, 1u);
}

void EmitTlasUpdate(LodInstance inst, uint2 blas)
{
    uint slot;
    work.InterlockedAdd(cTlasArgs + 16u * cInstances, 1u, slot);
    work.Store4(cTlasArgs + 16u * slot, uint4(inst.tlasIndex, inst.hitGroup, blas.x, blas.y));
}

// Per instance: the coarsest cut traces the mesh's own BLAS; a changed cut gets a cluster BLAS in a free entry of its
// ring (or entry 0 of a ring the CPU handed over, releasing the last), within its class's size and budget; either way the
// TLAS instance follows, as it does after the CPU rewrote it. A ring asked back goes once the cut is the coarsest.
[numthreads(64, 1, 1)]
void LodPlan(uint3 tid : SV_DispatchThreadID)
{
    const uint3 id = uint3(Item(tid), 0, 0);
    const uint i = id.x;
    if (i >= cInstances) return;
    const LodInstance inst = instances[i];
    const LodMesh m = meshes[inst.mesh];
    LodState s = state[i];
    const uint4 cnt = counters[i];
    bool moved = false;
    if (cnt.y == 0u)
    {
        if (s.current != LOD_BASE)
        {
            s.retired[s.current] = cFrame;
            s.current = LOD_BASE;
            s.currentCount = 0u;
            moved = true;
        }
        if (s.drop != 0u && s.ringId != LOD_NONE)
        {
            uint last = 0;
            for (uint e = 0; e < LOD_RING; ++e) last = max(last, s.retired[e]);
            ReleaseRing(s.ringId, last);
            s.ringId = LOD_NONE;
        }
        s.drop = 0u;
    }
    else if (s.current == LOD_BASE || cnt.x != s.count || cnt.z != s.hashA || cnt.w != s.hashB || s.pendingId != LOD_NONE)
    {
        const uint refs = inst.fixedCount + cnt.x;
        uint entry = LOD_NONE, cls = 0;
        uint2 ring = uint2(0, 0);
        const bool fresh = s.pendingId != LOD_NONE && refs <= ClassSize(s.pendingClass);
        if (fresh)
        {
            entry = 0u;
            cls = s.pendingClass;
            ring = s.pending;
        }
        else if (s.ringId != LOD_NONE && refs <= ClassSize(s.ringClass))
        {
            for (uint e = 0; e < LOD_RING && entry == LOD_NONE; ++e)
                if (e != s.current && (s.retired[e] == 0u || cFrame >= s.retired[e] + LOD_RING)) entry = e;
            cls = s.ringClass;
            ring = s.ring;
        }
        uint used = 0;
        if (entry != LOD_NONE) work.InterlockedAdd(cClassBudgets + 4u * cls, refs, used);
        if (entry != LOD_NONE && used + refs <= classInfo[cls])
        {
            uint slot;
            work.InterlockedAdd(cClassCounts + 4u * cls, 1u, slot);
            const uint args = cClassArgs + 16u * (classInfo[cClasses + cls] + slot);
            const uint64_t list = (uint64_t(cWorkVaHi) << 32 | cWorkVaLo) + inst.cutAddr;
            work.Store4(args, uint4(refs, 8u, uint(list), uint(list >> 32)));
            work.Store2(cClassDests + 8u * (classInfo[cClasses + cls] + slot), EntryAddr(ring, cls, entry));
            if (fresh)
            {
                // The last ring's entries are traced until this frame at the latest.
                if (s.ringId != LOD_NONE) ReleaseRing(s.ringId, cFrame);
                s.ring = s.pending;
                s.ringClass = s.pendingClass;
                s.ringId = s.pendingId;
                s.pendingId = LOD_NONE;
                for (uint e = 0; e < LOD_RING; ++e) s.retired[e] = 0u;
            }
            else if (s.current != LOD_BASE) s.retired[s.current] = cFrame;
            s.current = entry;
            s.count = cnt.x;
            s.hashA = cnt.z;
            s.hashB = cnt.w;
            s.half = 1u - s.half;
            s.currentCount = cnt.x;
            moved = true;
            uint ignored;
            feedback.InterlockedAdd(4, 1u, ignored);
            feedback.InterlockedAdd(8, refs, ignored);
        }
    }
    if (moved || rewritten[i] != 0u)
        EmitTlasUpdate(inst, s.current == LOD_BASE ? m.baseBlas : EntryAddr(s.ring, s.ringClass, s.current));
    state[i] = s;
    uint ignored;
    feedback.InterlockedAdd(12, inst.fixedCount + (s.current == LOD_BASE ? m.rootCount : s.currentCount), ignored);
}

// The traced cuts' groups stay resident while frames in flight may trace them.
[numthreads(64, 1, 1)]
void LodMark(uint3 tid : SV_DispatchThreadID)
{
    const uint3 id = uint3(Item(tid), 0, 0);
    if (id.x >= cClusterWork) return;
    const uint i = FindInstance(clusterWorkOff, id.x);
    const LodInstance inst = instances[i];
    const LodMesh m = meshes[inst.mesh];
    const LodState s = state[i];
    const uint k = id.x - inst.clusterWork;
    if (s.current == LOD_BASE || k >= s.currentCount) return;
    const uint c = cutIdx[inst.cutIdx + s.half * inst.clusterCount + k];
    InterlockedMax(lastUsed[clusters[m.clusterBase + c].x], cFrame);
}

// The partitioned TLAS update operation over the updates the plan wrote.
[numthreads(1, 1, 1)]
void LodFinalize(uint3 id : SV_DispatchThreadID)
{
    const uint count = work.Load(cTlasArgs + 16u * cInstances);
    const uint op = cTlasArgs + 16u * cInstances + 16u;
    const uint64_t args = (uint64_t(cWorkVaHi) << 32 | cWorkVaLo) + cTlasArgs;
    work.Store(op, 1u);                       // D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TYPE_UPDATE_INSTANCE
    work.Store(op + 4u, count);
    work.Store2(op + 8u, uint2(uint(args), uint(args >> 32)));
    work.Store2(op + 16u, uint2(16u, 0u));    // stride
    work.Store(op + 32u, 1u);                 // operation count
    feedback.Store(16, count);
}
