// Ocean surface simulation. Every entry point here runs on the streaming compute queue before the
// frame traces against the ocean, and reaches its resources through ResourceDescriptorHeap.
//
// Pipeline, once per frame:
//   Evolve   spectrum(t) -> four packed complex fields per cascade
//   FftH/FftV  inverse 2D FFT of those fields
//   Assemble   -> displacement (Dx,Dy,Dz,jacobian) and slope (dy/dx, dy/dz)
//   Foam       deformation diagnostics (legacy entry name; foam removed)
//   Mip        box pyramid, so the shading pass can filter by ray footprint
//   Tiles      quadtree leaves -> displaced vertices for the acceleration structures
//
// h0 and the wave vectors are baked on the CPU whenever the sea state changes: the same samples
// produce the per-cascade slope variances the BRDF uses, so the two cannot drift apart.

#include "OceanLayout.h"
#include "OceanMath.hlsli"

cbuffer OceanPush : register(b0) {
    uint gU0;
    uint gU1;
    uint gU2;
    uint gU3;
    float gTime;
    float gDt;
    float gF2;
    float gF3;
};

SamplerState g_oceanSampler : register(s0);

static const float OCEAN_PI = 3.14159265358979f;

float2 CMul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------------------------
// Evolve: h(k,t) = h0(k) e^{-i w t} + conj(h0(-k)) e^{i w t}, then the eight derived real fields
// packed two per complex transform. Each field is Hermitian, so the pair separates exactly into
// the real and imaginary parts of one inverse transform.
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void OceanEvolve(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OCEAN_FFT_SIZE || tid.y >= OCEAN_FFT_SIZE)
        return;

    Texture2DArray<float4> h0Tex = ResourceDescriptorHeap[OCEAN_SRV_H0];
    Texture2DArray<float4> waveTex = ResourceDescriptorHeap[OCEAN_SRV_WAVE];
    RWTexture2DArray<float4> fft = ResourceDescriptorHeap[OCEAN_UAV_FFT];
    StructuredBuffer<OceanParamsGPU> params = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];

    const uint c = tid.z;
    const uint3 at = uint3(tid.xy, c);

    const float4 h0 = h0Tex[at];
    const float4 wave = waveTex[at]; // (kx, kz, 1/k, omega)

    const float phase = -wave.w * gTime;
    float s, cs;
    sincos(phase, s, cs);
    const float2 ePlus = float2(cs, s);
    const float2 eMinus = float2(cs, -s);

    const float2 h = CMul(h0.xy, ePlus) + CMul(h0.zw, eMinus);

    const float kx = wave.x;
    const float kz = wave.y;
    const float invK = wave.z;
    // With our positive-exponent inverse FFT, negative lambda compresses crests.
    // Positive lambda sharpens troughs and deposits whitecaps in the wave valleys.
    const float lambda = -params[0].choppiness * OceanChopGain(length(wave.xy));

    // Vertical displacement and its gradient.
    const float2 specY = h;
    const float2 specDYdx = float2(-h.y * kx, h.x * kx);
    const float2 specDYdz = float2(-h.y * kz, h.x * kz);

    // Horizontal displacement: -i (k/|k|) h, scaled by the choppiness.
    const float tx = kx * invK * lambda;
    const float tz = kz * invK * lambda;
    const float2 specX = float2(h.y * tx, -h.x * tx);
    const float2 specZ = float2(h.y * tz, -h.x * tz);

    // Horizontal displacement gradients, for the folding jacobian. Multiplying by i k twice turns
    // the leading -i into a real factor, so these stay real-valued spectra.
    const float2 specDXdx = h * (kx * kx * invK * lambda);
    const float2 specDZdz = h * (kz * kz * invK * lambda);
    const float2 specDXdz = h * (kx * kz * invK * lambda);

    // Pack: field A goes to the real part, field B to the imaginary part of one transform.
    const float2 f0 = float2(specX.x - specZ.y, specX.y + specZ.x);
    const float2 f1 = float2(specY.x - specDXdz.y, specY.y + specDXdz.x);
    const float2 f2 = float2(specDYdx.x - specDYdz.y, specDYdx.y + specDYdz.x);
    const float2 f3 = float2(specDXdx.x - specDZdz.y, specDXdx.y + specDZdz.x);

    fft[uint3(tid.xy, c * 2u + 0u)] = float4(f0, f1);
    fft[uint3(tid.xy, c * 2u + 1u)] = float4(f2, f3);
}

// ---------------------------------------------------------------------------------------------
// Inverse FFT. One thread group per line; the whole line lives in LDS across all eight stages.
// A float4 carries two independent complex lanes, so one pass transforms two packed fields.
// ---------------------------------------------------------------------------------------------

groupshared float4 g_line[OCEAN_FFT_SIZE];

uint BitReverseN(uint v) {
    v = ((v & 0x5555u) << 1) | ((v & 0xAAAAu) >> 1);
    v = ((v & 0x3333u) << 2) | ((v & 0xCCCCu) >> 2);
    v = ((v & 0x0F0Fu) << 4) | ((v & 0xF0F0u) >> 4);
    v = ((v & 0x00FFu) << 8) | ((v & 0xFF00u) >> 8);
    return v >> (16u - OCEAN_FFT_LOG2);
}

// In-place Cooley-Tukey on g_line. Positive exponent: this is the synthesis direction, and the
// 1/N normalisation is deliberately left out because the baked amplitudes already carry it.
void FftLine(uint tid) {
    [unroll]
    for (uint stage = 1u; stage <= OCEAN_FFT_LOG2; ++stage) {
        const uint len = 1u << stage;
        const uint halfLen = len >> 1u;
        const uint grp = tid / halfLen;
        const uint j = tid - grp * halfLen;
        const uint i0 = grp * len + j;
        const uint i1 = i0 + halfLen;

        const float angle = 2.0f * OCEAN_PI * (float)j / (float)len;
        float sa, ca;
        sincos(angle, sa, ca);
        const float2 w = float2(ca, sa);

        const float4 a = g_line[i0];
        const float4 b = g_line[i1];
        const float4 bw = float4(CMul(b.xy, w), CMul(b.zw, w));
        // Every thread must finish reading its pair before any of them overwrites a slot.
        GroupMemoryBarrierWithGroupSync();
        g_line[i0] = a + bw;
        g_line[i1] = a - bw;
        GroupMemoryBarrierWithGroupSync();
    }
}

[numthreads(OCEAN_FFT_THREADS, 1, 1)]
void OceanFftH(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    RWTexture2DArray<float4> fft = ResourceDescriptorHeap[OCEAN_UAV_FFT];
    const uint row = gid.x;
    const uint slice = gid.y;

    // Load bit-reversed so the in-place butterflies come out in natural order.
    g_line[BitReverseN(tid)] = fft[uint3(tid, row, slice)];
    g_line[BitReverseN(tid + OCEAN_FFT_THREADS)] = fft[uint3(tid + OCEAN_FFT_THREADS, row, slice)];
    GroupMemoryBarrierWithGroupSync();

    FftLine(tid);

    fft[uint3(tid, row, slice)] = g_line[tid];
    fft[uint3(tid + OCEAN_FFT_THREADS, row, slice)] = g_line[tid + OCEAN_FFT_THREADS];
}

[numthreads(OCEAN_FFT_THREADS, 1, 1)]
void OceanFftV(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    RWTexture2DArray<float4> fft = ResourceDescriptorHeap[OCEAN_UAV_FFT];
    const uint col = gid.x;
    const uint slice = gid.y;

    g_line[BitReverseN(tid)] = fft[uint3(col, tid, slice)];
    g_line[BitReverseN(tid + OCEAN_FFT_THREADS)] = fft[uint3(col, tid + OCEAN_FFT_THREADS, slice)];
    GroupMemoryBarrierWithGroupSync();

    FftLine(tid);

    fft[uint3(col, tid, slice)] = g_line[tid];
    fft[uint3(col, tid + OCEAN_FFT_THREADS, slice)] = g_line[tid + OCEAN_FFT_THREADS];
}

// ---------------------------------------------------------------------------------------------
// Assemble: unpack the transforms into displacement and slope. The (-1)^(x+z) factor recentres
// the spectrum, whose wave vectors are stored with the zero frequency in the middle of the grid.
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void OceanAssemble(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OCEAN_FFT_SIZE || tid.y >= OCEAN_FFT_SIZE)
        return;

    RWTexture2DArray<float4> fft = ResourceDescriptorHeap[OCEAN_UAV_FFT];
    RWTexture2DArray<float4> disp = ResourceDescriptorHeap[OCEAN_UAV_DISP_MIPS];
    RWTexture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_UAV_DERIV_MIPS];
    RWTexture2DArray<float4> moments = ResourceDescriptorHeap[OCEAN_UAV_MOMENT_MIPS];
    StructuredBuffer<OceanParamsGPU> params = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];

    const uint c = tid.z;
    const float sign = (((tid.x + tid.y) & 1u) != 0u) ? -1.0f : 1.0f;

    const float4 p0 = fft[uint3(tid.xy, c * 2u + 0u)];
    const float4 p1 = fft[uint3(tid.xy, c * 2u + 1u)];

    const float dx = p0.x * sign;
    const float dz = p0.y * sign;
    const float dy = p0.z * sign;
    const float dXdz = p0.w * sign;
    const float dYdx = p1.x * sign;
    const float dYdz = p1.y * sign;
    const float dXdx = p1.z * sign;
    const float dZdz = p1.w * sign;

    const float scale = params[0].displacementScale;

    // Store linear derivative fields. Inverting each cascade before summation is incorrect.
    disp[uint3(tid.xy, c)] = float4(dx * scale, dy * scale, dz * scale, dXdz);
    deriv[uint3(tid.xy, c)] = float4(dYdx, dYdz, dXdx, dZdz);
    // Bound the bilinear displacement map in this cell, including its between-node derivatives.
    // Each derivative column interpolates between two edge secants; the Frobenius norm bounds
    // its operator norm. Also include the analytic strain used for shading.
    const uint2 p10 = (tid.xy + uint2(1,0)) % OCEAN_FFT_SIZE;
    const uint2 p01 = (tid.xy + uint2(0,1)) % OCEAN_FFT_SIZE;
    const uint2 p11 = (tid.xy + 1u) % OCEAN_FFT_SIZE;
    const float2 a = p0.xy * sign;
    const float2 b = fft[uint3(p10,c*2)].xy * -sign;
    const float2 d = fft[uint3(p01,c*2)].xy * -sign;
    const float2 e = fft[uint3(p11,c*2)].xy * sign;
    const float invStep = OCEAN_FFT_SIZE / params[0].cascadeLength[c];
    const float secantBound = sqrt(max(dot(b-a,b-a),dot(e-d,e-d)) + max(dot(d-a,d-a),dot(e-b,e-b))) * invStep;
    const float strainBound = max(secantBound, sqrt(dXdx*dXdx+dZdz*dZdz+2*dXdz*dXdz));
    moments[uint3(tid.xy, c)] = float4(dYdx*dYdx, dYdx*dYdz, dYdz*dYdz, strainBound);
}

float OceanConditioningGain() {
    RWTexture2DArray<float4> bounds = ResourceDescriptorHeap[OCEAN_UAV_MOMENT_MIPS + OCEAN_MIP_LEVELS - 1];
    float sum = 0.0f;
    [unroll] for (uint c=0;c<OCEAN_CASCADES;++c) sum += bounds[uint3(0,0,c)].w;
    // The share of the no-fold threshold this field may spend. The kilometre-scale sea state
    // multiplies the horizontal displacement after this point, so the host has already divided
    // the budget by the roughest patch's gain - a patch that gets the full boost still cannot
    // fold the surface into itself.
    StructuredBuffer<OceanParamsGPU> params = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    const float budget = max(params[0].deformationBudget, 1e-3f);
    return min(1.0f, budget / max(sum, 1e-8f));
}

// One spatially uniform gain preserves phase and continuity; no per-vertex fold clipping.
[numthreads(8,8,1)]
void OceanCondition(uint3 tid : SV_DispatchThreadID) {
    if (any(tid.xy >= OCEAN_FFT_SIZE)) return;
    RWTexture2DArray<float4> disp = ResourceDescriptorHeap[OCEAN_UAV_DISP_MIPS];
    RWTexture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_UAV_DERIV_MIPS];
    const float gain = OceanConditioningGain();
    float4 d=disp[tid], g=deriv[tid];
    d.xzw *= gain; g.zw *= gain;
    disp[tid]=d; deriv[tid]=g;
}

// ---------------------------------------------------------------------------------------------
// Surface diagnostics. Keep the entry-point name and descriptor layout compatible with the
// existing host/test harness, but generate no foam and read no foam history.
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void OceanFoam(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= OCEAN_FFT_SIZE || tid.y >= OCEAN_FFT_SIZE)
        return;
    RWTexture2DArray<float4> disp = ResourceDescriptorHeap[OCEAN_UAV_DISP_MIPS];
    RWTexture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_UAV_DERIV_MIPS];
    RWTexture2DArray<float4> surface = ResourceDescriptorHeap[OCEAN_UAV_SURFACE_MIPS];
    const float4 derivatives = deriv[tid];
    const float crossDerivative = disp[tid].w;
    const float3 stretch = float3(1.0f + derivatives.z, 1.0f + derivatives.w, crossDerivative);
    const float jacobian = stretch.x * stretch.y - crossDerivative * crossDerivative;
    const float minimumStretch = 0.5f * (stretch.x + stretch.y - length(float2(stretch.x-stretch.y, 2.0f*stretch.z)));
    surface[tid] = float4(0.0f, minimumStretch, 0.0f, jacobian);
}

// ---------------------------------------------------------------------------------------------
// Mip pyramids support geometry LOD, conditioning and diagnostics. Beauty shading samples
// full-resolution derivatives and does not convert filtered variance into BRDF roughness.
// ---------------------------------------------------------------------------------------------
[numthreads(8, 8, 1)]
void OceanMip(uint3 tid : SV_DispatchThreadID) {
    const uint dstSize = max(1u, (uint)OCEAN_FFT_SIZE >> gU1);
    if (tid.x >= dstSize || tid.y >= dstSize)
        return;

    // gU0: displacement, derivative, raw moments, surface; gU1: destination mip.
    const uint pyramidBase = gU0 == 0u ? OCEAN_UAV_DISP_MIPS : (gU0 == 1u ? OCEAN_UAV_DERIV_MIPS :
        (gU0 == 2u ? OCEAN_UAV_MOMENT_MIPS : OCEAN_UAV_SURFACE_MIPS));
    const uint srcSlot = pyramidBase + gU1 - 1u;
    const uint dstSlot = pyramidBase + gU1;

    RWTexture2DArray<float4> src = ResourceDescriptorHeap[srcSlot];
    RWTexture2DArray<float4> dst = ResourceDescriptorHeap[dstSlot];

    const uint2 s = tid.xy * 2u;
    const uint c = tid.z;
    const float4 v = src[uint3(s + uint2(0, 0), c)] + src[uint3(s + uint2(1, 0), c)] +
                     src[uint3(s + uint2(0, 1), c)] + src[uint3(s + uint2(1, 1), c)];
    float4 avg = v * 0.25f;
    if (gU0 == 2u) avg.w = max(max(src[uint3(s,c)].w,src[uint3(s+uint2(1,0),c)].w),
                              max(src[uint3(s+uint2(0,1),c)].w,src[uint3(s+1u,c)].w));
    dst[uint3(tid.xy, c)] = avg;

    // The final level of the surface pyramid is the average over the entire cascade, so its
    // foam channel is that cascade's whitecap coverage. Publishing it here costs one store and
    // saves a separate reduction.
    if (gU0 == 3u && dstSize == 1u) {
        RWStructuredBuffer<float4> stats = ResourceDescriptorHeap[OCEAN_UAV_STATS];
        stats[c] = float4(avg.xyz, OceanConditioningGain());
    }
}
