// FFT ocean simulation (Tessendorf 2001). h0 is baked on the CPU.

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

static const float OCEAN_PI = 3.14159265358979f;
static const float OCEAN_GRAVITY = 9.80665f;

float2 CMul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Two float4 lines: a cascade's four packed complex fields.
groupshared float4 g_line[2][OCEAN_FFT_SIZE];

uint BitReverseN(uint v) {
    v = ((v & 0x5555u) << 1) | ((v & 0xAAAAu) >> 1);
    v = ((v & 0x3333u) << 2) | ((v & 0xCCCCu) >> 2);
    v = ((v & 0x0F0Fu) << 4) | ((v & 0xF0F0u) >> 4);
    v = ((v & 0x00FFu) << 8) | ((v & 0xFF00u) >> 8);
    return v >> (16u - OCEAN_FFT_LOG2);
}

// In-place Cooley-Tukey, positive exponent; no 1/N (baked into h0).
void FftLines(uint tid) {
    [unroll]
    for (uint stage = 1u; stage <= OCEAN_FFT_LOG2; ++stage) {
        const uint len = 1u << stage;
        const uint halfLen = len >> 1u;
        const uint grp = tid / halfLen;
        const uint j = tid - grp * halfLen;
        const uint i0 = grp * len + j;
        const uint i1 = i0 + halfLen;

        float sa, ca;
        sincos(2.0f * OCEAN_PI * (float)j / (float)len, sa, ca);
        const float2 w = float2(ca, sa);

        const float4 a0 = g_line[0][i0], b0 = g_line[0][i1];
        const float4 a1 = g_line[1][i0], b1 = g_line[1][i1];
        const float4 bw0 = float4(CMul(b0.xy, w), CMul(b0.zw, w));
        const float4 bw1 = float4(CMul(b1.xy, w), CMul(b1.zw, w));
        // All reads before any in-place write.
        GroupMemoryBarrierWithGroupSync();
        g_line[0][i0] = a0 + bw0;
        g_line[0][i1] = a0 - bw0;
        g_line[1][i0] = a1 + bw1;
        g_line[1][i1] = a1 - bw1;
        GroupMemoryBarrierWithGroupSync();
    }
}

// h(k,t) = h0(k) e^{-i w t} + conj(h0(-k)) e^{i w t} (Tessendorf 2001); two real fields per transform.
void EvolveMode(uint2 at, uint c, OceanParamsGPU P, out float4 slice0, out float4 slice1) {
    Texture2DArray<float4> h0Tex = ResourceDescriptorHeap[OCEAN_SRV_H0];
    const float4 h0 = h0Tex[uint3(at, c)];

    const float dk = 2.0f * OCEAN_PI / P.cascadeLength[c];
    const float kx = ((float)at.x - (float)(OCEAN_FFT_SIZE / 2)) * dk;
    const float kz = ((float)at.y - (float)(OCEAN_FFT_SIZE / 2)) * dk;
    const float k = sqrt(kx * kx + kz * kz);
    const float invK = k > 0.0f ? rcp(k) : 0.0f;
    const float omega = sqrt(OCEAN_GRAVITY * k);

    float s, cs;
    sincos(-omega * gTime, s, cs);
    const float2 h = CMul(h0.xy, float2(cs, s)) + CMul(h0.zw, float2(cs, -s));

    // With our positive-exponent inverse FFT, negative lambda compresses crests.
    const float lambda = -P.choppiness * OceanChopGain(k, P.chopBand);

    const float2 specY = h;
    const float2 specDYdx = float2(-h.y * kx, h.x * kx);
    const float2 specDYdz = float2(-h.y * kz, h.x * kz);

    // Choppy displacement -i (k/|k|) h (Tessendorf 2001).
    const float tx = kx * invK * lambda;
    const float tz = kz * invK * lambda;
    const float2 specX = float2(h.y * tx, -h.x * tx);
    const float2 specZ = float2(h.y * tz, -h.x * tz);

    // Horizontal displacement gradients: real multiples of h.
    const float2 specDXdx = h * (kx * kx * invK * lambda);
    const float2 specDZdz = h * (kz * kz * invK * lambda);
    const float2 specDXdz = h * (kx * kz * invK * lambda);

    // Pack A + iB: A to the real part, B to the imaginary.
    slice0 = float4(specX.x - specZ.y, specX.y + specZ.x, specY.x - specDXdz.y, specY.y + specDXdz.x);
    slice1 = float4(specDYdx.x - specDYdz.y, specDYdx.y + specDYdz.x, specDXdx.x - specDZdz.y,
                    specDXdx.y + specDZdz.x);
}

// One group per (row, cascade).
[numthreads(OCEAN_FFT_THREADS, 1, 1)]
void OceanFftH(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    RWTexture2DArray<float4> fft = ResourceDescriptorHeap[OCEAN_UAV_FFT];
    StructuredBuffer<OceanParamsGPU> params = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    const OceanParamsGPU P = params[0];
    const uint row = gid.x;
    const uint c = gid.y;

    [unroll] for (uint half_ = 0u; half_ < 2u; ++half_) {
        const uint x = tid + half_ * OCEAN_FFT_THREADS;
        float4 s0, s1;
        EvolveMode(uint2(x, row), c, P, s0, s1);
        // Bit-reversed load, natural-order result.
        g_line[0][BitReverseN(x)] = s0;
        g_line[1][BitReverseN(x)] = s1;
    }
    GroupMemoryBarrierWithGroupSync();

    FftLines(tid);

    [unroll] for (uint h = 0u; h < 2u; ++h) {
        const uint x = tid + h * OCEAN_FFT_THREADS;
        fft[uint3(x, row, c * 2u + 0u)] = g_line[0][x];
        fft[uint3(x, row, c * 2u + 1u)] = g_line[1][x];
    }
}

// One group per (column, cascade). (-1)^(x+z): zero frequency is stored mid-grid.
[numthreads(OCEAN_FFT_THREADS, 1, 1)]
void OceanFftV(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    RWTexture2DArray<float4> fft = ResourceDescriptorHeap[OCEAN_UAV_FFT];
    StructuredBuffer<OceanParamsGPU> params = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    const OceanParamsGPU P = params[0];
    const uint parity = P.dispParity & 1u;
    RWTexture2DArray<float4> disp = ResourceDescriptorHeap[parity == 0u ? OCEAN_UAV_DISP0_MIPS : OCEAN_UAV_DISP1_MIPS];
    RWTexture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_UAV_DERIV_MIPS];
    const uint col = gid.x;
    const uint c = gid.y;

    [unroll] for (uint half_ = 0u; half_ < 2u; ++half_) {
        const uint y = tid + half_ * OCEAN_FFT_THREADS;
        g_line[0][BitReverseN(y)] = fft[uint3(col, y, c * 2u + 0u)];
        g_line[1][BitReverseN(y)] = fft[uint3(col, y, c * 2u + 1u)];
    }
    GroupMemoryBarrierWithGroupSync();

    FftLines(tid);

    [unroll] for (uint h = 0u; h < 2u; ++h) {
        const uint y = tid + h * OCEAN_FFT_THREADS;
        const float sign = (((col + y) & 1u) != 0u) ? -1.0f : 1.0f;
        const float4 p0 = g_line[0][y] * sign; // (Dx, Dz, Dy, dDx/dz)
        const float4 p1 = g_line[1][y] * sign; // (dy/dx, dy/dz, dDx/dx, dDz/dz)
        disp[uint3(col, y, c)] = float4(p0.x, p0.z, p0.y, p0.w);
        deriv[uint3(col, y, c)] = p1;
    }
}

// One mip of every pyramid. gU0: displacement parity; gU1: destination mip.
[numthreads(8, 8, 1)]
void OceanMip(uint3 tid : SV_DispatchThreadID) {
    const uint dstSize = max(1u, (uint)OCEAN_FFT_SIZE >> gU1);
    if (tid.x >= dstSize || tid.y >= dstSize)
        return;

    const uint dispBase = (gU0 & 1u) == 0u ? OCEAN_UAV_DISP0_MIPS : OCEAN_UAV_DISP1_MIPS;
    RWTexture2DArray<float4> dispSrc = ResourceDescriptorHeap[dispBase + gU1 - 1u];
    RWTexture2DArray<float4> dispDst = ResourceDescriptorHeap[dispBase + gU1];
    RWTexture2DArray<float4> derivSrc = ResourceDescriptorHeap[OCEAN_UAV_DERIV_MIPS + gU1 - 1u];
    RWTexture2DArray<float4> derivDst = ResourceDescriptorHeap[OCEAN_UAV_DERIV_MIPS + gU1];

    const uint2 s = tid.xy * 2u;
    const uint c = tid.z;
    dispDst[uint3(tid.xy, c)] = 0.25f * (dispSrc[uint3(s, c)] + dispSrc[uint3(s + uint2(1, 0), c)] +
                                         dispSrc[uint3(s + uint2(0, 1), c)] + dispSrc[uint3(s + 1u, c)]);
    derivDst[uint3(tid.xy, c)] = 0.25f * (derivSrc[uint3(s, c)] + derivSrc[uint3(s + uint2(1, 0), c)] +
                                          derivSrc[uint3(s + uint2(0, 1), c)] + derivSrc[uint3(s + 1u, c)]);
}
