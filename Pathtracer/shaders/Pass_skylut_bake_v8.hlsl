#include "Includes_v8.hlsli"

RWTexture2D<float4> gTransmittanceLUTOut : register(u25);
RWTexture2D<float4> gMultiScatterLUTOut : register(u27);

[numthreads(8, 8, 1)]
void mainTransmittance(uint3 DTid : SV_DispatchThreadID)
{
    const uint2 dims = uint2((uint)SKY_TRANSMITTANCE_LUT_W,
                             (uint)SKY_TRANSMITTANCE_LUT_H);
    if (DTid.x >= dims.x || DTid.y >= dims.y) return;

    float2 uv = (float2(DTid.xy) + 0.5f) / float2(dims);

    float r, mu;
    TransmittanceLutRMuFromUv(uv, r, mu);

    float3 tr = ComputeTransmittanceToTopRMu(r, mu);
    gTransmittanceLUTOut[DTid.xy] = float4(tr, 1.0f);
}

// Bake solar transmittance into the atmosphere lookup table.
float3 BakeSunTransmittance(float3 Q, float3 L)
{
    float t0, t1;
    if (!RaySphereIntersect(Q, L, ATMOS_TOP_RADIUS, t0, t1) || t1 <= 0.0f)
        return float3(1, 1, 1);

    float tG0, tG1;
    float RbBlock = ATMOS_BOTTOM_RADIUS - ATMOS_SUN_BLOCK_BIAS_KM;
    if (RaySphereIntersect(Q, L, RbBlock, tG0, tG1) && tG0 > 0.0f && tG0 < t1)
        return float3(0, 0, 0);

    float r  = clamp(length(Q), ATMOS_BOTTOM_RADIUS, ATMOS_TOP_RADIUS);
    float mu = clamp(dot(Q, L) / max(r, 1e-4f), -1.0f, 1.0f);
    float2 uv = TransmittanceLutUvFromRMu(r, mu);

    float fx = uv.x * SKY_TRANSMITTANCE_LUT_W - 0.5f;
    float fy = uv.y * SKY_TRANSMITTANCE_LUT_H - 0.5f;
    uint  x0 = (uint)clamp(fx, 0.0f, SKY_TRANSMITTANCE_LUT_W - 2.0f);
    uint  y0 = (uint)clamp(fy, 0.0f, SKY_TRANSMITTANCE_LUT_H - 2.0f);
    float wx = saturate(fx - (float)x0);
    float wy = saturate(fy - (float)y0);
    float3 v00 = gTransmittanceLUTOut[uint2(x0,      y0     )].rgb;
    float3 v10 = gTransmittanceLUTOut[uint2(x0 + 1u, y0     )].rgb;
    float3 v01 = gTransmittanceLUTOut[uint2(x0,      y0 + 1u)].rgb;
    float3 v11 = gTransmittanceLUTOut[uint2(x0 + 1u, y0 + 1u)].rgb;
    return lerp(lerp(v00, v10, wx), lerp(v01, v11, wx), wy);
}

// Hillaire 2020 multiple-scattering LUT.

#define SKY_MS_LUT_DIM 32u
#define SKY_MS_DIRS    64u
#define SKY_MS_STEPS   20

inline float3 MsFibonacciDir(uint i, uint n)
{
    float z   = 1.0f - (2.0f * (float)i + 1.0f) / (float)n;
    float s   = sqrt(max(0.0f, 1.0f - z * z));
    float phi = (float)i * 2.39996323f;
    return float3(cos(phi) * s, z, sin(phi) * s);
}

groupshared float3 gs_L2 [SKY_MS_DIRS];
groupshared float3 gs_fms[SKY_MS_DIRS];

[numthreads(64, 1, 1)]
// Integrate multiple scattering into one LUT texel.
void mainMultiScatter(uint3 Gid : SV_GroupID, uint Gi : SV_GroupIndex)
{
    const float Rb = ATMOS_BOTTOM_RADIUS;
    const float Rt = ATMOS_TOP_RADIUS;

    float2 uv      = (float2(Gid.xy) + 0.5f) / (float)SKY_MS_LUT_DIM;
    float sunCosZ,r;
    MultiScatterLutRMuFromUnit(float2(LutUnitRangeFromCoord(uv.x,(float)SKY_MS_LUT_DIM),
        LutUnitRangeFromCoord(uv.y,(float)SKY_MS_LUT_DIM)),r,sunCosZ);
    r = clamp(r, Rb + 1e-3f, Rt - 1e-3f);

    float  sinT = sqrt(saturate(1.0f - sunCosZ * sunCosZ));
    float3 L    = float3(sinT, sunCosZ, 0.0f);
    float3 O    = float3(0.0f, r, 0.0f);

    const float kUniformPhase = 1.0f / (4.0f * PI);

    const uint d = Gi;
    float3 L2  = float3(0, 0, 0);
    float3 fms = float3(0, 0, 0);

    float3 V = MsFibonacciDir(d, SKY_MS_DIRS);
    float tA0, tA1;
    if (RaySphereIntersect(O, V, Rt, tA0, tA1) && tA1 > 0.0f)
    {
        float tMax = tA1;
        float tG0, tG1;
        if (RaySphereIntersect(O, V, Rb, tG0, tG1) && tG0 > 0.0f)
            tMax = min(tMax, tG0);

        if (tMax > 0.0f)
        {
            float3 throughput = float3(1, 1, 1);

            [loop]
            for (int s = 0; s < SKY_MS_STEPS; ++s)
            {

                float u0=float(s)/SKY_MS_STEPS,u1=float(s+1)/SKY_MS_STEPS;
                float closest=clamp(-dot(O,V),0.0f,tMax);
                float t0,t1;
                if(closest<=0.0f) {t0=u0*u0*tMax;t1=u1*u1*tMax;}
                else if(closest>=tMax) {t0=(2*u0-u0*u0)*tMax;t1=(2*u1-u1*u1)*tMax;}
                else {
                    float v0=2*u0-1,v1=2*u1-1;
                    t0=closest+v0*abs(v0)*(v0<0 ? closest:tMax-closest);
                    t1=closest+v1*abs(v1)*(v1<0 ? closest:tMax-closest);
                }
                float ds=t1-t0,t=(t0+t1)*.5f;
                float3 P   = O + V * t;
                float  alt = max(0.0f, length(P) - Rb);

                MediumSample med  = SampleMedium(alt);
                float3 segTr      = exp(-med.extinction * ds);
                float3 sunTr      = BakeSunTransmittance(P, VisibleSunDirection(P,L));

                float3 Pnorm       = SafeNormalize(P);
                float  cosHorizon  = -sqrt(max(0.0f,
                    1.0f - (Rb * Rb) / dot(P, P)));
                float  earthShadow = SunDiskFractionAboveHorizon(dot(Pnorm, L),
                                                                 cosHorizon);

                float3 integ;
                integ.x = (med.extinction.x > 1e-10f)
                    ? (1.0f - segTr.x) / med.extinction.x : ds;
                integ.y = (med.extinction.y > 1e-10f)
                    ? (1.0f - segTr.y) / med.extinction.y : ds;
                integ.z = (med.extinction.z > 1e-10f)
                    ? (1.0f - segTr.z) / med.extinction.z : ds;

                float3 scatterIso = med.scatterR + med.scatterM;

                L2  += throughput * scatterIso * kUniformPhase
                     * (ATMOS_SOLAR_IRRADIANCE * earthShadow * sunTr) * integ;
                fms += throughput * scatterIso * integ;

                throughput *= segTr;
            }
        }
    }

    gs_L2 [d] = L2;
    gs_fms[d] = fms;
    GroupMemoryBarrierWithGroupSync();

    if (d == 0u)
    {
        float3 sumL2  = float3(0, 0, 0);
        float3 sumFms = float3(0, 0, 0);
        [loop]
        for (uint i = 0u; i < SKY_MS_DIRS; ++i)
        {
            sumL2  += gs_L2 [i];
            sumFms += gs_fms[i];
        }
        sumL2  /= (float)SKY_MS_DIRS;
        sumFms /= (float)SKY_MS_DIRS;

        float3 psi = sumL2 / max(1.0f - sumFms, 1e-3f);
        gMultiScatterLUTOut[Gid.xy] = float4(psi, 1.0f);
    }
}
