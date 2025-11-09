#include "Includes_v7.hlsli"

static const float3 kLUMA = float3(0.2126, 0.7152, 0.0722);

#define uThreshold 0.5f

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight)
        return;

    uint2 launch = DTid.xy;

    if (gScratchPing[uint3(launch, 3)].x == 1) {
        gScratchPing[uint3(launch, 1)] = gScratchPing[uint3(launch, 2)];
        return;
    }

    float w        = gScratchPing[uint3(launch, 5)].w;
    float threshold = lerp(1.0f, uThreshold, saturate(w / 5.0f));

    float3 centerRGB  = gScratchPing[uint3(launch, 0)].rgb;
    float3 centerN    = normalize(gScratchPing[uint3(launch, 4)].xyz);
    float3 centerPos  = gScratchPing[uint3(launch, 5)].xyz;
    float  rough      = gScratchPing[uint3(launch, 3)].y;
    uint   centerObj  = asuint(gScratchPing[uint3(launch, 3)].z);

    float cosThresh   = lerp(0.92f, 0.85f, saturate(rough));
    float distThresh  = 0.05f;

    float3 neighbourSum = 0.0;
    uint   neighbourCnt = 0;

    [unroll]
    for (int dy = -1; dy <= 1; ++dy)
    {
        [unroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            if (dx == 0 && dy == 0) continue;

            int2 n = int2(launch) + int2(dx, dy);
            if (n.x < 0 || n.y < 0 || n.x >= int(gImageWidth) || n.y >= int(gImageHeight))
                continue;

            float3 nRGB   = gScratchPing[uint3(n, 0)].rgb;
            float3 nN     = normalize(gScratchPing[uint3(n, 4)].xyz);
            float3 nPos   = gScratchPing[uint3(n, 5)].xyz;
            uint   nObj   = asuint(gScratchPing[uint3(n, 3)].z);

            // geometry checks
            bool sameObj     = (nObj == centerObj);
            bool normalOK    = !RejectNormal_GI(centerN, nN, cosThresh);
            bool distanceOK  = !RejectDistance_GI(centerPos, nPos, centerN, distThresh);

            if (sameObj && normalOK && distanceOK)
            {
                neighbourSum += nRGB;
                ++neighbourCnt;
            }
        }
    }

    // Reject filter on no good neighbors
    if (neighbourCnt > 0)
    {
        float3 neighbourAvg = neighbourSum / float(neighbourCnt);

        float centerLum    = Luma(centerRGB);
        float neighbourLum = Luma(neighbourAvg);

        if (centerLum > neighbourLum * (1.0 + threshold))
        {
            centerRGB = neighbourAvg;
        }
    }

    gScratchPing[uint3(launch, 1)] = float4(centerRGB, 1.0);
}
