#ifndef CUMULUS_GUIDE_MATH_V8
#define CUMULUS_GUIDE_MATH_V8
float2 CumulusMotion(float3 P,float3 previousP)
{
    float2 now=GetCurrentFramePixelCoordinates_World(P,view,projection,float2(IMG_W,IMG_H));
    float2 old=GetLastFramePixelCoordinates_World(previousP,prevView,prevProjection,float2(IMG_W,IMG_H));
    return now.x>-1e8f && old.x>-1e8f ? old-now : float2(0,0);
}
#endif
