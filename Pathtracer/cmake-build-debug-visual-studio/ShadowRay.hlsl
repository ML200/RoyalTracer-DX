// #DXR Extra - Another ray type

// Ray payload for the shadow rays
struct [[raypayload]] ShadowHitInfo {
  bool isHit: read(caller,anyhit,closesthit,miss)
            : write(caller,anyhit,closesthit,miss);
};


struct Attributes {
  float2 uv;
};

[shader("closesthit")] void ShadowClosestHit(inout ShadowHitInfo hit,
                                             Attributes bary) {
  hit.isHit = true;
}

[shader("miss")] void ShadowMiss(inout ShadowHitInfo hit
                                  : SV_RayPayload) {
  hit.isHit = false;
}