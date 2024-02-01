//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

struct PSInput {
  float4 position : SV_POSITION;
  float4 color : COLOR;
};

// #DXR Extra: Perspective Camera
cbuffer CameraParams : register(b0) {
  float4x4 view;
  float4x4 projection;
}
// #DXR Extra - Refitting
struct InstanceProperties {
  float4x4 objectToWorld;
};

StructuredBuffer<InstanceProperties> instanceProps : register(t0);

uint instanceIndex : register(b1);

PSInput VSMain(float4 position : POSITION, float4 color : COLOR) {
  PSInput result;

  // #DXR Extra: Perspective Camera
  // #DXR Extra - Refitting
  float4 pos = mul(instanceProps[instanceIndex].objectToWorld, position);
  pos = mul(view, pos);
  pos = mul(projection, pos);
  result.position = pos;
  result.color = color;

  return result;
}

float4 PSMain(PSInput input) : SV_TARGET { return input.color; }
