#pragma once
#include <DirectXMath.h>
using namespace DirectX;

struct Transform {
    XMFLOAT3 position = { 0, 0, 0 };
    XMFLOAT3 rotation = { 0, 0, 0 };
    XMFLOAT3 scale    = { 1, 1, 1 };

    XMMATRIX GetMatrix() const {
        return XMMatrixScaling(scale.x, scale.y, scale.z)
             * XMMatrixRotationRollPitchYaw(
                   XMConvertToRadians(rotation.x),
                   XMConvertToRadians(rotation.y),
                   XMConvertToRadians(rotation.z))
             * XMMatrixTranslation(position.x, position.y, position.z);
    }
};
