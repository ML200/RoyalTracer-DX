Texture2D<float4> reconstructed : register(t0);
RWTexture2D<float4> displayColor : register(u0);

[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
    uint width, height; displayColor.GetDimensions(width, height);
    if (pixel.x < width && pixel.y < height)
        displayColor[pixel] = float4(saturate(reconstructed[pixel].rgb), 1.0f);
}
