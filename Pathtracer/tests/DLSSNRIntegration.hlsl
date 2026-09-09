Texture2D<float4> reconstructed : register(t0);
RWTexture2D<float4> displayColor : register(u0);

// Synthetic test input is already in display range. Exercise the actual GPU
// RR -> separate opaque SDR texture -> NR flow without the scene renderer.
[numthreads(8, 8, 1)]
void main(uint2 pixel : SV_DispatchThreadID) {
    uint width, height; displayColor.GetDimensions(width, height);
    if (pixel.x < width && pixel.y < height)
        displayColor[pixel] = float4(saturate(reconstructed[pixel].rgb), 1.0f);
}
