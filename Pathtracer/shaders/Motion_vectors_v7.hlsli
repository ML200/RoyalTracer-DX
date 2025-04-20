inline uint MapPixelID(uint2 dims, uint2 lIndex){
    return lIndex.x * dims.y + lIndex.y;
}
