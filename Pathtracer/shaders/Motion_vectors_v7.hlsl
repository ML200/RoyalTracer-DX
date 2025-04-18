/*
This helper class contains methods for calculating efficient motion vectors for the camera as well as object in the scene.
*/

// Compute the motion vector for the current pixels object hit (will me added to the camera motion vector)
float2 ComputeMotionVectorObject(float3 worldHitPos, uint hitInstance){


}

// Compute the motion vector stemming from the camera movement (player etc). This is based on its pure transformation (using projection matrix) -> works for every kind of cam movement out of the box
float2 ComputeMotionVectorCamera(float3 worldHitPos){
    // Convert to clip space in the current frame


    // Convert to clip space in the previous frame


}