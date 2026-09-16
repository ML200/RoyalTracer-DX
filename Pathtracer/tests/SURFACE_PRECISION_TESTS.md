# Surface position precision regression

Run from an x64 Visual Studio Developer PowerShell:

```powershell
./tests/run_surface_precision_tests.ps1
```

Requires a GPU supporting DXR 1.1. The runner builds the two 500-by-500 city floor
triangles through the production BLAS/TLAS helpers, then launches 262,144 primary
rays from each of eight camera heights, on both sides of the floor.

For each hit, it compares the ray-distance position with the barycentrically
interpolated triangle position used by `EvalSurfaceState`. Both positions are
offset using the production `offset_ray`, then tested with an outgoing ray that
points away from the floor. Every primary ray must hit, the interpolated position
must remain on the floor, and no outgoing ray from that position may hit it.

The ray-distance version is diagnostic: its error and false-hit count are printed,
but are not required to fail on every GPU. On the tested GPU, it produces repeated
stripes and triangular regions of self-intersections. The interpolated position
produces zero false hits in all eight cases.

This fixture isolates a real position-reconstruction error. It does not establish
that every reported floor artifact has the same cause, or test arbitrary instance
transforms and normal maps.
