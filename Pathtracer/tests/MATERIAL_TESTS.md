# Material sampling regressions

Run `tests/run_sharc_tests.ps1` from a Visual Studio Developer PowerShell.
The `materialSamplingCheck` entry uses production material functions with
deterministic material and LUT fixtures, independently of the renderer,
radiance cache, ReSTIR and denoiser.

- **GGX distribution and energy:** a white metal with roughness one, viewed
  normally, has outgoing density `1/(4*pi)` on the upper hemisphere. Half its
  microfacet samples must be null events. Its reflected integral under unit
  incident radiance is `(1-ln(2))/Ess`, including the fixture's existing energy
  compensation. Folding rejected directions above the surface doubles this
  integral without changing the reported PDF. Rejection preserves the original
  sampling distribution and consumes no additional random numbers.
- **City floor:** the fixture matches `FloorMat`'s opaque dielectric, quantized
  base color and roughness, with no clearcoat or sheen. Both the mixture
  estimator used by PT and the selected-lobe estimator used by legacy ReSTIR
  are compared with a separate uniform-hemisphere integral. Two view angles
  and a bright band of incident light near the horizon exercise the error's
  dependence on lighting direction. Each case uses 2,097,152 samples.
- **Smooth clearcoat:** 4,096 directions sweep the highlight at roughness
  0.06 and zero. An independent NDF using the sine of the angle provides the
  reference. Half-precision `N.H` produces a broad plateau, and the subtractive
  NDF denominator can collapse to zero. The production NDF instead computes
  the cross-product magnitude in float, and the fused evaluator uses the
  same roughness-to-alpha mapping as the sampler and standalone PDF.

The existing `materialCheck` entry separately checks agreement between fused
and individual lobe evaluation across diffuse, metal, coat, sheen, anisotropic,
and transmitting materials. Agreement between those entry points alone cannot
detect a shared sampling or precision error.

These regressions establish the corrected sampling and clearcoat behavior;
they do not establish that every material approximation is unbiased or that a
particular scene artifact has been reproduced.
