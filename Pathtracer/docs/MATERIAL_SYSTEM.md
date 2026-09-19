# Material system after the performance rollback

The renderer uses its original lobe samplers, probabilities, broad/specular split,
SSS random walk, transmission, visibility, integrators and replay. The overhaul's
medium stack, broad multiple-scattering lobes, specular-priority setting and large
energy atlas have been removed.

Two shading changes remain:

- Diffuse uses energy-preserving Oren–Nayar (EON). Its analytic multiple-scattering
  term keeps white diffuse at unit directional albedo. A diffuse roughness of zero
  gives Lambertian diffuse. The original cosine proposal and PDF remain valid.
- Lower lobes receive the energy remaining after the upper lobe's compensated
  reflection, rather than the previous Fresnel attenuation heuristic. Existing
  GGX and coat multiple-scattering gains remain. Sheen retains its original
  directional-albedo attenuation.

The original two 16 × 16 LUT slices now contain RGBA floats: 8 KB total. The GGX
slice stores integrals of unit-Fresnel reflection weighted by `1`, `(1 - V.H)^5`
and `(1 - V.H)^10`. These moments account for the existing Schlick Fresnel and
coat compensation without IOR, anisotropy or diffuse coupling atlases. Sampling
maps endpoint values to texel centers. LUT generation uses a fixed seed.

The GPU material remains 48 bytes and the sampling state remains four floats.
Diffuse roughness uses the five previously spare bits 19–23 of packed word 6;
texture indices, thin-glass/SSS flags and SSS weight retain their existing bits.
The inspector exposes diffuse roughness independently of specular roughness.

This retains the legacy approximation for anisotropic GGX energy and transmission.
The moments describe isotropic reflection with Schlick Fresnel. For an interface
that can exhibit TIR, attenuation conservatively reserves the full reflected-energy
bound. Layer attenuation depends on the outgoing direction, as in the legacy sheen
and multiple-scattering implementation; this is not a new reciprocal layered BSDF.

Verification: the GPU tests cover fused evaluation and PDF consistency, cosine/VNDF
sampling, coat highlight precision, and independently integrated white furnaces.
For measured isotropic GGX moments, six view/roughness combinations, three diffuse
roughnesses, and GGX/coat/sheen stacks have a maximum furnace error of about 0.12%.
Pure EON's maximum error is about 0.10%. These checks isolate the shading formulas;
they do not certify interpolation accuracy of the coarse LUT or legacy volume
transport. Scene loading, instancing/BVH regressions and the Release build pass.

Minecraft leaves and thin plant materials load with the foliage SSS preset:
weight 0.65, radius one block, phase anisotropy 0.35 and IOR 1.4. The bright
scattering albedo follows the tinted texture average, preserving flower colours.
Recognized foliage includes grasses, flowers, saplings, vines, crops, roots,
fungi, thin coral and aquatic plants; kelp and seagrass now resolve their models
instead of being skipped. Foliage LOD and flat variants keep the preset. Within
potted models, ceramic and soil retain ordinary materials; coarse projections
that combine pot and plant also use ordinary materials. These presets use the
existing volume random walk, whose thin/open-geometry behaviour remains a legacy
approximation. Plain glass and panes use neutral `(1, 1, 1)` transmission.
