# Cumulus performance

The current cloud path preserves the accepted density field and deterministic
RR guides. Optimizations target repeated material work and bounded lighting
cost; they must keep density, guides, and transport consistent.

## Current implementation

- Two RG8 noise planes replace the original RGBA8 field without changing the
  procedural inputs or total 64 MiB noise storage.
- Cloud lighting uses one to four weighted source samples after each shell
  crossing; two is the default. Inverse-probability weighting preserves the
  discrete source sum in expectation.
- Atmospheric air scattering shares the cloud march's extinction and uses the
  existing near/far shadow cache for direct cloud shadows.
- Broad ambient fill uses upper and lower atmospheric hemispheres, while the
  detailed direct term retains local shadows and the current phase function.

## Measurements

At 1920×1080 on the configured RTX 5090, the accepted lighting revision measures
about 5.2 ms for primary cloud work and 0.7–1.0 ms for cache updates, depending
on sun elevation. The complete 3 ms frame target remains unmet; cloud rays are
the dominant cost. These are headless shader timings and exclude RR,
presentation, and scene tracing.

The density optimization skips filtered noise only when a conservative upper
bound proves the remaining field empty. Surviving samples execute the original
material arithmetic. The parity oracle and production shader suite require
bit-identical density, opacity, guides, motion, and transmittance for the
covered cases.

## Next candidates

1. Add conservative empty-interval traversal shared by radiance, guides, and
   shadow queries.
2. Replace six full material evaluations used for normals with validated
   derivatives of the existing density function.
3. Bound procedural shadow fallbacks for horizon rays without changing the
   near-field penetration model.
4. Measure rough and diffuse secondary rays using their actual angular
   footprints before reducing their detail.

Every candidate needs matched-exposure comparisons for ground, horizon,
above-cloud, interior, wind, twilight, and rebased-camera views. Current suite
commands and artifacts are listed in `CUMULUS_SCENE.md`.
