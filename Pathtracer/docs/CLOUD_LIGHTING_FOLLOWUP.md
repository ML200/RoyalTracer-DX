# Cloud lighting follow-up

This note records the remaining cloud lighting work.

## Current limitation

The cloud path evaluates density, direct sunlight, cached broad illumination,
and atmospheric transmission independently at many march points. The cache
reduces repeated work for reused directions, but it does not transport light
between separate cloud bodies. The measured 1080p cost is therefore dominated
by cloud rays rather than cache updates.

## Direction

Keep the existing density field and deterministic extinction guides while
testing a bounded broad-light field. Store low-frequency irradiance and optical
depth in the existing cache layout, update it from a fixed number of samples,
and use the field only for indirect cloud fill. Direct sunlight and local
shadow correction continue to use the existing estimator.

Do not add a nested light march to every density sample. Compare any new field
against the current path with identical density, exposure, phase function,
camera motion, wind, and RR guides. Measure cache update, primary, secondary,
and guide timings separately.

## Acceptance

The experiment must preserve nonnegative transmittance, stable motion guides,
and correct zero-density behavior. Validate thin wisps, dense interiors,
overlapping layers, horizon views, sun-facing sunsets, and camera rebases.
Reject the change if it shifts the mean radiance or guide ownership without a
measured quality benefit at the same frame budget.
