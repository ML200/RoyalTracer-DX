# Recovered slow cloud reference

These nine source files preserve the cloud implementation preceding the rejected
Frostbite/HanPi performance rewrite. Keep them unchanged when optimizing production.

The sources were recovered from this task's original implementation and analysis
outputs. The noise bake was reconstructed from those outputs and its unchanged
helpers. A fresh build reproduced the previously retained slow shader binaries
exactly: primary radiance, opacity and guides matched at 1920×1080, including dawn,
dusk, above-cloud and orbital views. This is the version associated with the user's
roughly 1.1 ms cache plus 11 ms cloud-ray measurement.

From a Visual Studio Developer PowerShell, in the project root:

```powershell
./tests/run_cumulus_tests.ps1 -Width 1920 -Height 1080 -OutputDirectory out/cumulus/reference1080 -Reference
./tests/run_cumulus_tests.ps1 -Width 1920 -Height 1080 -OutputDirectory out/cumulus/restored-final1080
python ./tests/compare_cumulus_reference.py out/cumulus/reference1080 out/cumulus/restored-final1080
```

Reference mode stages the current shared engine headers and overlays these sources.
It restores the single lighting cascade, two environment layers and 48 reflection
samples. Both modes use 64 primary samples and the same camera, weather, sun and
sampling sequence. This isolates cloud changes, rather than freezing the whole
engine. Future shared atmosphere changes must be considered when interpreting it.

The comparison covers all 16 output channels for 14 primary cases. It intentionally
requires identical output, so appearance changes need an explicit decision instead
of silently refreshing the reference. Reflection quality is approximate and is not
covered by this equality test; queue correctness is tested separately on the GPU.
