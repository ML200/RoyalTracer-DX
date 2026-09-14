# Frozen cumulus reference

These shader sources are the slow, bit stable baseline used by
`compare_cumulus_reference.py`. Keep the source files unchanged when updating
the production cloud shaders.

Run both reference and candidate modes at the same resolution:

```powershell
./tests/run_cumulus_tests.ps1 -Reference -OutputDirectory out/cumulus/reference
./tests/run_cumulus_tests.ps1 -OutputDirectory out/cumulus/candidate
python ./tests/compare_cumulus_reference.py out/cumulus/reference out/cumulus/candidate
```

The comparison checks all 16 primary output channels for the deterministic
cases listed by the comparison script. It does not cover secondary quality or
the full renderer.
