# Vendored Streamline SDK

Updated on 9 September 2026 from NVIDIA's official [Streamline 2.14.1 release](https://github.com/NVIDIA-RTX/Streamline/releases/tag/v2.14.1), published 8 September 2026.

Archive: `streamline-sdk-v2.14.1.zip`.

SHA-256:

```text
92c4d954631a1710da86ca3fa8d5034f2b9503838c95fc4ae977ae149319781b
```

The vendored `include/`, `lib/x64/`, and `bin/x64/` trees, including development binaries, come from the same release. Its DLSS SR/RR/FG model runtimes report 310.9.1. Root and per-runtime license files accompany the SDK. CMake selects release binaries and stages them after the project's legacy `include/` assets.

The public release declares NR feature/tag IDs but does not contain an NR API header, guide, plugin or model DLL. The user's separate 310.8.0 NR runtime is staged under the executable's `dlssnr/` directory and accessed through the experimental adapter described in `docs/DLSS5.md`.
