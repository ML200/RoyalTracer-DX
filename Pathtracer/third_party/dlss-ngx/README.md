# NGX parameter interface headers

`include/nvsdk_ngx_defs.h` and `include/nvsdk_ngx_params.h` are unmodified NVIDIA headers copied from `external/ngx-sdk/include` in the official Streamline 2.14.1 SDK. `LICENSE.txt` is the accompanying `external/ngx-sdk/license.txt`.

Source: [Streamline 2.14.1 release](https://github.com/NVIDIA-RTX/Streamline/releases/tag/v2.14.1), archive `streamline-sdk-v2.14.1.zip`.

Archive SHA-256:

```text
92c4d954631a1710da86ca3fa8d5034f2b9503838c95fc4ae977ae149319781b
```

The native NR bridge needs only the parameter ABI and result definitions. No NGX core import library is linked and these headers do not provide a public NR API. See `docs/DLSS5.md` for the separately inspected runtime contract and provenance.
