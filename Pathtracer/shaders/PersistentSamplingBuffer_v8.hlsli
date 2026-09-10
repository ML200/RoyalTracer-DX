#ifndef PERSISTENT_SAMPLING_BUFFER_HLSLI
#define PERSISTENT_SAMPLING_BUFFER_HLSLI
#if SHARC_UPDATE_PASS && !defined(SHARC_READ_ONLY)
globallycoherent RWByteAddressBuffer g_sharc : register(u27);
#else
RWByteAddressBuffer g_sharc : register(u27);
#endif
#endif
