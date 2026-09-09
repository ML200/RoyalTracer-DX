// Experimental adapter for the user-supplied signed DLSS-NR 310.8.0 runtime.
// The companion filename contains "nvngx.dll", as required by this runtime's
// caller-module check. No hooks, binary patches, driver modifications or ReShade.
// ABI findings and provenance: docs/DLSS5.md.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "DLSSNRBridge.h"
#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_params.h>
#include <string>
#include <unordered_map>
#include <variant>
#include <memory>
#include <array>
#include <bcrypt.h>

namespace dlssnr {
namespace {
constexpr uint32_t kSuccess = NVSDK_NGX_Result_Success;
constexpr uint32_t kInvalid = NVSDK_NGX_Result_FAIL_InvalidParameter;

bool KnownRuntime(const wchar_t* path) {
    // Private entry points are supported only for the exact binary inspected
    // and tested here. Revalidate the adapter before accepting another build.
    constexpr unsigned char expected[32] = {
        0xe1,0x6b,0xcf,0x15,0xe1,0x6e,0x13,0xf5,0x27,0x49,0x1c,0xdf,0x78,0x45,0xb2,0xfe,
        0x65,0x21,0xa7,0x38,0xd8,0xf7,0xc9,0xc7,0x21,0x86,0x6a,0x84,0x96,0xe1,0xfc,0x8e
    };
    HANDLE file = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) return false;
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
    bool ok = BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) >= 0;
    if (ok) ok = BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0;
    std::array<unsigned char, 65536> buffer;
    DWORD read = 0;
    while (ok) {
        if (!ReadFile(file, buffer.data(), (DWORD)buffer.size(), &read, nullptr)) { ok = false; break; }
        if (!read) break;
        ok = BCryptHashData(hash, buffer.data(), read, 0) >= 0;
    }
    unsigned char digest[32]{};
    if (ok) ok = BCryptFinishHash(hash, digest, sizeof(digest), 0) >= 0 && memcmp(digest, expected, sizeof(digest)) == 0;
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    CloseHandle(file);
    return ok;
}

// Implement the public NGX parameter interface with our own storage. In
// particular, do not mutate Streamline's capability block or guess vtable slots.
class Parameters final : public NVSDK_NGX_Parameter {
    using Value = std::variant<unsigned long long, long long, double, void*>;
    std::unordered_map<std::string, Value> values;
    template<class T> NVSDK_NGX_Result Read(const char* key, T* out) const {
        if (!key || !out) return NVSDK_NGX_Result_FAIL_InvalidParameter;
        auto it = values.find(key);
        if (it == values.end()) return NVSDK_NGX_Result_FAIL_UnsupportedParameter;
        return std::visit([&](auto value) {
            using V = decltype(value);
            if constexpr (std::is_pointer_v<T> && std::is_pointer_v<V>)
                *out = static_cast<T>(value);
            else if constexpr (!std::is_pointer_v<T> && !std::is_pointer_v<V>)
                *out = static_cast<T>(value);
            else return NVSDK_NGX_Result_FAIL_UnsupportedParameter;
            return NVSDK_NGX_Result_Success;
        }, it->second);
    }
public:
    void Set(const char* k, unsigned long long v) override { values[k] = v; }
    void Set(const char* k, unsigned int v) override { values[k] = static_cast<unsigned long long>(v); }
    void Set(const char* k, int v) override { values[k] = static_cast<long long>(v); }
    void Set(const char* k, float v) override { values[k] = static_cast<double>(v); }
    void Set(const char* k, double v) override { values[k] = v; }
    void Set(const char* k, ID3D11Resource* v) override { values[k] = static_cast<void*>(v); }
    void Set(const char* k, ID3D12Resource* v) override { values[k] = static_cast<void*>(v); }
    void Set(const char* k, void* v) override { values[k] = v; }
#define NR_GET(T) NVSDK_NGX_Result Get(const char* k, T* v) const override { return Read(k, v); }
    NR_GET(unsigned long long) NR_GET(unsigned int) NR_GET(int) NR_GET(float) NR_GET(double)
    NR_GET(ID3D11Resource*) NR_GET(ID3D12Resource*) NR_GET(void*)
#undef NR_GET
    void Reset() override { values.clear(); }
};
using Init = uint32_t(__cdecl*)(unsigned long long, const wchar_t*, ID3D12Device*, uint32_t, NVSDK_NGX_Parameter*);
using Populate = uint32_t(__cdecl*)(NVSDK_NGX_Parameter*);
using Create = uint32_t(__cdecl*)(ID3D12GraphicsCommandList*, uint32_t, NVSDK_NGX_Parameter*, void**);
using Evaluate = uint32_t(__cdecl*)(ID3D12GraphicsCommandList*, void*, NVSDK_NGX_Parameter*, void*);
using Release = uint32_t(__cdecl*)(void*);
using Shutdown = uint32_t(__cdecl*)(ID3D12Device*);

template<class T> T Resolve(HMODULE module, const char* name) { return reinterpret_cast<T>(GetProcAddress(module, name)); }
void Subrect(Parameters& p, const char* resource, uint32_t w, uint32_t h) {
    const std::string prefix = std::string("DLSSNR.") + resource + "Subrect";
    p.Set((prefix + "BaseX").c_str(), 0u); p.Set((prefix + "BaseY").c_str(), 0u);
    p.Set((prefix + "Width").c_str(), w); p.Set((prefix + "Height").c_str(), h);
}
}

struct Context {
    HMODULE module = nullptr;
    ID3D12Device* device = nullptr;
    Parameters params;
    void* feature = nullptr;
    Create create = nullptr;
    Evaluate evaluate = nullptr;
    Release release = nullptr;
    Shutdown shutdown = nullptr;
    uint32_t width = 0, height = 0;
    bool initialized = false;
};

uint32_t RoyalNRCreateContext(const wchar_t* runtime, ID3D12Device* device, Context** out) {
    if (!runtime || !device || !out) return kInvalid;
    *out = nullptr;
    if (!KnownRuntime(runtime)) return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
    auto ctx = std::make_unique<Context>();
    ctx->module = LoadLibraryExW(runtime, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!ctx->module) return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    auto init = Resolve<Init>(ctx->module, "NVSDK_NGX_D3D12_Init_Ext");
    auto populate = Resolve<Populate>(ctx->module, "NVSDK_NGX_D3D12_PopulateParameters_Impl");
    ctx->create = Resolve<Create>(ctx->module, "NVSDK_NGX_D3D12_CreateFeature");
    ctx->evaluate = Resolve<Evaluate>(ctx->module, "NVSDK_NGX_D3D12_EvaluateFeature");
    ctx->release = Resolve<Release>(ctx->module, "NVSDK_NGX_D3D12_ReleaseFeature");
    ctx->shutdown = Resolve<Shutdown>(ctx->module, "NVSDK_NGX_D3D12_Shutdown1");
    if (!init || !populate || !ctx->create || !ctx->evaluate || !ctx->release || !ctx->shutdown) {
        FreeLibrary(ctx->module); return NVSDK_NGX_Result_FAIL_FeatureNotFound;
    }
    wchar_t dataPath[MAX_PATH]{};
    GetTempPathW(MAX_PATH, dataPath);
    // 0 = unregistered custom application, never another game's NGX identity.
    volatile uint32_t result = init(0, dataPath, device, 0x15, &ctx->params);
    if (result != kSuccess) { FreeLibrary(ctx->module); return result; }
    ctx->initialized = true; ctx->device = device; device->AddRef();
    result = populate(&ctx->params);
    if (result != kSuccess) { RoyalNRDestroyContext(ctx.release()); return result; }
    *out = ctx.release();
    return kSuccess;
}

uint32_t RoyalNRCreateFeature(Context* ctx, ID3D12GraphicsCommandList* cmd, uint32_t w, uint32_t h, const Tuning* tuning) {
    if (!ctx || !cmd || !w || !h || !tuning || ctx->feature) return kInvalid;
    if (tuning->modelStyle >= kModelStyleCount || tuning->renderPreset >= kRenderPresetCount) return kInvalid;
    auto& p = ctx->params;
    p.Set("DLSSNR.Width", static_cast<int>(w)); p.Set("DLSSNR.Height", static_cast<int>(h));
    p.Set("CreationNodeMask", 1u); p.Set("VisibilityNodeMask", 1u);
    p.Set("DLSSNR.Hint.Render.Preset", tuning->renderPreset);
    p.Set("DLSSNR.ScalingRatio", 1.0f); p.Set("DLSSNR.Upscaling", 0u);
    p.Set("DLSSNR.Enabled", 1u);
    p.Set("DLSSNR.Intensity", tuning->intensity);
    p.Set("DLSSNR.LocalToneStrength", tuning->localTone);
    p.Set("DLSSNR.LocalStructureStrength", tuning->localStructure);
    p.Set("DLSSNR.SkinStructureStrength", tuning->skinStructure);
    p.Set("DLSSNR.UseAutoMask", tuning->autoMask);
    p.Set("DLSSNR.Style", tuning->modelStyle); p.Set("DLSSNR.UICorrection", 0u);
    volatile uint32_t result = ctx->create(cmd, 18, &p, &ctx->feature);
    if (result == kSuccess) { ctx->width = w; ctx->height = h; }
    return result;
}

uint32_t RoyalNREvaluate(Context* ctx, ID3D12GraphicsCommandList* cmd, const Frame* f) {
    if (!ctx || !ctx->feature || !cmd || !f || !f->color || !f->output || !f->depth || !f->motion ||
        f->color == f->output || f->width != ctx->width || f->height != ctx->height || !f->guideWidth || !f->guideHeight) return kInvalid;
    auto& p = ctx->params;
    p.Set("DLSSNR.Color", f->color); p.Set("DLSSNR.Output", f->output);
    p.Set("DLSSNR.Depth", f->depth); p.Set("DLSSNR.MVec", f->motion);
    Subrect(p, "Color", f->width, f->height); Subrect(p, "Output", f->width, f->height);
    Subrect(p, "Depth", f->guideWidth, f->guideHeight); Subrect(p, "MVec", f->guideWidth, f->guideHeight);
    p.Set("DLSSNR.MVecScaleX", f->motionScaleX); p.Set("DLSSNR.MVecScaleY", f->motionScaleY);
    p.Set("DLSSNR.DepthInverted", 1u); p.Set("DLSSNR.Reset", f->reset);
    // The engine runs NR before UI. Explicitly clear optional pointers.
    for (const char* key : {"DLSSNR.UI", "DLSSNR.UIAlpha", "DLSSNR.Backbuffer", "DLSSNR.ControlMask", "DLSSNR.BidirectionalDistortionField"})
        p.Set(key, static_cast<void*>(nullptr));
    volatile uint32_t result = ctx->evaluate(cmd, ctx->feature, &p, nullptr);
    return result;
}

uint32_t RoyalNRReleaseFeature(Context* ctx) {
    if (!ctx || !ctx->feature) return kSuccess;
    volatile uint32_t result = ctx->release(ctx->feature);
    if (result == kSuccess) { ctx->feature = nullptr; ctx->width = ctx->height = 0; }
    return result;
}
void RoyalNRDestroyContext(Context* ctx) {
    if (!ctx) return;
    RoyalNRReleaseFeature(ctx);
    if (ctx->initialized) { volatile uint32_t result = ctx->shutdown(ctx->device); (void)result; }
    if (ctx->device) ctx->device->Release();
    if (ctx->module) FreeLibrary(ctx->module);
    delete ctx;
}
}
