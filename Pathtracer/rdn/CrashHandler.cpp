#include "stdafx.h"
#define ENABLE_D3D12_DIAGNOSTICS 1
#include "Diagnostics.h"

#if ENABLE_D3D12_DIAGNOSTICS
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")

namespace dxdiag {
namespace {
void module_and_offset(uintptr_t addr, wchar_t* out, size_t cap) {
    HMODULE mod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(addr), &mod) &&
        mod) {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(mod, path, MAX_PATH);
        const wchar_t* base = wcsrchr(path, L'\\');
        base = base ? base + 1 : path;
        _snwprintf_s(out, cap, _TRUNCATE, L"%ls+0x%llX", base, (unsigned long long)(addr - (uintptr_t)mod));
    } else {
        _snwprintf_s(out, cap, _TRUNCATE, L"0x%llX", (unsigned long long)addr);
    }
}

// Once per process: a second SymInitialize fails.
bool EnsureSymbols(HANDLE process) {
    static bool initialized = false, ok = false;
    if (!initialized) {
        initialized = true;
        SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
        ok = SymInitializeW(process, nullptr, TRUE) != FALSE;
    }
    return ok;
}

void symbol_line(HANDLE process, bool haveSyms, uintptr_t addr, wchar_t* name, size_t cap) {
    name[0] = L'\0';
    if (!haveSyms)
        return;
    alignas(SYMBOL_INFOW) char buf[sizeof(SYMBOL_INFOW) + 256 * sizeof(wchar_t)] = {};
    SYMBOL_INFOW* sym = reinterpret_cast<SYMBOL_INFOW*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFOW);
    sym->MaxNameLen = 256;
    DWORD64 disp = 0;
    if (!SymFromAddrW(process, (DWORD64)addr, &disp, sym))
        return;
    IMAGEHLP_LINEW64 line = {};
    line.SizeOfStruct = sizeof(line);
    DWORD ldisp = 0;
    if (SymGetLineFromAddrW64(process, (DWORD64)addr, &ldisp, &line))
        _snwprintf_s(name, cap, _TRUNCATE, L"  %ls+0x%llX  (%ls:%lu)", sym->Name, (unsigned long long)disp,
                     line.FileName, line.LineNumber);
    else
        _snwprintf_s(name, cap, _TRUNCATE, L"  %ls+0x%llX", sym->Name, (unsigned long long)disp);
}
}

void LogCallStack(const wchar_t* title) {
    void* frames[40] = {};
    const USHORT n = CaptureStackBackTrace(1, 40, frames, nullptr);
    const HANDLE process = GetCurrentProcess();
    const bool haveSyms = EnsureSymbols(process);
    if (title && title[0])
        CrashLogF(L"%ls\n", title);
    for (USHORT i = 0; i < n; ++i) {
        wchar_t loc[300], name[512];
        module_and_offset((uintptr_t)frames[i], loc, 300);
        symbol_line(process, haveSyms, (uintptr_t)frames[i], name, 512);
        CrashLogF(L"  #%02d %ls%ls\n", i, loc, name);
    }
}

LONG WINAPI CrashExceptionFilter(EXCEPTION_POINTERS* ep) {
    static LONG entered = 0;
    if (InterlockedIncrement(&entered) > 1)
        return EXCEPTION_CONTINUE_SEARCH;

    const EXCEPTION_RECORD* rec = ep ? ep->ExceptionRecord : nullptr;
    wchar_t where[300] = L"?";
    if (rec)
        module_and_offset((uintptr_t)rec->ExceptionAddress, where, 300);
    CrashLogF(L"\n*** unhandled exception 0x%08X at %ls (thread %lu) ***\n", rec ? (unsigned)rec->ExceptionCode : 0u,
              where, GetCurrentThreadId());
    if (rec && rec->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && rec->NumberParameters >= 2)
        CrashLogF(L"  access violation: %ls address 0x%llX\n",
                  rec->ExceptionInformation[0] == 0 ? L"reading"
                                                    : (rec->ExceptionInformation[0] == 1 ? L"writing" : L"executing"),
                  (unsigned long long)rec->ExceptionInformation[1]);
    if (rec && rec->ExceptionCode == 0xE06D7363u)
        CrashLog(L"  (unhandled C++ exception)\n");

    const HANDLE process = GetCurrentProcess();
    const bool haveSyms = EnsureSymbols(process);
    if (ep && ep->ContextRecord) {
        CONTEXT ctx = *ep->ContextRecord;
        STACKFRAME64 frame = {};
        frame.AddrPC.Offset = ctx.Rip;
        frame.AddrPC.Mode = AddrModeFlat;
        frame.AddrFrame.Offset = ctx.Rbp;
        frame.AddrFrame.Mode = AddrModeFlat;
        frame.AddrStack.Offset = ctx.Rsp;
        frame.AddrStack.Mode = AddrModeFlat;
        for (int i = 0; i < 48; ++i) {
            if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, GetCurrentThread(), &frame, &ctx, nullptr,
                             SymFunctionTableAccess64, SymGetModuleBase64, nullptr))
                break;
            if (frame.AddrPC.Offset == 0)
                break;
            wchar_t loc[300], name[512];
            module_and_offset((uintptr_t)frame.AddrPC.Offset, loc, 300);
            symbol_line(process, haveSyms, (uintptr_t)frame.AddrPC.Offset, name, 512);
            CrashLogF(L"  #%02d %ls%ls\n", i, loc, name);
        }
    }
    CrashLog(L"*** end of exception report ***\n");
    return EXCEPTION_CONTINUE_SEARCH;
}
}
#endif
