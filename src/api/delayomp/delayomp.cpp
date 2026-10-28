// Windows Intel/LLVM ABI: redirect the entire OpenMP delay import table to
// one compatible provider. No Intel SDK is needed at build time.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
// delayimp.h requires Windows types; preserve this include order.
// clang-format off
#include <windows.h>
#include <delayimp.h>
// clang-format on

#include <cstring>
#include <stdexcept>

extern "C" IMAGE_DOS_HEADER __ImageBase;

namespace {
    FARPROC WINAPI openmp_delay_hook(unsigned notification, PDelayLoadInfo info)
    {
        if (notification != dliNotePreLoadLibrary || std::strcmp(info->szDll, MHS_OPENMP_IMPORT_DLL) != 0) {
            return nullptr;
        }

        // A local static makes selection consistent even when the first calls
        // arrive concurrently. Keep the module loaded for the DLL's lifetime.
        static const HMODULE runtime = [info] {
            const wchar_t* candidates[] = {L"libiomp5md.dll", MHS_OPENMP_IMPORT_DLL_W};
            const auto* image = reinterpret_cast<const unsigned char*>(&__ImageBase);
            const auto* imports = reinterpret_cast<const IMAGE_THUNK_DATA*>(image + info->pidd->rvaINT);
            for (const wchar_t* name : candidates) {
                // Includes directories registered by Python's add_dll_directory.
                HMODULE candidate = LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
                if (!candidate) {
                    candidate = LoadLibraryW(name);
                }
                if (!candidate) {
                    continue;
                }
                // Validate the real DLL's whole import table before *any*
                // OpenMP function runs, rather than discovering missing symbols
                // part way through a solve or mixing providers per symbol.
                bool compatible = true;
                for (const auto* entry = imports; entry->u1.AddressOfData; ++entry) {
                    const char* name = IMAGE_SNAP_BY_ORDINAL(entry->u1.Ordinal)
                        ? MAKEINTRESOURCEA(IMAGE_ORDINAL(entry->u1.Ordinal))
                        : reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(image + entry->u1.AddressOfData)->Name;
                    if (!GetProcAddress(candidate, name)) {
                        compatible = false;
                        break;
                    }
                }
                if (compatible) {
                    return candidate;
                }
                FreeLibrary(candidate);
            }
            throw std::runtime_error("no compatible OpenMP runtime could be loaded");
        }();
        // The delay-load helper owns the returned reference. It may release a
        // redundant reference when two threads resolve their first imports.
        HMODULE reference = nullptr;
        if (!GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(runtime), &reference)) {
            throw std::runtime_error("could not retain the selected OpenMP runtime");
        }
        return reinterpret_cast<FARPROC>(reference);
    }
}

extern "C" const PfnDliHook __pfnDliNotifyHook2 = openmp_delay_hook;
