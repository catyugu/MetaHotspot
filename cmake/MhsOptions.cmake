# Strict-warning/standard flags for MetaHotspot's own targets, as a single
# INTERFACE "options" library (modern-cpp-template idiom). Third-party
# dependencies brought in via CPM never link this.
#
# Targets opt in by linking `mhs_options`:
#   target_link_libraries(mhs_foo PRIVATE mhs_options)
#
# Flags are selected per compiler frontend via generator expressions, so a
# single CMakeLists applies to MSVC, clang-cl, clang and GCC alike.

add_library(mhs_options INTERFACE)

set(_MHS_MSVC_FRONTEND $<CXX_COMPILER_FRONTEND_VARIANT:MSVC>)
set(_MHS_GNU_OR_CLANG $<OR:$<CXX_COMPILER_FRONTEND_VARIANT:GNU>,$<CXX_COMPILER_FRONTEND_VARIANT:Clang>>)

target_compile_options(mhs_options INTERFACE
    $<${_MHS_MSVC_FRONTEND}:
        /W4 /WX /bigobj /permissive- /utf-8 /wd4244
    >
    $<${_MHS_GNU_OR_CLANG}:
        -Wall -Wextra -Wpedantic -Werror
        -Wno-language-extension-token
        -Wno-microsoft-enum-value
        -Wno-nested-anon-types
        -Wno-stringop-overflow
        -Wno-array-bounds
    >
)

unset(_MHS_MSVC_FRONTEND)
unset(_MHS_GNU_OR_CLANG)
