add_library(mhs_options INTERFACE)

# Resolve the command-line frontend at configure time (compatible with CMake 3.26).
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(_MHS_MSVC_FRONTEND 1)
    set(_MHS_GNU_FRONTEND 0)
else()
    set(_MHS_MSVC_FRONTEND 0)
    set(_MHS_GNU_FRONTEND 1)
endif()

# Project warnings and source encoding only; CMake owns optimization and debug defaults.
target_compile_options(mhs_options INTERFACE
    $<${_MHS_GNU_FRONTEND}:-Wall>
    $<${_MHS_GNU_FRONTEND}:-Wextra>
    $<${_MHS_GNU_FRONTEND}:-Wpedantic>
    $<$<AND:${_MHS_GNU_FRONTEND},$<CXX_COMPILER_ID:Clang,AppleClang>>:
    -Wno-language-extension-token
    -Wno-microsoft-enum-value
    -Wno-nested-anon-types
    >
    $<$<AND:${_MHS_GNU_FRONTEND},$<CXX_COMPILER_ID:GNU>>:
    -Wno-stringop-overflow
    -Wno-array-bounds
    >
)

target_compile_options(mhs_options INTERFACE
    $<${_MHS_MSVC_FRONTEND}:
    /W4
    /utf-8
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
    # Exported STL members and base classes are normal library interfaces.
    $<$<COMPILE_LANGUAGE:CXX>:/wd4251>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4275>
    >
)

if(MHS_WERROR)
    target_compile_options(mhs_options INTERFACE
        $<${_MHS_GNU_FRONTEND}:-Werror>
        $<${_MHS_MSVC_FRONTEND}:/WX>
    )
endif()

# Profiling and debugger conveniences belong to the top-level build owner.
if(PROJECT_IS_TOP_LEVEL)
    target_compile_options(mhs_options INTERFACE
        $<$<AND:${_MHS_GNU_FRONTEND},$<CONFIG:Debug>>:-g3>
        $<$<AND:${_MHS_GNU_FRONTEND},$<CONFIG:Debug>>:-ggdb>
        $<$<AND:${_MHS_GNU_FRONTEND},$<CONFIG:Release>>:-fno-omit-frame-pointer>
        $<$<AND:${_MHS_GNU_FRONTEND},$<CONFIG:RelWithDebInfo>>:-fno-omit-frame-pointer>
    )
    target_link_options(mhs_options INTERFACE
        $<$<AND:$<PLATFORM_ID:Linux>,${_MHS_GNU_FRONTEND},$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>>:-rdynamic>
    )
endif()

option(MHS_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
if(MHS_ENABLE_COVERAGE)
    target_compile_options(mhs_options INTERFACE
        $<$<AND:${_MHS_GNU_FRONTEND},$<CONFIG:Debug>>:--coverage>
    )
    target_link_options(mhs_options INTERFACE
        $<$<AND:${_MHS_GNU_FRONTEND},$<CONFIG:Debug>>:--coverage>
    )
endif()

unset(_MHS_GNU_FRONTEND)
unset(_MHS_MSVC_FRONTEND)
