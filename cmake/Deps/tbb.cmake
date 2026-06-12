include(${CMAKE_CURRENT_LIST_DIR}/../CPM.cmake)

# oneTBB - parallel assembly
CPMAddPackage(
    NAME TBB
    GITHUB_REPOSITORY "oneapi-src/oneTBB"
    GIT_TAG "v2023.0.0"
    OPTIONS
    "BUILD_SHARED_LIBS ON"
    "TBB_STRICT OFF"
    "TBBMALLOC_BUILD OFF"
    "TBB_TEST OFF"
    "TBB_EXAMPLES OFF"
)

# TBB 2023.0.0 needs two MinGW compatibility settings in Debug builds.
if(MINGW AND TARGET tbb)
    target_compile_definitions(tbb
        PUBLIC TBB_USE_PROFILING_TOOLS=0
        PRIVATE _WIN32_WINNT=0x0A00 WINVER=0x0A00
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        get_target_property(_tbb_compile_options tbb COMPILE_OPTIONS)
        if(_tbb_compile_options)
            list(REMOVE_ITEM _tbb_compile_options "-U__STRICT_ANSI__")
            set_property(TARGET tbb PROPERTY COMPILE_OPTIONS "${_tbb_compile_options}")
        endif()
    endif()
endif()
