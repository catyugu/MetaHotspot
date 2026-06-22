include(${CMAKE_CURRENT_LIST_DIR}/../CPM.cmake)

# muparser - mathematical expression parser (https://github.com/beltoforion/muparser)
CPMAddPackage(
    NAME muparser
    GITHUB_REPOSITORY beltoforion/muparser
    GIT_TAG v2.3.5
    OPTIONS
    "ENABLE_SAMPLES OFF"
    "ENABLE_OPENMP OFF"
    "BUILD_SHARED_LIBS OFF"
    "BUILD_TESTING OFF"
)
if(MSVC)
    target_compile_options(muparser PRIVATE -W0)
else()
    target_compile_options(muparser PRIVATE -Wno-unused-parameter -Wno-error)
endif()
