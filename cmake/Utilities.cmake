# Project-wide warning/standard settings.
#
# These are applied ONLY to MetaHotspot's own targets, never to third-party
# dependencies brought in via CPM. Use mhs_set_strict_warnings() on each
# project target; do not call add_compile_options() at file scope.

function(mhs_set_strict_warnings TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE
            /W4 /WX /bigobj /permissive- /utf-8
        )
    else()
        target_compile_options(${TARGET_NAME} PRIVATE
            -Wall -Wextra -Wpedantic -Werror
            -Wno-language-extension-token
            -Wno-microsoft-enum-value
            -Wno-nested-anon-types
        )
    endif()
endfunction()

# Helper for adding a MetaHotspot static library.
#
# Centralises the boilerplate that every module's CMakeLists repeats:
#   - STATIC library type (explicit, never implicit)
#   - the include directory that lets targets see each other via PUBLIC
#   - the strict warning flags, scoped to this target only
#
# Usage:
#   mhs_add_library(common_lib SOURCES logger.cpp
#                   PUBLIC_LINKS spdlog
#                   PRIVATE_LINKS Eigen3::Eigen)
#
# The result is a target named <name> that downstream targets can link with
# target_link_libraries(... PUBLIC/PRIVATE <name>).


function(mhs_add_library NAME)
    set(options)
    set(one_value_args)
    set(multi_value_args SOURCES PUBLIC_LINKS PRIVATE_LINKS)
    cmake_parse_arguments(MHS_LIB "${options}" "" "${multi_value_args}" ${ARGN})

    add_library(${NAME} STATIC ${MHS_LIB_SOURCES})

    if(MHS_LIB_PUBLIC_LINKS)
        target_link_libraries(${NAME} PUBLIC ${MHS_LIB_PUBLIC_LINKS})
    endif()
    if(MHS_LIB_PRIVATE_LINKS)
        target_link_libraries(${NAME} PRIVATE ${MHS_LIB_PRIVATE_LINKS})
    endif()

    target_include_directories(${NAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )

    mhs_set_strict_warnings(${NAME})
endfunction()


# Copy Intel oneTBB runtime DLL next to a Windows executable so it can be
# launched from the build tree. No-op on non-Windows and when TBB is absent.
function(mhs_copy_tbb_runtime TARGET_NAME)
    if(TARGET TBB::tbb)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:TBB::tbb>
            $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMENT "Copying TBB runtime to ${TARGET_NAME}"
        )
    endif()
endfunction()


# Copy Intel oneMKL runtime DLLs next to a Windows executable. The set of
# DLLs depends on the link type (sdl vs static) and threading runtime
# (intel_thread vs openmp); we glob via CONFIGURE_DEPENDS so adding a new
# runtime is picked up on the next configure.
function(mhs_copy_mkl_runtime TARGET_NAME)
    if(NOT MHS_ENABLE_PARDISO OR NOT DEFINED MKL_BIN_DIR)
        return()
    endif()
    file(GLOB _mkl_runtime_dlls CONFIGURE_DEPENDS
        "${MKL_BIN_DIR}/mkl_*.dll"
        "${MKL_BIN_DIR}/libiomp5md.dll"
        "${MKL_BIN_DIR}/libomp.dll"
    )
    foreach(_dll IN LISTS _mkl_runtime_dlls)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_dll}"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        )
    endforeach()
endfunction()

function(mhs_copy_runtime_dlls TARGET_NAME)
    mhs_copy_tbb_runtime(${TARGET_NAME})
    mhs_copy_mkl_runtime(${TARGET_NAME})
endfunction()
