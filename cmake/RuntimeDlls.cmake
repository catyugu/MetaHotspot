# Runtime deployment for executables and shared libraries.
#
# $<TARGET_RUNTIME_DLLS:> covers imported shared targets in the link closure
# (oneTBB, and oneMKL when enabled). FindOpenMP exposes an interface target, so
# MSVC's OpenMP DLL is not part of that list. Let CMake's own redistributable
# discovery locate it instead of depending on Visual Studio's directory layout.
# The MSVC C runtime is deliberately not bundled; users get it from the Visual
# C++ Redistributable.

function(_mhs_find_msvc_openmp_runtime out)
    set(${out} "" PARENT_SCOPE)
    # clang-cl uses the MSVC command-line frontend but links LLVM OpenMP,
    # not the MSVC vcomp redistributable.
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        return()
    endif()

    # Discover compiler redistributables without adding the module's default
    # install rule; MetaHotspot only app-local deploys the OpenMP runtime.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
    set(CMAKE_INSTALL_OPENMP_LIBRARIES ON)
    include(InstallRequiredSystemLibraries)

    set(_openmp "")
    foreach(_runtime IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        get_filename_component(_name "${_runtime}" NAME)
        string(TOLOWER "${_name}" _name)
        if(_name MATCHES "^vcomp[0-9]+d?\\.dll$")
            list(APPEND _openmp "${_runtime}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _openmp)
    if(NOT _openmp)
        message(FATAL_ERROR "CMake could not locate the MSVC OpenMP redistributable")
    endif()
    set(${out} "${_openmp}" PARENT_SCOPE)
endfunction()

_mhs_find_msvc_openmp_runtime(_mhs_msvc_openmp_runtime)

function(mhs_stage_runtime target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:${target}>"
        $<TARGET_RUNTIME_DLLS:${target}>
        COMMAND_EXPAND_LISTS
    )
    if(_mhs_msvc_openmp_runtime)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${_mhs_msvc_openmp_runtime} "$<TARGET_FILE_DIR:${target}>"
            COMMAND_EXPAND_LISTS
        )
    endif()
endfunction()

# Installs the runtime DLLs of the given targets next to the installed binaries.
# On Windows those are oneTBB, oneMKL when enabled, and the MSVC OpenMP runtime.
function(mhs_install_windows_runtime)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "DESTINATION;COMPONENT" "TARGETS")
    if(NOT WIN32)
        return()
    endif()
    set(_component "")
    if(arg_COMPONENT)
        set(_component COMPONENT ${arg_COMPONENT})
    endif()
    foreach(_target IN LISTS arg_TARGETS)
        if(TARGET ${_target})
            install(FILES $<TARGET_RUNTIME_DLLS:${_target}>
                DESTINATION ${arg_DESTINATION} ${_component})
        endif()
    endforeach()
    if(_mhs_msvc_openmp_runtime)
        install(FILES ${_mhs_msvc_openmp_runtime}
            DESTINATION ${arg_DESTINATION} ${_component})
    endif()
endfunction()
