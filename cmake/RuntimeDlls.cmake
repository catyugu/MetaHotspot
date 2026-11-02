# Runtime deployment for executables and shared libraries.
#
# $<TARGET_RUNTIME_DLLS:> covers the imported shared targets of the link closure
# (oneTBB, and oneMKL when enabled) and is empty where a platform has no runtime
# DLLs, so the copy needs no platform guard. OpenMP is not part of it: FindOpenMP
# exposes an interface target, so the runtime behind /openmp is staged by name.
# The MSVC C runtime is never staged; it comes from the Visual C++ Redistributable.

function(mhs_stage_runtime target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:${target}>"
        $<TARGET_RUNTIME_DLLS:${target}>
        COMMAND_EXPAND_LISTS
    )
endfunction()

# Sets <out> to the OpenMP runtime that an MSVC build loads, and to nothing when
# the platform has none or the build is not the supported Windows configuration.
function(mhs_windows_openmp_runtime out)
    set(${out} "" PARENT_SCOPE)
    if(NOT WIN32)
        return()
    endif()
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        return()
    endif()

    if(CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
        set(_arch arm64)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_arch x64)
    else()
        set(_arch x86)
    endif()
    set(_name "${_arch}/Microsoft.VC${MSVC_TOOLSET_VERSION}.OpenMP/vcomp140.dll")

    set(_globs "")
    if(DEFINED ENV{VCToolsRedistDir})
        file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" _redist)
        list(APPEND _globs "${_redist}/${_name}")
    endif()
    if(IS_ABSOLUTE "${CMAKE_CXX_COMPILER}")
        get_filename_component(_vc "${CMAKE_CXX_COMPILER}" REALPATH)
        string(REGEX REPLACE "/Tools/MSVC/.*$" "" _vc "${_vc}")
        list(APPEND _globs "${_vc}/Redist/MSVC/*/${_name}")
    endif()

    set(_matches "")
    foreach(_glob IN LISTS _globs)
        file(GLOB _found "${_glob}")
        list(APPEND _matches ${_found})
    endforeach()
    if(NOT _matches)
        message(FATAL_ERROR
            "vcomp140.dll is missing from the MSVC redistributable files; looked in: ${_globs}")
    endif()
    list(SORT _matches)
    list(GET _matches -1 _newest)
    set(${out} "${_newest}" PARENT_SCOPE)
endfunction()

function(mhs_stage_windows_openmp target)
    mhs_windows_openmp_runtime(_runtime)
    if(NOT _runtime)
        return()
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_runtime}" "$<TARGET_FILE_DIR:${target}>"
    )
endfunction()

# Installs the runtime DLLs of the given targets next to the installed binaries.
# On Windows those are oneTBB, oneMKL when enabled, and the OpenMP runtime.
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
    mhs_windows_openmp_runtime(_runtime)
    if(_runtime)
        install(FILES "${_runtime}" DESTINATION ${arg_DESTINATION} ${_component})
    endif()
endfunction()
