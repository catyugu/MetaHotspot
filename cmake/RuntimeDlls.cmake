# Runtime deployment for executables and shared libraries.
#
# $<TARGET_RUNTIME_DLLS:> covers imported shared targets in the link closure
# (oneTBB, and oneMKL when enabled). FindOpenMP exposes an interface target, so
# its DLL needs separate discovery for both MSVC and clang-cl.
# SDKs use the Visual C++ Redistributable. Wheels stage the matching compiler
# redistributables so repair tools cannot select older DLLs from Python itself.

function(_mhs_find_openmp_runtime out)
    set(${out} "" PARENT_SCOPE)
    set(_mhs_msvc_crt_runtime "" PARENT_SCOPE)
    if(NOT WIN32)
        return()
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # FindOpenMP provides import libraries, not a SHARED imported target.
        # Read the DLL name embedded in each COFF import library rather than
        # guessing LLVM's versioned DLL name or Visual Studio's layout.
        get_filename_component(_compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set(_openmp "")
        foreach(_library IN LISTS OpenMP_CXX_LIBRARIES)
            if(NOT EXISTS "${_library}")
                continue()
            endif()
            file(STRINGS "${_library}" _names REGEX "[Ll][Ii][Bb].*[Oo][Mm][Pp].*\\.dll")
            get_filename_component(_lib_dir "${_library}" DIRECTORY)
            foreach(_name IN LISTS _names)
                string(REGEX MATCH "[A-Za-z0-9_.-]+\\.dll" _dll_name "${_name}")
                if(_dll_name)
                    unset(_dll CACHE)
                    find_file(_dll NAMES "${_dll_name}"
                        HINTS "${_compiler_bin}" "${_lib_dir}" "${_lib_dir}/../bin"
                        ENV PATH REQUIRED)
                    list(APPEND _openmp "${_dll}")
                endif()
            endforeach()
        endforeach()
        list(REMOVE_DUPLICATES _openmp)
        if(NOT _openmp)
            message(FATAL_ERROR "Cannot locate the LLVM OpenMP runtime from ${OpenMP_CXX_LIBRARIES}")
        endif()
        set(${out} "${_openmp}" PARENT_SCOPE)
        return()
    endif()

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        return()
    endif()

    # Discover compiler redistributables without adding the module's default
    # install rule; SDKs deploy OpenMP and wheels collect the needed CRT DLLs.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP ON)
    set(CMAKE_INSTALL_OPENMP_LIBRARIES ON)
    include(InstallRequiredSystemLibraries)
    set(_mhs_msvc_crt_runtime "${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}" PARENT_SCOPE)

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

_mhs_find_openmp_runtime(_mhs_openmp_runtime)

function(mhs_stage_runtime target)
    set(_runtime ${_mhs_openmp_runtime})
    if(SKBUILD)
        # Dependency discovery must see this compiler's CRT rather than an
        # older runtime beside python.exe. Only needed DLLs enter the wheel.
        list(APPEND _runtime ${_mhs_msvc_crt_runtime})
        list(REMOVE_DUPLICATES _runtime)
    endif()
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy -t "$<TARGET_FILE_DIR:${target}>"
        $<TARGET_RUNTIME_DLLS:${target}> ${_runtime}
        COMMAND_EXPAND_LISTS VERBATIM
    )
endfunction()

# Installs the runtime DLLs of the given targets next to the installed binaries.
# On Windows those are oneTBB, oneMKL when enabled, and the OpenMP runtime.
function(mhs_install_windows_runtime)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "DESTINATION" "TARGETS")
    if(NOT WIN32)
        return()
    endif()
    foreach(_target IN LISTS arg_TARGETS)
        if(TARGET ${_target})
            install(FILES $<TARGET_RUNTIME_DLLS:${_target}>
                DESTINATION ${arg_DESTINATION})
        endif()
    endforeach()
    if(_mhs_openmp_runtime)
        install(FILES ${_mhs_openmp_runtime}
            DESTINATION ${arg_DESTINATION})
    endif()
endfunction()
