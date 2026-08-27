# Runtime DLL copying helpers for Windows executables (TBB / oneMKL).
# Moved out of Utilities.cmake during the modern-cpp-template CMake refactor;
# needed because a Windows executable launched from the build tree must find
# its DLLs next to itself. No-ops when the relevant dependency is absent.

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
