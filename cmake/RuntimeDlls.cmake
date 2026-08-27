# Copy Intel oneMKL runtime DLLs next to a Windows executable.
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
    mhs_copy_mkl_runtime(${TARGET_NAME})
endfunction()
