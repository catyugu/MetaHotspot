function(mhs_copy_runtime_dlls TARGET_NAME)
    # Copy TBB DLL
    if(TARGET TBB::tbb)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:TBB::tbb>
            $<TARGET_FILE_DIR:${TARGET_NAME}>
            COMMENT "Copying TBB runtime to ${TARGET_NAME}"
        )
    endif()

    # Copy MKL/Pardiso DLLs
    if(MHS_ENABLE_PARDISO AND DEFINED MKL_BIN_DIR)
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
    endif()
endfunction()
