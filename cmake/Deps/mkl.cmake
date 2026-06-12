include(${CMAKE_CURRENT_LIST_DIR}/../CPM.cmake)

# Intel oneMKL provides MKL::MKL. Pardiso is optional: when oneMKL is absent,
# the existing solver factory falls back to Eigen SparseLU.
set(MHS_ENABLE_PARDISO FALSE CACHE INTERNAL "" FORCE)

if(USE_MKL)
    set(MKL_LINK "sdl" CACHE STRING "MKL link type (sdl|static|dynamic)")
    set(MKL_THREADING "intel_thread" CACHE STRING "MKL threading runtime")
    set(MKL_INTERFACE "lp64" CACHE STRING "MKL index interface (lp64|ilp64)")
    find_package(MKL QUIET)

    if(TARGET MKL::MKL)
        set(MHS_ENABLE_PARDISO TRUE CACHE INTERNAL "" FORCE)

        # SDL packages expose MKL::mkl_rt; use it only for optional DLL copying.
        if(TARGET MKL::mkl_rt)
            get_target_property(_mkl_rt_loc MKL::mkl_rt LOCATION)
            get_filename_component(MKL_BIN_DIR "${_mkl_rt_loc}" DIRECTORY)
        else()
            message(WARNING
                "oneMKL was found without MKL::mkl_rt; runtime DLLs will not "
                "be copied automatically")
        endif()
    else()
        message(WARNING
            "USE_MKL=ON but oneMKL was not found; disabling Pardiso and "
            "falling back to Eigen SparseLU")
    endif()
else()
    message(STATUS "USE_MKL=OFF; Pardiso support is disabled")
endif()
