if(NOT WIN32 OR NOT "libomp" IN_LIST OpenMP_CXX_LIB_NAMES)
    return()
endif()

# FindOpenMP already located the import library. Its first DLL record also
# handles MSVC's versioned LLVM runtime names, without scanning every import.
file(STRINGS "${OpenMP_libomp_LIBRARY}" _runtime_dll
    LIMIT_COUNT 1 REGEX "^libomp[0-9A-Za-z_.-]*\\.dll$")

add_library(mhs_delayomp OBJECT delayomp.cpp)
target_compile_definitions(mhs_delayomp PRIVATE
    MHS_OPENMP_IMPORT_DLL="${_runtime_dll}"
    MHS_OPENMP_IMPORT_DLL_W=L"${_runtime_dll}")
target_link_libraries(mhs_delayomp
    PRIVATE $<BUILD_INTERFACE:mhs_options>
    INTERFACE delayimp)
target_link_options(mhs_delayomp INTERFACE
    "LINKER:/DELAYLOAD:${_runtime_dll}" "LINKER:/DELAY:NOBIND")
