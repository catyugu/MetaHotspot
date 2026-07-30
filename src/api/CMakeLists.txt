add_library(mhs_c_api SHARED
    metahotspot.cpp
    metahotspot_macromodel.cpp
)
target_link_libraries(mhs_c_api PRIVATE
    mhs_compiler
    mhs_solver
    mhs_macromodel
    mhs_io
)
target_include_directories(mhs_c_api PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(mhs_c_api PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_definitions(mhs_c_api PRIVATE
    MHS_API_EXPORT
    MHS_BUILD_SHARED
)
mhs_set_strict_warnings(mhs_c_api)
mhs_copy_runtime_dlls(mhs_c_api)
