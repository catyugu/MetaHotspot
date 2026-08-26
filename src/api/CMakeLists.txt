add_library(mhs_c_api SHARED
    metahotspot.cpp
)
target_link_libraries(mhs_c_api PRIVATE
    mhs_compiler
    mhs_solver
    mhs_io
)
target_include_directories(mhs_c_api
    PUBLIC ${CMAKE_SOURCE_DIR}/include
    PRIVATE ${CMAKE_SOURCE_DIR}/src
)
target_compile_definitions(mhs_c_api PRIVATE
    MHS_API_EXPORT
    MHS_BUILD_SHARED
)
target_link_libraries(mhs_c_api PRIVATE mhs_options)
mhs_copy_runtime_dlls(mhs_c_api)
