add_executable(metahotspot main.cpp)
target_link_libraries(metahotspot PRIVATE mhs_lib)
target_include_directories(metahotspot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external
)
mhs_copy_runtime_dlls(metahotspot)
