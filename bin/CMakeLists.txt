add_executable(metahotspot main.cpp cli.cpp cli.hpp)
target_link_libraries(metahotspot PRIVATE mhs_engine mhs_io mhs_logging)
target_include_directories(metahotspot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
mhs_copy_runtime_dlls(metahotspot)
