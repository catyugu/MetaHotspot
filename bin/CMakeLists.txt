add_executable(metahotspot main.cpp cli.cpp cli.hpp)
target_link_libraries(metahotspot PRIVATE mhs_lib logger_lib)
target_include_directories(metahotspot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
mhs_copy_runtime_dlls(metahotspot)
