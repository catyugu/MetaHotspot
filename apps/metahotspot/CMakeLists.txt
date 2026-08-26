# Executables intentionally do NOT link mhs_options: the old structure only
# applied strict warnings to library targets, not to the metahotspot CLI.
add_executable(metahotspot main.cpp cli.cpp cli.hpp)
target_link_libraries(metahotspot PRIVATE mhs_compiler mhs_solver mhs_io mhs_logging)
target_include_directories(metahotspot PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
mhs_copy_runtime_dlls(metahotspot)
