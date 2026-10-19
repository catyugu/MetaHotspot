# Executables intentionally do NOT link mhs_options: the old structure only
# applied strict warnings to library targets, not to the metahotspot CLI.
add_executable(metahotspot main.cpp cli.cpp cli.hpp)
target_link_libraries(metahotspot PRIVATE mhs_compiler mhs_solver mhs_io mhs_logging)
mhs_copy_runtime_dlls(metahotspot)

if(APPLE)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
