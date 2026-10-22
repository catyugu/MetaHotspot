# The CLI is part of this project, so it builds under the same warning policy.
add_executable(metahotspot main.cpp cli.cpp cli.hpp)
target_link_libraries(metahotspot PRIVATE mhs_metahotspot mhs_options)
mhs_stage_runtime(metahotspot)

if(APPLE)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set_property(TARGET metahotspot PROPERTY INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
