add_library(mhs_core INTERFACE)

target_include_directories(mhs_core INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(mhs_core INTERFACE mhs_expression)

# The model, mesh, solution and solve-option types: the data every other
# component and every consumer passes around.
mhs_install_headers(core
    types.hpp
    discrete_mesh.hpp
    fluid_domain.hpp
    output_mesh.hpp
    model.hpp
    model_definition.hpp
    solution.hpp
    solver.hpp
)
