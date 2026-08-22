add_library(mhs_expression STATIC expression/expr.cpp)
target_include_directories(mhs_expression PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_expression PRIVATE
    mhs_options
    TBB::tbb
    muparser
)

set(_linear_sources
    linear/linear_solver.cpp
)

# AMG solver lives in its own TU compiled WITHOUT the strict mhs_options flags:
# amg_solver.cpp instantiates the heavy AMGCL template library and must not be
# compiled with /WX / -Werror. It also deliberately does NOT link the `amgcl`
# CMake target: that INTERFACE target propagates amgcl's own -Wall / OpenMP
# flags (e.g. '-Xclang -fopenmp', which clang-cl rejects) into every consumer,
# which breaks the strict-warning mhs_linear. We only take AMGCL's headers.
add_library(mhs_amg STATIC linear/amg_solver.cpp)
target_include_directories(mhs_amg PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${amgcl_SOURCE_DIR}
)
target_link_libraries(mhs_amg PUBLIC Eigen3::Eigen)
target_compile_features(mhs_amg PRIVATE cxx_std_20)
target_compile_definitions(mhs_amg PRIVATE AMGCL_NO_BOOST)
# AMGCL emits -Wunsafe-buffer-usage in its smoothed-aggregation templates on
# clang/clang-cl; harmless for us, but suppress so the log stays readable.
target_compile_options(mhs_amg PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wno-unsafe-buffer-usage>
)

if(MHS_ENABLE_PARDISO)
    list(APPEND _linear_sources linear/pardiso_lu_solver.cpp)
endif()

add_library(mhs_linear STATIC ${_linear_sources})
target_include_directories(mhs_linear PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mhs_linear PUBLIC Eigen3::Eigen)
target_link_libraries(mhs_linear PRIVATE mhs_options)
target_link_libraries(mhs_linear PRIVATE mhs_amg)

if(MHS_ENABLE_PARDISO)
    target_link_libraries(mhs_linear PUBLIC MKL::MKL)
    target_compile_definitions(mhs_linear PUBLIC MHS_ENABLE_PARDISO)
endif()
