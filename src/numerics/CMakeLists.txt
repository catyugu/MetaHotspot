add_library(mhs_expression STATIC expression/expr.cpp)
target_include_directories(mhs_expression PRIVATE ${PROJECT_SOURCE_DIR}/include/metahotspot)
target_link_libraries(mhs_expression PRIVATE
    $<BUILD_INTERFACE:mhs_options>
    TBB::tbb
    muparser::muparser
)

set(_linear_sources
    linear/linear_solver.cpp
    linear/amg_solver.cpp
)
if(MHS_ENABLE_PARDISO)
    list(APPEND _linear_sources linear/pardiso_lu_solver.cpp)
endif()

add_library(mhs_linear STATIC ${_linear_sources})
target_include_directories(mhs_linear PRIVATE ${PROJECT_SOURCE_DIR}/include/metahotspot)
# The builtin backend is what this project uses, whether or not a machine
# happens to have Boost installed.
target_compile_definitions(mhs_linear PRIVATE AMGCL_NO_BOOST)
target_link_libraries(mhs_linear
    PUBLIC Eigen3::Eigen
    PRIVATE
    $<BUILD_INTERFACE:mhs_options>
    $<BUILD_INTERFACE:amgcl::amgcl>
    OpenMP::OpenMP_CXX
)

# amgcl is third-party template code and this is the only TU that instantiates
# it. MSVC's /external warning level does not cover its back-end C47XX
# diagnostics, so the one that fires there (C4701, "potentially uninitialized
# local variable") is turned off for this file alone rather than for the target.
set_source_files_properties(linear/amg_solver.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/wd4701>"
)

if(MHS_ENABLE_PARDISO)
    target_link_libraries(mhs_linear PUBLIC MKL::MKL)
    target_compile_definitions(mhs_linear PUBLIC MHS_ENABLE_PARDISO)
endif()
