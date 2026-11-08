set(_linear_sources
    solver.cpp
    amg_solver.cpp
    sparse_lu_solver.cpp
)
if(MHS_ENABLE_PARDISO)
    list(APPEND _linear_sources pardiso_lu_solver.cpp)
endif()

add_library(mhs_linear STATIC ${_linear_sources})
target_include_directories(mhs_linear PRIVATE ${PROJECT_SOURCE_DIR}/include/metahotspot)
# The builtin backend is what this project uses, whether or not a machine
# happens to have Boost installed.
target_compile_definitions(mhs_linear PRIVATE AMGCL_NO_BOOST)
target_link_libraries(mhs_linear
    PUBLIC mhs_core
    PRIVATE
    $<BUILD_INTERFACE:mhs_options>
    $<BUILD_INTERFACE:amgcl::amgcl>
    OpenMP::OpenMP_CXX
    $<BUILD_INTERFACE:Eigen3::Eigen>
)

# amgcl is third-party template code and this is the only TU that instantiates
# it. MSVC's /external warning level does not cover its back-end C47XX
# diagnostics, so the one that fires there (C4701, "potentially uninitialized
# local variable") is turned off for this file alone rather than for the target.
set_source_files_properties(amg_solver.cpp PROPERTIES
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:MSVC>:/wd4701>"
)

if(MHS_ENABLE_PARDISO)
    target_link_libraries(mhs_linear PRIVATE $<BUILD_INTERFACE:Eigen3::Eigen> MKL::MKL)
    target_compile_definitions(mhs_linear PRIVATE MHS_ENABLE_PARDISO)
endif()
