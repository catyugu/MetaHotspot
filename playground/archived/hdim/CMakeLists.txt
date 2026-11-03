# HDI reproduction driver, built against the repository's library for the linear
# solvers only; the grid, the assembly and the decomposition are local to this
# directory.
add_executable(mhs_hdim
    hdi.cpp
    main.cpp
    reference.cpp
)

target_link_libraries(mhs_hdim PRIVATE
    mhs::metahotspot
    Eigen3::Eigen
    TBB::tbb
)
target_compile_features(mhs_hdim PRIVATE cxx_std_20)

mhs_copy_runtime_dlls(mhs_hdim)
