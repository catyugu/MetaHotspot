add_library(mhs_expression STATIC expression.cpp)
target_link_libraries(mhs_expression PRIVATE
    mhs_core
    $<BUILD_INTERFACE:mhs_options>
    TBB::tbb
    muparser::muparser
)
