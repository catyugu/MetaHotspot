mhs_add_library(expr_lib
    SOURCES expr.cpp
    PRIVATE_LINKS TBB::tbb muparser
)
