add_library(mhs_core STATIC memory.cpp vector_ops.cpp csr_ops.cpp)
target_link_libraries(mhs_core PRIVATE $<BUILD_INTERFACE:mhs_options>)
