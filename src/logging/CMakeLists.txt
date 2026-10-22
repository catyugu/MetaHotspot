add_library(mhs_logging STATIC logger.cpp)
target_include_directories(mhs_logging PRIVATE ${PROJECT_SOURCE_DIR}/include/metahotspot)
# spdlog is PUBLIC: logger.hpp is a public header that includes
# <spdlog/spdlog.h> to derive its format-string arguments.
target_link_libraries(mhs_logging PUBLIC spdlog::spdlog PRIVATE $<BUILD_INTERFACE:mhs_options>)

if(MHS_VERBOSE)
    target_compile_definitions(mhs_logging PRIVATE MHS_VERBOSE)
endif()
