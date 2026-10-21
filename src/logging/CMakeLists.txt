add_library(mhs_logging STATIC logger.cpp)
target_include_directories(mhs_logging PRIVATE ${PROJECT_SOURCE_DIR}/src)
# spdlog is PUBLIC: logger.hpp is an installed header that includes
# <spdlog/spdlog.h> to derive its format-string arguments.
target_link_libraries(mhs_logging PUBLIC spdlog::spdlog PRIVATE $<BUILD_INTERFACE:mhs_options>)
mhs_install_headers(logging logger.hpp)

if(MHS_VERBOSE)
    target_compile_definitions(mhs_logging PRIVATE MHS_VERBOSE)
endif()
