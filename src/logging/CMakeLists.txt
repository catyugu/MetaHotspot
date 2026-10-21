add_library(mhs_logging STATIC logger.cpp)
target_include_directories(mhs_logging PRIVATE ${PROJECT_SOURCE_DIR}/src)
# spdlog is PUBLIC: logger.hpp (a public header) includes <spdlog/spdlog.h>,
# so consumers like mhs_solver need the include dir transitively.
target_link_libraries(mhs_logging PUBLIC spdlog::spdlog PRIVATE $<BUILD_INTERFACE:mhs_options>)

if(MHS_VERBOSE)
    target_compile_definitions(mhs_logging PRIVATE MHS_VERBOSE)
endif()
