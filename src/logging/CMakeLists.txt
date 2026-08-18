add_library(mhs_logging STATIC logger.cpp)
target_include_directories(mhs_logging PRIVATE ${CMAKE_SOURCE_DIR}/src)
# spdlog is PUBLIC: logger.hpp (a public header) includes <spdlog/spdlog.h>,
# so consumers like mhs_solver need the include dir transitively.
target_link_libraries(mhs_logging PUBLIC spdlog PRIVATE mhs_options)

if(VERBOSE)
    target_compile_definitions(mhs_logging PRIVATE VERBOSE)
endif()
