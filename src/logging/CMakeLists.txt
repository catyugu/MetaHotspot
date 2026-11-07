add_library(mhs_logging STATIC logger.cpp)
target_include_directories(mhs_logging PRIVATE ${PROJECT_SOURCE_DIR}/include/metahotspot)
# spdlog is PUBLIC: logger.hpp exposes its format-string types.
target_link_libraries(mhs_logging PUBLIC spdlog::spdlog PRIVATE $<BUILD_INTERFACE:mhs_options>)
# All translation units, including installed SDK consumers, agree on the cut-off.
target_compile_definitions(mhs_logging PUBLIC MHS_LOG_COMPILED_LEVEL=${_MHS_LOG_COMPILED_LEVEL})
