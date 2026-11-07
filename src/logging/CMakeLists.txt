add_library(mhs_logging STATIC logger.cpp)
target_include_directories(mhs_logging PRIVATE ${PROJECT_SOURCE_DIR}/include/metahotspot)
# spdlog is PUBLIC: logger.hpp uses its format-string argument deduction.
target_link_libraries(mhs_logging PUBLIC spdlog::spdlog PRIVATE $<BUILD_INTERFACE:mhs_options>)

set(_mhs_log_levels TRACE DEBUG INFO WARN ERROR CRITICAL OFF)
string(TOUPPER "${MHS_LOG_COMPILED_LEVEL}" _mhs_log_compiled)
list(FIND _mhs_log_levels "${_mhs_log_compiled}" _mhs_log_level)
if(_mhs_log_level EQUAL -1)
    message(FATAL_ERROR "Invalid MHS_LOG_COMPILED_LEVEL='${MHS_LOG_COMPILED_LEVEL}'; choose one of: ${_mhs_log_levels}")
endif()
# PUBLIC also ensures SDK consumers compile their log call sites to the same
# threshold as the library, including when installed through CMake EXPORT.
target_compile_definitions(mhs_logging PUBLIC MHS_LOG_COMPILED_LEVEL=${_mhs_log_level})
