#pragma once

/* Internal opaque handle definitions shared between metahotspot.cpp and
   metahotspot_macromodel.cpp.  Not part of the public API. */

#include "api/metahotspot.h" // mhs_solve_options_t etc.
#include "mhs/model_definition.hpp"
#include "mhs/solver.hpp" // mhs::sim::SolveOptions
#include "solver/assembler.hpp" // mhs::sim::Operators (assemble scratch)
#include <Eigen/Sparse>
#include <cstdint>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Shared error-buffer helpers                                        */
/* ------------------------------------------------------------------ */

void mhs_detail_set_last_error(const std::string& msg);
void mhs_detail_clear_last_error();
const char* mhs_detail_last_error();

/* ------------------------------------------------------------------ */
/*  Shared error-handling macros                                       */
/* ------------------------------------------------------------------ */

#define SET_ERR(msg)                                                                                                   \
    do {                                                                                                               \
        std::ostringstream _oss;                                                                                       \
        _oss << msg;                                                                                                   \
        mhs_detail_set_last_error(_oss.str());                                                                         \
    } while (0)

#define CHECK_NULL(p)                                                                                                  \
    do {                                                                                                               \
        if (!(p)) {                                                                                                    \
            SET_ERR("NULL pointer: " #p);                                                                              \
            return MHS_ERR_NULL_PTR;                                                                                   \
        }                                                                                                              \
    } while (0)

#define MHS_TRY(err_code, ...)                                                                                         \
    try {                                                                                                              \
        mhs_detail_clear_last_error();                                                                                 \
        __VA_ARGS__;                                                                                                   \
        mhs_detail_clear_last_error();                                                                                 \
        return MHS_OK;                                                                                                 \
    }                                                                                                                  \
    catch (const std::exception& e) {                                                                                  \
        SET_ERR(e.what());                                                                                             \
        return err_code;                                                                                               \
    }

/* ------------------------------------------------------------------ */
/*  Shared helper declarations                                         */
/* ------------------------------------------------------------------ */

/** Convert C-level solver opts to SolveOptions. */
mhs::sim::SolveOptions to_solve_options(const mhs_solve_options_t* opts, double transient_duration = 0.0);

/* ------------------------------------------------------------------ */
/*  Opaque handle structs  (exposed to both core and macromodel TUs)   */
/* ------------------------------------------------------------------ */

struct BlockLocation {
    uint32_t layer;
    uint32_t block;
};

struct mhs_model_t {
    mhs::model::ModelDefinition def;
    std::vector<BlockLocation> block_locations;
};

struct mhs_compiled_t {
    mhs::core::Model model;
    mhs::sim::Operators assemble_scratch; // reused across calls, not thread-safe
};

struct mhs_solution_t {
    mhs::core::Solution sol;
};
