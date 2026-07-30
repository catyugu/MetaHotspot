#pragma once

/* Internal opaque handle definitions shared between the core C API and
   optional extension modules (e.g. macromodel). Not part of the public API. */

#include "api/metahotspot.h"       // mhs_solve_options_t etc.
#include "model/model_definition.hpp"
#include "runtime/model.hpp"
#include "runtime/solution.hpp"
#include "solver/scheduler.hpp"      // mhs::sim::SolveOptions
#include <Eigen/Sparse>
#include <cstdint>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Shared error-buffer helpers                                        */
/* ------------------------------------------------------------------ */

/** Set the thread-local last-error string (defined in metahotspot.cpp). */
void mhs_detail_set_last_error(const std::string& msg);

/** Clear the thread-local last-error string. */
void mhs_detail_clear_last_error();

/** Get the thread-local last-error string. */
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

#define MHS_TRY_ID(invalid, ...)                                                                                       \
    try {                                                                                                              \
        mhs_detail_clear_last_error();                                                                                 \
        __VA_ARGS__;                                                                                                   \
    }                                                                                                                  \
    catch (const std::exception& e) {                                                                                  \
        SET_ERR(e.what());                                                                                             \
        return invalid;                                                                                                \
    }

/* ------------------------------------------------------------------ */
/*  Shared helper declarations                                         */
/* ------------------------------------------------------------------ */

/** Convert C-level solver opts to C++ SolveOptions (defined in metahotspot.cpp). */
mhs::sim::SolveOptions to_solve_options(const mhs_solve_options_t* opts, double transient_duration = 0.0);

/* ------------------------------------------------------------------ */
/*  Opaque handle structs                                              */
/* ------------------------------------------------------------------ */

struct BlockLocation {
    uint32_t layer;
    uint32_t block;
};

struct mhs_model_t {
    mhs::model::ModelDefinition def;
    std::vector<BlockLocation> block_locations;
};

struct mhs_operators_t {
    Eigen::SparseMatrix<double> K;
    Eigen::SparseMatrix<double> C;
    Eigen::VectorXd rhs;
};

struct mhs_compiled_t {
    mhs::core::Model model;
};

struct mhs_solution_t {
    mhs::core::Solution sol;
};
