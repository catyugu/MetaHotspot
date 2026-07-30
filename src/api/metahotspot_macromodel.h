#ifndef METAHOTSPOT_MACROMODEL_H
#define METAHOTSPOT_MACROMODEL_H

/*
 * MetaHotspot Macromodel Plugin C API — optional extension.
 *
 * This header must be included separately from metahotspot.h.
 * It depends on the core metahotspot.h types (mhs_compiled_t,
 * mhs_solution_t, mhs_solve_options_t).
 */

#include "api/metahotspot.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Face direction enum  (moved from core header)                      */
/* ------------------------------------------------------------------ */
typedef int32_t mhs_face_t;
enum { MHS_FACE_XM = 0, MHS_FACE_XP = 1, MHS_FACE_YM = 2, MHS_FACE_YP = 3, MHS_FACE_ZM = 4, MHS_FACE_ZP = 5 };

/* ------------------------------------------------------------------ */
/*  Non-owning CSC matrix view (macromodel-only)                       */
/* ------------------------------------------------------------------ */
typedef struct {
    int32_t rows, columns, nnz;
    const int32_t* outer_indices;
    const int32_t* inner_indices;
    const double* values;
} mhs_macro_csc_view_t;

/* ------------------------------------------------------------------ */
/*  Macromodel port model view                                        */
/* ------------------------------------------------------------------ */

/**
 * Macro K, C, f view — flat CSC arrays for both matrices plus a plain rhs vector.
 * Pointers remain valid for the duration of the mhs_macromodel_solve call.
 */
typedef struct {
    mhs_macro_csc_view_t K;
    mhs_macro_csc_view_t C;
    const double* rhs; // [n]
    size_t n; // macro state count
} mhs_macro_operators_view_t;

/**
 * Non-owning macro port model and physical-interface input.
 */
typedef struct {
    mhs_macro_operators_view_t operators;
    const double* basis; /* NULL = unit basis */
    size_t physical_port_count;
    const size_t* model_cells;
    mhs_face_t model_face;
    const double* exterior_half_conductance;
} mhs_macro_port_model_t;

/* ------------------------------------------------------------------ */
/*  Macromodel solve                                                  */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_macromodel_solve(const mhs_compiled_t* compiled, const mhs_macro_port_model_t* macro,
    const double* state, size_t state_count, const mhs_solve_options_t* opts, mhs_solution_t** out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_MACROMODEL_H */
