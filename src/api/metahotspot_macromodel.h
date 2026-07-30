#ifndef METAHOTSPOT_MACROMODEL_H
#define METAHOTSPOT_MACROMODEL_H

/*
 * MetaHotspot Macromodel Plugin C API — optional extension.
 *
 * This header must be included separately from metahotspot.h.
 * It depends on the core metahotspot.h types (mhs_operators_view_t,
 * mhs_face_t, mhs_compiled_t, mhs_solution_t, mhs_solve_options_t).
 *
 * The plugin is built as a separate shared library (mhs_macromodel_c_api).
 */

#include "api/metahotspot.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Macromodel port model view                                        */
/* ------------------------------------------------------------------ */

/**
 * Non-owning macro port model and physical-interface input.
 *
 * - `operators` : the macro K, C, f with dimension n × n (macro state).
 * - `basis`     : row-major [physical_port_count × n] matrix; NULL = unit basis.
 * - `physical_port_count` : number of physical interface ports.
 * - `model_cells` : FVM cell index for each physical port.
 * - `model_face` : interface face direction.
 * - `exterior_half_conductance` : [physical_port_count] — macro-side half conductance.
 */
typedef struct {
    mhs_operators_view_t operators;
    const double* basis;                    /* NULL = unit basis */
    size_t physical_port_count;
    const size_t* model_cells;
    mhs_face_t model_face;
    const double* exterior_half_conductance;
} mhs_macro_port_model_t;

/* ------------------------------------------------------------------ */
/*  Macromodel solve                                                  */
/* ------------------------------------------------------------------ */

/**
 * Solve an FVM model coupled to a macro port model.
 *
 * The FVM-side interface conductance is reevaluated from the current
 * nonlinear iterate. When basis is NULL, the macro state dimension
 * must equal physical_port_count (unit basis). When basis is non-NULL,
 * its column count equals operators.n.
 *
 * `state_count` must equal compiled_cell_count + operators.n.
 */
MHS_API mhs_status_t mhs_macromodel_solve(const mhs_compiled_t* compiled,
    const mhs_macro_port_model_t* macro, const double* state, size_t state_count,
    const mhs_solve_options_t* opts, mhs_solution_t** out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_MACROMODEL_H */
