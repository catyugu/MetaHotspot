#ifndef METAHOTSPOT_MACROMODEL_H
#define METAHOTSPOT_MACROMODEL_H

/* MetaHotspot DtN macromodel extension. */
#include "api/metahotspot.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mhs_macro_port_map_t mhs_macro_port_map_t;

/** One geometric boundary patch and therefore one physical DtN port.
 *  coordinate and rectangle coordinates use the compiled model's SI units.
 *  For X faces rectangle=(y,z), for Y faces=(x,z), for Z faces=(x,y).
 */
typedef struct {
    mhs_face_t face;
    double coordinate;
    mhs_rect2d_t rectangle;
} mhs_macro_port_patch_t;

/** Reduced DtN model. basis is row-major [physical_port_count x operators.n]. */
typedef struct {
    mhs_operators_t operators;
    const double* basis; /* NULL = identity */
    size_t physical_port_count;
} mhs_macro_dtn_model_t;

/** Compile geometric patches against a compiled model. */
MHS_API mhs_status_t mhs_macromodel_port_map_create(const mhs_compiled_t* compiled,
    const mhs_macro_port_patch_t* patches, size_t patch_count, mhs_macro_port_map_t** out);
MHS_API void mhs_macromodel_port_map_destroy(mhs_macro_port_map_t* map);
MHS_API size_t mhs_macromodel_port_count(const mhs_macro_port_map_t* map);

/** Assemble an isolated component as [physical ports, FVM cell states]. */
MHS_API mhs_status_t mhs_macromodel_assemble_dtn(const mhs_compiled_t* compiled,
    const mhs_macro_port_map_t* ports, const double* state, size_t state_count, double time,
    mhs_operators_t* out);

/** Solve an FVM model coupled to a reduced DtN model. */
MHS_API mhs_status_t mhs_macromodel_solve(const mhs_compiled_t* compiled,
    const mhs_macro_port_map_t* ports, const mhs_macro_dtn_model_t* dtn,
    const double* state, size_t state_count, const mhs_solve_options_t* opts, mhs_solution_t** out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_MACROMODEL_H */
