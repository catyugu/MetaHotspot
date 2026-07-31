#ifndef METAHOTSPOT_SOLUTION_HISTORY_H
#define METAHOTSPOT_SOLUTION_HISTORY_H

#include "api/metahotspot.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Read-only row-major output history owned by mhs_solution_t.
 *
 * states[record * state_count + state] is the state value at times[record].
 * The view remains valid until the solution handle is destroyed.
 */
typedef struct {
    const double* times;
    const double* states;
    size_t record_count;
    size_t state_count;
} mhs_solution_history_view_t;

MHS_API mhs_status_t mhs_solution_history_view(
    const mhs_solution_t* solution, mhs_solution_history_view_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_SOLUTION_HISTORY_H */
