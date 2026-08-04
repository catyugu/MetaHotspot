#ifndef METAHOTSPOT_H
#define METAHOTSPOT_H

/*
 * MetaHotspot C API — opaque-pointer model construction, compilation, and solve.
 *
 * Three opaque handles, three life-cycle phases:
 *   mhs_model_t     — mutable construction (add layers, blocks, BCs, materials)
 *   mhs_compiled_t  — read-only compiled runtime model (reusable for repeated solves)
 *   mhs_solution_t  — read-only result (temperature field + probe traces)
 *
 * All functions that can fail return mhs_status_t; detailed error messages
 * are available via mhs_last_error(), which is thread-local and reset on
 * every API call.  Destroy functions are void and NULL-safe.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  DLL export / import                                                */
/* ------------------------------------------------------------------ */
#if defined(_WIN32) && defined(MHS_BUILD_SHARED)
#ifdef MHS_API_EXPORT
#define MHS_API __declspec(dllexport)
#else
#define MHS_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(MHS_BUILD_SHARED)
#define MHS_API __attribute__((visibility("default")))
#else
#define MHS_API
#endif

/* ------------------------------------------------------------------ */
/*  Opaque handle forward declarations                                 */
/* ------------------------------------------------------------------ */
typedef struct mhs_model_t mhs_model_t;
typedef struct mhs_compiled_t mhs_compiled_t;
typedef struct mhs_solution_t mhs_solution_t;

/* ------------------------------------------------------------------ */
/*  Enumerations                                                       */
/* ------------------------------------------------------------------ */
typedef int32_t mhs_study_t;
enum { MHS_STUDY_STEADY = 0, MHS_STUDY_TRANSIENT = 1 };

typedef int32_t mhs_length_unit_t;
enum { MHS_UNIT_METER = 0, MHS_UNIT_MILLIMETER, MHS_UNIT_MICROMETER, MHS_UNIT_NANOMETER, MHS_UNIT_INCH, MHS_UNIT_MIL };

typedef int32_t mhs_axis_t;
enum { MHS_AXIS_X = 0, MHS_AXIS_Y = 1, MHS_AXIS_Z = 2 };

typedef int32_t mhs_geometry_op_t;
enum { MHS_GEOM_ADD = 0, MHS_GEOM_SUB = 1 };

typedef int32_t mhs_solver_type_t;
enum { MHS_SOLVER_PARDISO = 0, MHS_SOLVER_EIGEN_SPARSE_LU = 1, MHS_SOLVER_EIGEN_BICGSTAB = 2 };

typedef int32_t mhs_fluid_bc_t;
enum { MHS_FLUID_NONE = 0, MHS_FLUID_PRESSURE = 1, MHS_FLUID_MASS_FLOW = 2, MHS_FLUID_VELOCITY = 3 };

typedef int32_t mhs_integrator_t;
enum { MHS_INTEGRATOR_BDF1 = 0, MHS_INTEGRATOR_BDF2 = 1 };

typedef int32_t mhs_step_strategy_t;
enum { MHS_STEP_ADAPTIVE = 0, MHS_STEP_FIXED = 1 };

typedef int32_t mhs_face_t;
enum { MHS_FACE_XM = 0, MHS_FACE_XP = 1, MHS_FACE_YM = 2, MHS_FACE_YP = 3, MHS_FACE_ZM = 4, MHS_FACE_ZP = 5 };

typedef int32_t mhs_status_t;
enum {
    MHS_OK = 0,
    MHS_ERR_NULL_PTR = -1,
    MHS_ERR_INVALID_ARG = -2,
    MHS_ERR_COMPILE = -3,
    MHS_ERR_ASSEMBLE = -4,
    MHS_ERR_SOLVE = -5,
    MHS_ERR_IO = -6,
    MHS_ERR_OOM = -7,
    MHS_ERR_RUNTIME = -8,
};

/* ------------------------------------------------------------------ */
/*  Composite types                                                    */
/* ------------------------------------------------------------------ */

/** Axis-aligned rectangle on a boundary face. */
typedef struct {
    double a_min, a_max, b_min, b_max;
} mhs_rect2d_t;

/** 2-D point (for piecewise-function knot data). */
typedef struct {
    double x, y;
} mhs_point2d_t;

/** One face region for use with atomic boundary functions. */
typedef struct {
    mhs_axis_t axis;
    double coordinate;
    mhs_rect2d_t rectangle;
} mhs_face_region_t;

/** Solver options.  Populate with mhs_solve_options_default(). */
typedef struct {
    mhs_solver_type_t solver_type;
    double linear_tolerance;
    int32_t linear_max_iterations;
    double underrelaxation;
    int32_t nonlinear_max_iterations;
    double nonlinear_relative_tolerance;
    double nonlinear_absolute_tolerance;
    int32_t integrator; // mhs_integrator_t
    int32_t step_strategy; // mhs_step_strategy_t
    double error_rel_tol;
    double error_safety;
    double min_dt;
    double max_dt;
    double fixed_dt;
} mhs_solve_options_t;

/** Compiled model metadata — scalar fields + per-cell array views.

    All pointer fields are read-only views valid while the compiled handle lives.
    For inactive grid cells, grid_to_cell[i] == SIZE_MAX.
*/
typedef struct {
    size_t cell_count;
    mhs_study_t study_type;
    double initial_temperature;

    size_t nx, ny, nz;
    const size_t* grid_to_cell; // [nx*ny*nz], SIZE_MAX for inactive cells
    const uint32_t* layer_ids; // [cell_count]
    const uint32_t* block_ids; // [cell_count]
} mhs_compiled_metadata_t;

/** Non-owning CSC matrix view — valid until the source handle is destroyed. */
typedef struct {
    int32_t rows, columns, nnz;
    const int32_t* outer_indices;
    const int32_t* inner_indices;
    const double* values;
} mhs_csc_view_t;

/** Operators K, C, f of the linearised system: C * dx/dt + K * x = f. */
typedef struct {
    mhs_csc_view_t K;
    mhs_csc_view_t C;
    const double* rhs; // [n]
    size_t n; // state count
} mhs_operators_t;

/** Solution bulk data view — valid until the solution handle is destroyed. */
typedef struct {
    size_t fvm_count;
    size_t state_count;
    double time;
    const double* state; // [state_count], temperatures first
} mhs_solution_view_t;

/** Read-only row-major output history owned by mhs_solution_t.

    states[record * state_count + state] is the state value at times[record].
    The view remains valid until the solution handle is destroyed.
*/
typedef struct {
    const double* times;
    const double* states;
    size_t record_count;
    size_t state_count;
} mhs_solution_history_view_t;

/** Non-owning probe trace view — valid while the solution handle is alive. */
typedef struct {
    const char* name;
    const double* times; // [record_count]
    const double* values; // [record_count]
    size_t record_count;
} mhs_probe_view_t;

/* ------------------------------------------------------------------ */
/*  Global helpers                                                     */
/* ------------------------------------------------------------------ */

/** Fill opts with sensible defaults (Pardiso, 1e-8, 1e-6, …). */
MHS_API void mhs_solve_options_default(mhs_solve_options_t* opts);

/** Thread-local last error message. Reset on every API call. */
MHS_API const char* mhs_last_error(void);

/* ------------------------------------------------------------------ */
/*  Model life-cycle                                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_create(mhs_model_t** out);
MHS_API void mhs_model_destroy(mhs_model_t* m);
MHS_API mhs_status_t mhs_model_read_xml(mhs_model_t* m, const char* path);

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_settings(mhs_model_t* m, mhs_study_t study, mhs_length_unit_t length_unit,
    double initial_temperature_K, double duration, double output_interval);

MHS_API mhs_status_t mhs_model_set_mesh(
    mhs_model_t* m, size_t nx, const double* x, size_t ny, const double* y, size_t nz, const double* z);

MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression);

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, layers, blocks, rects           */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity);

MHS_API mhs_status_t mhs_model_add_layer(
    mhs_model_t* m, const char* thickness, const char* x_offset, const char* y_offset, uint32_t* out_id);

MHS_API mhs_status_t mhs_model_add_block(mhs_model_t* m, uint32_t layer, const char* material_name,
    const char* heat_source, const char* x_offset, const char* y_offset, const char* thickness, uint32_t* out_id);

MHS_API mhs_status_t mhs_model_add_rect(mhs_model_t* m, uint32_t block, mhs_geometry_op_t op, const char* x,
    const char* y, const char* width, const char* height);

/* ------------------------------------------------------------------ */
/*  Model construction  —  atomic boundary conditions                 */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_dirichlet(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* temperature);

MHS_API mhs_status_t mhs_model_add_neumann(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* heat_flux);

MHS_API mhs_status_t mhs_model_add_convection(mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions,
    const char* coefficient, const char* ambient_temperature);

MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature);
MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux);
MHS_API mhs_status_t mhs_model_set_default_convection(
    mhs_model_t* m, const char* coefficient, const char* ambient_temperature);

/* ------------------------------------------------------------------ */
/*  Model construction  —  function library                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression);
MHS_API mhs_status_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center);
MHS_API mhs_status_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase);
MHS_API mhs_status_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta);
MHS_API mhs_status_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, size_t count);
MHS_API mhs_status_t mhs_model_add_function_periodic_piecewise_constant(
    mhs_model_t* m, const char* name, const double* values, size_t count, double period);

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z);
MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_axis_t axis, double coordinate,
    mhs_rect2d_t region, mhs_fluid_bc_t kind, double value, double inlet_temperature);

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(const mhs_model_t* m, mhs_compiled_t** out);
MHS_API void mhs_compiled_destroy(mhs_compiled_t* c);

/* ------------------------------------------------------------------ */
/*  Compiled metadata                                                  */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_metadata(const mhs_compiled_t* c, mhs_compiled_metadata_t* out);

/* ------------------------------------------------------------------ */
/*  Assembly                                                            */
/* ------------------------------------------------------------------ */

/** Evaluate K, C, f at *temperature* at time *time*.
    Returns an mhs_operators_t valid until the next assemble call. */
MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* temperature, size_t temperature_count, double time, mhs_operators_t* out);

/* ------------------------------------------------------------------ */
/*  Half-conductance  k*A/(dx/2)                                       */
/* ------------------------------------------------------------------ */

/** Compute k*A/(dx/2) for each cell at the given face.
 *  Temperature and time are used to evaluate material properties. */
MHS_API mhs_status_t mhs_compiled_half_conductance(const mhs_compiled_t* c, const size_t* cells, mhs_face_t face,
    double temperature, double time, double* out, size_t n);

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const double* state, size_t state_count,
    const mhs_solve_options_t* opts, mhs_solution_t** out);

MHS_API void mhs_solution_destroy(mhs_solution_t* s);

/* ------------------------------------------------------------------ */
/*  VTU export                                                         */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_write_vtu(const mhs_compiled_t* c, const mhs_solution_t* s, const char* path);

/* ------------------------------------------------------------------ */
/*  Solution views                                                     */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_view(const mhs_solution_t* s, mhs_solution_view_t* out);
MHS_API mhs_status_t mhs_solution_history_view(const mhs_solution_t* solution, mhs_solution_history_view_t* out);

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API size_t mhs_solution_probe_count(const mhs_solution_t* s);
MHS_API mhs_status_t mhs_solution_probe_view(const mhs_solution_t* s, size_t index, mhs_probe_view_t* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_H */
