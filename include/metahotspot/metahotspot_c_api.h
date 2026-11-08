#ifndef METAHOTSPOT_C_API_H
#define METAHOTSPOT_C_API_H

/*
 * MetaHotspot C API — opaque-pointer model construction, compilation, and solve.
 *
 * Four independently owned opaque handles:
 *   mhs_model_t     — mutable construction (add volumes, sources, BCs, materials)
 *   mhs_compiled_t  — read-only compiled runtime model (reusable for repeated solves)
 *   mhs_solution_t  — read-only result (temperature field + probe traces)
 *   mhs_operators_t — owned K/C/f snapshot
 *
 * All functions that can fail return mhs_status_t; detailed error messages
 * are available via mhs_last_error(), which is thread-local and reset on
 * each successful status-returning call. Other calls preserve the error.
 * Destroy functions are void and NULL-safe. Create/compile/assemble/solve
 * initialize a non-NULL output handle address to NULL before validation.
 * Input arrays/strings are copied; outputs use caller-owned exact-size buffers.
 * Zero-size output arrays may be NULL. Handles are independent; destroying a
 * model or compiled handle does not invalidate its existing snapshots.
 * Concurrent use of a mutable handle requires caller synchronization.
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
/*  Version                                                            */
/* ------------------------------------------------------------------ */
/* The single source of the library version: CMake reads project(VERSION)
 * from these macros, the Python package takes its metadata version from
 * them, and mhs_version_string() reports them at run time. */
#define MHS_VERSION_MAJOR 0
#define MHS_VERSION_MINOR 0
#define MHS_VERSION_PATCH 0

#define MHS_VERSION_STRINGIFY_(major, minor, patch) #major "." #minor "." #patch
#define MHS_VERSION_STRINGIFY(major, minor, patch) MHS_VERSION_STRINGIFY_(major, minor, patch)
#define MHS_VERSION_STRING MHS_VERSION_STRINGIFY(MHS_VERSION_MAJOR, MHS_VERSION_MINOR, MHS_VERSION_PATCH)

/* ------------------------------------------------------------------ */
/*  Opaque handle forward declarations                                 */
/* ------------------------------------------------------------------ */
/** Fixed-width entity references and signed CSR storage indices. */
typedef uint32_t mhs_index_t;
typedef int32_t mhs_sparse_index_t;
#define MHS_INVALID_INDEX UINT32_MAX

typedef struct mhs_model_t mhs_model_t;
typedef struct mhs_compiled_t mhs_compiled_t;
typedef struct mhs_solution_t mhs_solution_t;
typedef struct mhs_operators_t mhs_operators_t;

/* ------------------------------------------------------------------ */
/*  Enumerations                                                       */
/* ------------------------------------------------------------------ */
typedef int32_t mhs_study_t;
enum { MHS_STUDY_STEADY = 0, MHS_STUDY_TRANSIENT = 1 };

typedef int32_t mhs_axis_t;
enum { MHS_AXIS_X = 0, MHS_AXIS_Y = 1, MHS_AXIS_Z = 2 };

typedef int32_t mhs_geometry_op_t;
enum { MHS_GEOM_ADD = 0, MHS_GEOM_SUB = 1 };

typedef int32_t mhs_solver_type_t;
enum { MHS_SOLVER_PARDISO = 0, MHS_SOLVER_AMG = 1 };

typedef int32_t mhs_fluid_bc_t;
enum { MHS_FLUID_PRESSURE_INLET = 1, MHS_FLUID_MASS_FLOW_INLET = 2, MHS_FLUID_OUTLET = 3 };

typedef int32_t mhs_status_t;
enum {
    MHS_OK = 0,
    MHS_ERROR = -1,
};

typedef int32_t mhs_log_level_t;
enum {
    MHS_LOG_LEVEL_TRACE = 0,
    MHS_LOG_LEVEL_DEBUG = 1,
    MHS_LOG_LEVEL_INFO = 2,
    MHS_LOG_LEVEL_WARN = 3,
    MHS_LOG_LEVEL_ERROR = 4,
    MHS_LOG_LEVEL_CRITICAL = 5,
    MHS_LOG_LEVEL_OFF = 6
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

/** One continuous SI rectangle; tangential axes are X:(y,z), Y:(x,z), Z:(x,y). */
typedef struct {
    mhs_axis_t axis;
    double coordinate;
    mhs_rect2d_t rectangle;
} mhs_surface_region_t;

/** Solver options.  Populate with mhs_solve_options_default(). */
typedef struct {
    mhs_solver_type_t solver_type;
    double linear_tolerance;
    int32_t linear_max_iterations;
    double underrelaxation;
    int32_t nonlinear_max_iterations;
    double nonlinear_relative_tolerance;
    double nonlinear_absolute_tolerance;
    double time_tolerance;
    double max_time_step; /**< Seconds; zero selects automatic. */
} mhs_solve_options_t;

/** Explicit compiled geometry sizes. State order is CellId order. */
typedef struct {
    size_t cell_count, face_count, projection_count, source_count;
    mhs_study_t study_type;
    double initial_temperature;
} mhs_compiled_info_t;

/** Caller-owned arrays. XYZ bounds are flattened (count,3), all IDs zero-based.
 * Face bounds are (face_count,4) in ascending
 * tangential axes: YZ for X, XZ for Y, XY for Z, each [a_min,a_max,b_min,b_max].
 * Positive flux runs face_minus -> face_plus along face_axis. Exterior patches
 * have exactly one MHS_INVALID_INDEX cell; the missing side determines the normal. */
typedef struct {
    size_t cell_count, face_count, projection_count;
    double* cell_min;
    double* cell_max;
    mhs_index_t* material_id;
    mhs_index_t* face_minus;
    mhs_index_t* face_plus;
    mhs_axis_t* face_axis;
    double* face_bounds;
    mhs_index_t* projection_cell;
    mhs_index_t* projection_source;
    double* projection_volume;
    double* projection_center;
} mhs_cell_fields_t;

/** Evaluate power in each geometric volume-projection term at supplied state/time. */
MHS_API mhs_status_t mhs_compiled_eval_source_terms(const mhs_compiled_t* c, const double* state, size_t state_count,
    double time, double* powers, size_t projection_count);

/** Caller-owned buffers for material values evaluated at a supplied state/time. */
typedef struct {
    double* conductivity_x;
    double* conductivity_y;
    double* conductivity_z;
    double* density;
    double* specific_heat;
    size_t count;
} mhs_material_values_t;

/** Sizes for an owned operator handle. K and C are square state_count matrices. */
typedef struct {
    size_t state_count;
    size_t k_nnz;
    size_t c_nnz;
} mhs_operators_info_t;

/** Scalar solution metadata. Array data is retrieved with copy functions. */
typedef struct {
    size_t state_count;
    size_t record_count;
    size_t probe_count;
    double time;
    int32_t converged; /**< 1 on convergence, 0 otherwise. */
} mhs_solution_info_t;

/* ------------------------------------------------------------------ */
/*  Global helpers                                                     */
/* ------------------------------------------------------------------ */

/** Fill opts with sensible defaults (AMG, 1e-8, 1e-6, …). */
MHS_API void mhs_solve_options_default(mhs_solve_options_t* opts);

/** Runtime logging control (global to the process). */
MHS_API mhs_status_t mhs_log_set_level(mhs_log_level_t level);
MHS_API mhs_log_level_t mhs_log_get_level(void);
/** Library version, e.g. "1.0.0". */
MHS_API const char* mhs_version_string(void);

/** Thread-local error; valid until the next status-returning call on this thread. */
MHS_API const char* mhs_last_error(void);

/* ------------------------------------------------------------------ */
/*  Model life-cycle                                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_create(mhs_model_t** out);
MHS_API void mhs_model_destroy(mhs_model_t* m);

/* ------------------------------------------------------------------ */
/*  Model construction  —  study, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_study(
    mhs_model_t* m, mhs_study_t study, double initial_temperature_K, double duration, double output_interval);

/** Cartesian vertex arrays in SI metres; each axis must be finite, strictly
 * increasing and contain at least two entries. All axes are replaced atomically. */
MHS_API mhs_status_t mhs_model_set_cartesian_mesh(
    mhs_model_t* m, size_t nx, const double* x, size_t ny, const double* y, size_t nz, const double* z);

MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression);

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, sources, volumes           */
/* ------------------------------------------------------------------ */

/** NULL dynamic_viscosity selects a solid; a non-NULL expression selects a fluid. */
MHS_API mhs_status_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity, mhs_index_t* out_id);

/** Geometry, mesh coordinates and probes use SI metres. Expressions use canonical x/y/z/T/t.
 * Later volumes override earlier volumes. MHS_INVALID_INDEX omits the heat source. */
MHS_API mhs_status_t mhs_model_add_source(mhs_model_t* m, const char* expression, mhs_index_t* out_id);
MHS_API mhs_status_t mhs_model_add_volume(
    mhs_model_t* m, mhs_index_t material, mhs_index_t source, mhs_index_t* out_id);
MHS_API mhs_status_t mhs_model_volume_add_box(
    mhs_model_t* m, mhs_index_t volume, mhs_geometry_op_t op, const double min_xyz[3], const double max_xyz[3]);

/* ------------------------------------------------------------------ */
/*  Model construction  —  atomic boundary conditions                 */
/* ------------------------------------------------------------------ */

/** Surface conditions require a nonempty array of SI rectangles. Later regions
 * override earlier ones; uncovered exterior faces use the default thermal BC. */
MHS_API mhs_status_t mhs_model_add_dirichlet(
    mhs_model_t* m, const mhs_surface_region_t* regions, size_t n_regions, const char* temperature);

MHS_API mhs_status_t mhs_model_add_neumann(
    mhs_model_t* m, const mhs_surface_region_t* regions, size_t n_regions, const char* heat_flux);

MHS_API mhs_status_t mhs_model_add_convection(mhs_model_t* m, const mhs_surface_region_t* regions, size_t n_regions,
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

/** Named point probe, with three finite SI coordinates copied into the model. */
MHS_API mhs_status_t mhs_model_add_point_probe(mhs_model_t* m, const char* name, const double xyz[3]);
/* Same region layout as thermal boundaries. NULL inlet_temperature selects the
 * initial temperature; outlets ignore it. value is pressure (Pa), or mass flow (kg/s). */
MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, const mhs_surface_region_t* regions, size_t n_regions,
    mhs_fluid_bc_t kind, double value, const double* inlet_temperature);

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(const mhs_model_t* m, mhs_compiled_t** out);
MHS_API void mhs_compiled_destroy(mhs_compiled_t* c);

/* ------------------------------------------------------------------ */
/*  Compiled metadata                                                  */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_get_info(const mhs_compiled_t* c, mhs_compiled_info_t* out);
MHS_API mhs_status_t mhs_compiled_copy_cell_fields(const mhs_compiled_t* c, mhs_cell_fields_t* fields);
MHS_API mhs_status_t mhs_compiled_eval_materials(const mhs_compiled_t* c, const double* temperature,
    size_t temperature_count, double time, mhs_material_values_t* values);

/* ------------------------------------------------------------------ */
/*  Assembly                                                            */
/* ------------------------------------------------------------------ */

/** Evaluate K, C, f at *temperature* (cell_count values) and return an owned snapshot. */
MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* temperature, size_t temperature_count, double time, mhs_operators_t** out);

MHS_API void mhs_operators_destroy(mhs_operators_t* operators);
MHS_API mhs_status_t mhs_operators_get_info(const mhs_operators_t* operators, mhs_operators_info_t* out);
/** Export native CSR: row_ptr[state_count+1], col_idx[nnz], values[nnz]. */
MHS_API mhs_status_t mhs_operators_copy_k(const mhs_operators_t* operators, mhs_sparse_index_t* row_ptr,
    size_t row_ptr_count, mhs_sparse_index_t* col_idx, size_t col_idx_count, double* values, size_t value_count);
MHS_API mhs_status_t mhs_operators_copy_c(const mhs_operators_t* operators, mhs_sparse_index_t* row_ptr,
    size_t row_ptr_count, mhs_sparse_index_t* col_idx, size_t col_idx_count, double* values, size_t value_count);
MHS_API mhs_status_t mhs_operators_copy_rhs(const mhs_operators_t* operators, double* out, size_t count);

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

/** state is NULL/0 for model defaults, otherwise exactly cell_count entries.
 * opts may be NULL to select defaults. Check solution.converged separately. */
MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const double* state, size_t state_count,
    const mhs_solve_options_t* opts, mhs_solution_t** out);

MHS_API void mhs_solution_destroy(mhs_solution_t* s);

/* ------------------------------------------------------------------ */
/*  Solution copy-out accessors                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_get_info(const mhs_solution_t* s, mhs_solution_info_t* out);
MHS_API mhs_status_t mhs_solution_copy_state(const mhs_solution_t* s, double* out, size_t count);
MHS_API mhs_status_t mhs_solution_copy_history_times(const mhs_solution_t* s, double* out, size_t count);
MHS_API mhs_status_t mhs_solution_copy_history_states(const mhs_solution_t* s, double* out, size_t count);

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_probe_get_info(
    const mhs_solution_t* s, size_t index, size_t* name_size, size_t* record_count);
MHS_API mhs_status_t mhs_solution_copy_probe(
    const mhs_solution_t* s, size_t index, char* name, size_t name_size, double* values, size_t record_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_C_API_H */
