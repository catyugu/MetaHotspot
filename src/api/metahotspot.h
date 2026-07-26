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
 * All mutation functions return mhs_status_t.  Functions returning an ID return
 * MHS_*_ID_INVALID on failure.  Detailed error messages are available via
 * mhs_last_error(), which is thread-local and reset on every API call.
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
typedef struct mhs_assembly_t mhs_assembly_t;

/* ------------------------------------------------------------------ */
/*  ID types  (uint32_t, UINT32_MAX = invalid)                         */
/* ------------------------------------------------------------------ */
typedef uint32_t mhs_layer_id_t;
typedef uint32_t mhs_block_id_t;
typedef uint32_t mhs_material_id_t;
typedef uint32_t mhs_function_id_t;
typedef uint32_t mhs_probe_id_t;

#define MHS_LAYER_ID_INVALID UINT32_MAX
#define MHS_BLOCK_ID_INVALID UINT32_MAX
#define MHS_MATERIAL_ID_INVALID UINT32_MAX
#define MHS_FUNCTION_ID_INVALID UINT32_MAX
#define MHS_PROBE_ID_INVALID UINT32_MAX

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

typedef int32_t mhs_operator_t;
enum { MHS_OPERATOR_STIFFNESS = 0, MHS_OPERATOR_CAPACITY = 1 };

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
    MHS_ERR_UNSET = -8,
    MHS_ERR_RUNTIME = -9,
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

/** Solver options.  Populate with mhs_solver_opts_default(). */
typedef struct {
    mhs_solver_type_t solver_type;
    double linear_tolerance;
    int32_t linear_max_iterations;
    double underrelaxation;
    int32_t nonlinear_max_iterations;
    double nonlinear_relative_tolerance;
    double nonlinear_absolute_tolerance;
} mhs_solver_opts_t;

/** Non-owning CSC matrix view.  Pointers remain valid while the source
 *  assembly handle remains alive. */
typedef struct {
    int32_t rows, columns, nnz;
    const int32_t* outer_indices;
    const int32_t* inner_indices;
    const double* values;
} mhs_csc_view_t;

/** Diagnostics returned by mhs_compiled_step(). */
typedef struct {
    int32_t accepted; // 0/1
    double error_ratio;
    double suggested_dt_factor;
    int32_t nonlinear_iterations;
} mhs_step_info_t;

/**
 * Compiled model metadata view — replaces ~12 individual accessor functions.
 *
 * All pointer fields are valid until the compiled model is destroyed.
 * Grid index: linear_idx = ix * (ny * nz) + iy * nz + iz
 * grid_to_cell entry == SIZE_MAX means inactive (hole/void).
 */
typedef struct {
    size_t cell_count;
    size_t state_count;
    size_t node_count;
    size_t grid_count; // nx * ny * nz
    mhs_study_t study_type;
    double initial_temperature;
    const uint32_t* layer_ids; // [cell_count] post-processing only
    const uint32_t* block_ids; // [cell_count] post-processing only
    const size_t* grid_to_cell; // [grid_count]
    size_t nx, ny, nz;
} mhs_compiled_metadata_t;

/**
 * Solution bulk data view.
 *
 * All pointer fields are valid until the solution handle is destroyed.
 * cell_temperatures is the cell-centroid field [cell_count].
 * states is the full DOF state vector [state_count].
 */
typedef struct {
    size_t cell_count;
    size_t state_count;
    double time;
    const double* cell_temperatures; // [cell_count]
    const double* states; // [state_count]
} mhs_solution_view_t;

/**
 * Probe metadata — names and record counts.
 *
 * Populated by mhs_solution_probe_metadata().  Must be freed via
 * mhs_solution_probe_metadata_free() to release heap-allocated arrays.
 */
typedef struct {
    size_t count;
    const char* const* names; // [count]
    const size_t* record_counts; // [count]
} mhs_probe_metadata_t;

/* ------------------------------------------------------------------ */
/*  Global helpers                                                     */
/* ------------------------------------------------------------------ */

/** Fill opts with sensible defaults (Pardiso, 1e-8, 1e-6, …). */
MHS_API void mhs_solver_opts_default(mhs_solver_opts_t* opts);

/** Human-readable name for a status code (static, no ownership transfer). */
MHS_API const char* mhs_status_string(mhs_status_t status);

/** Thread-local last error message from the most recent API call that
 *  returned a non-OK status (or an INVALID ID).  Reset on every API call.
 *  Valid until the next API call. */
MHS_API const char* mhs_last_error(void);

/* ------------------------------------------------------------------ */
/*  Model life-cycle                                                   */
/* ------------------------------------------------------------------ */

/** Create an empty model handle.  Must be paired with mhs_model_destroy(). */
MHS_API mhs_status_t mhs_model_create(mhs_model_t** out);

/** Destroy a model handle.  Passing NULL is a no-op. */
MHS_API mhs_status_t mhs_model_destroy(mhs_model_t* m);

/** Load a model from a MetaHotspot XML case file. */
MHS_API mhs_status_t mhs_model_read_xml(mhs_model_t* m, const char* path);

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_settings(mhs_model_t* m, mhs_study_t study, mhs_length_unit_t length_unit,
    double initial_temperature_K, double duration, double output_interval);

/** Set mesh vertices.  count must be >= 2 for any axis being set; pass 0
 *  for unused axes.  vertices pointers may be NULL when count is 0. */
MHS_API mhs_status_t mhs_model_set_mesh(
    mhs_model_t* m, size_t nx, const double* x, size_t ny, const double* y, size_t nz, const double* z);

MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression);

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, layers, blocks, rects           */
/* ------------------------------------------------------------------ */

MHS_API mhs_material_id_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity);

MHS_API mhs_layer_id_t mhs_model_add_layer(
    mhs_model_t* m, const char* thickness, const char* x_offset, const char* y_offset);

MHS_API mhs_block_id_t mhs_model_add_block(mhs_model_t* m, mhs_layer_id_t layer, const char* material_name,
    const char* heat_source, const char* x_offset, const char* y_offset, const char* thickness);

MHS_API mhs_status_t mhs_model_add_rect(mhs_model_t* m, mhs_block_id_t block, mhs_geometry_op_t op, const char* x,
    const char* y, const char* width, const char* height);

/* ------------------------------------------------------------------ */
/*  Model construction  —  atomic boundary conditions                 */
/* ------------------------------------------------------------------ */

/** Add a Dirichlet (fixed-temperature) boundary with one or more face regions. */
MHS_API mhs_status_t mhs_model_add_dirichlet(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* temperature);

/** Add a Neumann (fixed-heat-flux) boundary with one or more face regions. */
MHS_API mhs_status_t mhs_model_add_neumann(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* heat_flux);

/** Add a convection (Robin / Cauchy) boundary with one or more face regions. */
MHS_API mhs_status_t mhs_model_add_convection(mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions,
    const char* coefficient, const char* ambient_temperature);

MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature);
MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux);
MHS_API mhs_status_t mhs_model_set_default_convection(
    mhs_model_t* m, const char* coefficient, const char* ambient_temperature);

/* ------------------------------------------------------------------ */
/*  Model construction  —  function library                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_function_id_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression);
MHS_API mhs_function_id_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center);
MHS_API mhs_function_id_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase);
MHS_API mhs_function_id_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta);
MHS_API mhs_function_id_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, size_t count);
MHS_API mhs_function_id_t mhs_model_add_function_periodic_piecewise_constant(
    mhs_model_t* m, const char* name, const double* values, size_t count, double period);

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_probe_id_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z);
MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_axis_t axis, double coordinate,
    mhs_rect2d_t region, mhs_fluid_bc_t kind, double value, double inlet_temperature);

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(const mhs_model_t* m, mhs_compiled_t** out);
MHS_API mhs_status_t mhs_compiled_destroy(mhs_compiled_t* c);

/* ------------------------------------------------------------------ */
/*  Compiled metadata view (replaces ~12 individual accessors)         */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_metadata(const mhs_compiled_t* c, mhs_compiled_metadata_t* out);

/* ------------------------------------------------------------------ */
/*  Assembly (matrix + RHS extraction)                                */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* state, double time, mhs_assembly_t** out);
MHS_API mhs_status_t mhs_assembly_destroy(mhs_assembly_t* a);
MHS_API size_t mhs_assembly_n(const mhs_assembly_t* a);
MHS_API mhs_status_t mhs_assembly_matrix(const mhs_assembly_t* a, mhs_operator_t which, mhs_csc_view_t* out);
MHS_API const double* mhs_assembly_rhs(const mhs_assembly_t* a);

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const double* state, size_t state_count,
    const mhs_solver_opts_t* opts, mhs_solution_t** out);
MHS_API mhs_status_t mhs_solution_destroy(mhs_solution_t* s);

/* ------------------------------------------------------------------ */
/*  Single transient step (BDF1)                                       */
/* ------------------------------------------------------------------ */

/** Execute a single transient step (BDF1).
 *
 *  Advances *state* (length state_count()) from *time* by *dt* and writes
 *  the result into *out_state* (pre-allocated, same length).  *info* receives
 *  diagnostics; pass NULL to skip.
 *
 *  The compiled model must have study = TRANSIENT. */
MHS_API mhs_status_t mhs_compiled_step(const mhs_compiled_t* c, const double* state, double time, double dt,
    double* out_state, mhs_step_info_t* info, const mhs_solver_opts_t* opts);

/* ------------------------------------------------------------------ */
/*  VTU export                                                         */
/* ------------------------------------------------------------------ */

/** Write a VTU file from a compiled model and solution. */
MHS_API mhs_status_t mhs_compiled_write_vtu(const mhs_compiled_t* c, const mhs_solution_t* s, const char* path);

/* ------------------------------------------------------------------ */
/*  Solution view (replaces ~7 individual accessors)                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_view(const mhs_solution_t* s, mhs_solution_view_t* out);

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API size_t mhs_solution_probe_count(const mhs_solution_t* s);
MHS_API const char* mhs_solution_probe_name(const mhs_solution_t* s, size_t index);
MHS_API size_t mhs_solution_probe_record_count(const mhs_solution_t* s, size_t probe_index);
MHS_API const double* mhs_solution_probe_times(const mhs_solution_t* s, size_t probe_index);
MHS_API const double* mhs_solution_probe_values(const mhs_solution_t* s, size_t probe_index);
MHS_API mhs_status_t mhs_solution_probe_metadata(const mhs_solution_t* s, mhs_probe_metadata_t* out);
MHS_API mhs_status_t mhs_solution_probe_metadata_free(mhs_probe_metadata_t* meta);

/* ------------------------------------------------------------------ */
/*  Model introspection (before compile)                               */
/* ------------------------------------------------------------------ */

MHS_API const char* mhs_model_material_name(const mhs_model_t* m, size_t index);
MHS_API size_t mhs_model_material_count(const mhs_model_t* m);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_H */
