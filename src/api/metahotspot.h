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
typedef uint32_t mhs_boundary_id_t;
typedef uint32_t mhs_function_id_t;
typedef uint32_t mhs_probe_id_t;

#define MHS_LAYER_ID_INVALID UINT32_MAX
#define MHS_BLOCK_ID_INVALID UINT32_MAX
#define MHS_MATERIAL_ID_INVALID UINT32_MAX
#define MHS_BOUNDARY_ID_INVALID UINT32_MAX
#define MHS_FUNCTION_ID_INVALID UINT32_MAX
#define MHS_PROBE_ID_INVALID UINT32_MAX

/* ------------------------------------------------------------------ */
/*  Enumerations                                                       */
/* ------------------------------------------------------------------ */
typedef enum { MHS_STUDY_STEADY, MHS_STUDY_TRANSIENT } mhs_study_t;

typedef enum {
    MHS_UNIT_METER,
    MHS_UNIT_MILLIMETER,
    MHS_UNIT_MICROMETER,
    MHS_UNIT_NANOMETER,
    MHS_UNIT_INCH,
    MHS_UNIT_MIL
} mhs_length_unit_t;

typedef enum { MHS_AXIS_X, MHS_AXIS_Y, MHS_AXIS_Z } mhs_axis_t;

typedef enum { MHS_GEOM_ADD, MHS_GEOM_SUB } mhs_geometry_op_t;

typedef enum { MHS_SOLVER_PARDISO, MHS_SOLVER_EIGEN_SPARSE_LU, MHS_SOLVER_EIGEN_BICGSTAB } mhs_solver_type_t;

typedef enum { MHS_FLUID_NONE, MHS_FLUID_PRESSURE, MHS_FLUID_MASS_FLOW, MHS_FLUID_VELOCITY } mhs_fluid_bc_t;

typedef enum { MHS_OPERATOR_STIFFNESS, MHS_OPERATOR_CAPACITY } mhs_operator_t;

typedef enum {
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
} mhs_status_t;

/* ------------------------------------------------------------------ */
/*  Composite types                                                    */
/* ------------------------------------------------------------------ */

/** Axis-aligned rectangle on a boundary face. */
typedef struct {
    double a_min;
    double a_max;
    double b_min;
    double b_max;
} mhs_rect2d_t;

/** 2-D point (for piecewise-function knot data). */
typedef struct {
    double x;
    double y;
} mhs_point2d_t;

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
    int32_t rows;
    int32_t columns;
    int32_t nnz;
    const int32_t* outer_indices;
    const int32_t* inner_indices;
    const double* values;
} mhs_csc_view_t;

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
/*  Model construction  —  boundary conditions (two-step build)       */
/* ------------------------------------------------------------------ */

MHS_API mhs_boundary_id_t mhs_model_add_boundary(mhs_model_t* m);
MHS_API mhs_status_t mhs_boundary_set_dirichlet(mhs_model_t* m, mhs_boundary_id_t id, const char* temperature);
MHS_API mhs_status_t mhs_boundary_set_neumann(mhs_model_t* m, mhs_boundary_id_t id, const char* heat_flux);
MHS_API mhs_status_t mhs_boundary_set_convection(
    mhs_model_t* m, mhs_boundary_id_t id, const char* coefficient, const char* ambient_temperature);
MHS_API mhs_status_t mhs_boundary_add_face_region(
    mhs_model_t* m, mhs_boundary_id_t id, mhs_axis_t axis, double coordinate, mhs_rect2d_t region);

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

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_probe_id_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z);
MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_axis_t axis, double coordinate,
    mhs_rect2d_t region, mhs_fluid_bc_t kind, double value, double inlet_temperature);

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(mhs_model_t* m, mhs_compiled_t** out);
MHS_API mhs_status_t mhs_compiled_destroy(mhs_compiled_t* c);

MHS_API size_t mhs_compiled_cell_count(const mhs_compiled_t* c);
MHS_API size_t mhs_compiled_state_count(const mhs_compiled_t* c);
MHS_API size_t mhs_compiled_node_count(const mhs_compiled_t* c);
MHS_API double mhs_compiled_initial_temperature(const mhs_compiled_t* c);
MHS_API mhs_study_t mhs_compiled_study_type(const mhs_compiled_t* c);

MHS_API size_t mhs_compiled_layer_count(const mhs_compiled_t* c);
MHS_API size_t mhs_compiled_block_count(const mhs_compiled_t* c, uint32_t layer);

/** Const access to per-cell layer IDs, length cell_count(). */
MHS_API const uint32_t* mhs_compiled_layer_ids(const mhs_compiled_t* c);
/** Const access to per-cell block IDs, length cell_count(). */
MHS_API const uint32_t* mhs_compiled_block_ids(const mhs_compiled_t* c);

/** Total number of cells in the Cartesian grid (nx * ny * nz). */
MHS_API size_t mhs_compiled_grid_count(const mhs_compiled_t* c);

/**
 * Map from linear grid index to active-cell index.
 * Length = grid_count().  Entry == SIZE_MAX means inactive.
 * Linear index: idx = ix * (ny * nz) + iy * nz + iz
 * Pointer valid until the compiled model is destroyed. */
MHS_API const size_t* mhs_compiled_grid_to_cell(const mhs_compiled_t* c);

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const mhs_solver_opts_t* opts, mhs_solution_t** out);
MHS_API mhs_status_t mhs_solve(mhs_model_t* m, const mhs_solver_opts_t* opts, mhs_solution_t** out);
MHS_API mhs_status_t mhs_solution_destroy(mhs_solution_t* s);

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
/*  Solution accessors                                                 */
/* ------------------------------------------------------------------ */

MHS_API size_t mhs_solution_state_count(const mhs_solution_t* s);
MHS_API size_t mhs_solution_cell_count(const mhs_solution_t* s);
MHS_API size_t mhs_solution_node_count(const mhs_solution_t* s);
MHS_API double mhs_solution_time(const mhs_solution_t* s);
MHS_API const double* mhs_solution_states(const mhs_solution_t* s);
MHS_API const double* mhs_solution_cell_temperatures(const mhs_solution_t* s);
MHS_API const double* mhs_solution_node_temperatures(const mhs_solution_t* s);

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API size_t mhs_solution_probe_count(const mhs_solution_t* s);
MHS_API const char* mhs_solution_probe_name(const mhs_solution_t* s, size_t index);
MHS_API size_t mhs_solution_probe_record_count(const mhs_solution_t* s, size_t probe_index);
MHS_API const double* mhs_solution_probe_times(const mhs_solution_t* s, size_t probe_index);
MHS_API const double* mhs_solution_probe_values(const mhs_solution_t* s, size_t probe_index);

/* ------------------------------------------------------------------ */
/*  Model introspection (before compile)                               */
/* ------------------------------------------------------------------ */

MHS_API const char* mhs_model_material_name(const mhs_model_t* m, size_t index);
MHS_API size_t mhs_model_material_count(const mhs_model_t* m);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_H */
