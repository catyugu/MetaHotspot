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
/*  ID types  (int32_t, -1 = invalid)                                  */
/* ------------------------------------------------------------------ */
typedef int32_t mhs_layer_id_t;
typedef int32_t mhs_block_id_t;
typedef int32_t mhs_material_id_t;
typedef int32_t mhs_boundary_id_t;
typedef int32_t mhs_function_id_t;
typedef int32_t mhs_probe_id_t;

#define MHS_LAYER_ID_INVALID ((mhs_layer_id_t) - 1)
#define MHS_BLOCK_ID_INVALID ((mhs_block_id_t) - 1)
#define MHS_MATERIAL_ID_INVALID ((mhs_material_id_t) - 1)
#define MHS_BOUNDARY_ID_INVALID ((mhs_boundary_id_t) - 1)
#define MHS_FUNCTION_ID_INVALID ((mhs_function_id_t) - 1)
#define MHS_PROBE_ID_INVALID ((mhs_probe_id_t) - 1)

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

/** Axis-aligned rectangle on a boundary face.
 *
 *  For a Z-axis face:  a = X range,  b = Y range.
 *  For an X-axis face: a = Y range,  b = Z range.
 *  For a Y-axis face:  a = X range,  b = Z range.
 */
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

/** Load a model from a MetaHotspot XML case file.
 *  The handle must have been created with mhs_model_create() first.
 *  Existing content in the handle is replaced (destroyed). */
MHS_API mhs_status_t mhs_model_read_xml(mhs_model_t* m, const char* path);

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

/** Set global study parameters.
 *  duration and output_interval are ignored for steady-state studies.
 *  Temperature is always in Kelvin regardless of length_unit. */
MHS_API mhs_status_t mhs_model_set_settings(mhs_model_t* m, mhs_study_t study, mhs_length_unit_t length_unit,
    double initial_temperature_K, double duration, double output_interval);

/** Set mesh vertices for all three axes atomically.
 *  count must be >= 2 for any axis being set; pass 0 for unused axes.
 *  vertices pointers may be NULL when count is 0.
 *  Coordinate unit is determined by mhs_model_set_settings(). */
MHS_API mhs_status_t mhs_model_set_mesh(
    mhs_model_t* m, int32_t nx, const double* x, int32_t ny, const double* y, int32_t nz, const double* z);

/** Add a named geometry variable (e.g. "w_top = 0.01").
 *  Variable expressions are evaluated at geometry-resolution time and may
 *  reference other variables defined earlier. */
MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression);

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, layers, blocks, rects           */
/* ------------------------------------------------------------------ */

/** Register a named material.  Returned mhs_material_id_t is used
 *  (indirectly, via name) when adding blocks.
 *
 *  kx, ky, kz, rho, c are expression strings (may reference T, t).
 *  dynamic_viscosity: NULL signals a solid material; non-NULL enables
 *  fluid coupling for any block using this material.
 *
 *  Returns MHS_MATERIAL_ID_INVALID on error; call mhs_last_error(). */
MHS_API mhs_material_id_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity);

/** Add a layer.  thickness, x_offset, y_offset are expression strings
 *  (geometry variables may be referenced).
 *  Returns MHS_LAYER_ID_INVALID on error. */
MHS_API mhs_layer_id_t mhs_model_add_layer(
    mhs_model_t* m, const char* thickness, const char* x_offset, const char* y_offset);

/** Add a block to an existing layer.
 *  material_name must match a previously registered material.
 *  heat_source is an expression string (may reference x, y, z, T, t).
 *  x_offset, y_offset are expression strings.
 *  thickness: NULL = inherit layer thickness (non-NULL only valid for layer 0).
 *  Returns MHS_BLOCK_ID_INVALID on error. */
MHS_API mhs_block_id_t mhs_model_add_block(mhs_model_t* m, mhs_layer_id_t layer, const char* material_name,
    const char* heat_source, const char* x_offset, const char* y_offset, const char* thickness);

/** Add a rectangular geometry operation (add or subtract) to a block.
 *  x, y, width, height are expression strings. */
MHS_API mhs_status_t mhs_model_add_rect(mhs_model_t* m, mhs_block_id_t block, mhs_geometry_op_t op, const char* x,
    const char* y, const char* width, const char* height);

/* ------------------------------------------------------------------ */
/*  Model construction  —  boundary conditions (two-step build)       */
/* ------------------------------------------------------------------ */

/** Allocate an empty boundary-condition slot.
 *  You MUST call one of mhs_boundary_set_dirichlet / _neumann / _convection
 *  on the returned ID, then add one or more face regions, before compiling.
 *  Returns MHS_BOUNDARY_ID_INVALID on error. */
MHS_API mhs_boundary_id_t mhs_model_add_boundary(mhs_model_t* m);

/** Set a Dirichlet condition for a pending boundary slot. */
MHS_API mhs_status_t mhs_boundary_set_dirichlet(mhs_model_t* m, mhs_boundary_id_t id, const char* temperature);

/** Set a Neumann (flux) condition for a pending boundary slot. */
MHS_API mhs_status_t mhs_boundary_set_neumann(mhs_model_t* m, mhs_boundary_id_t id, const char* heat_flux);

/** Set a convection (Robin) condition for a pending boundary slot. */
MHS_API mhs_status_t mhs_boundary_set_convection(
    mhs_model_t* m, mhs_boundary_id_t id, const char* coefficient, const char* ambient_temperature);

/** Append a face region to a pending boundary slot.
 *  A face region defines a rectangular patch on a constant-coordinate
 *  plane.  Multiple regions on the same boundary are OR-ed together. */
MHS_API mhs_status_t mhs_boundary_add_face_region(
    mhs_model_t* m, mhs_boundary_id_t id, mhs_axis_t axis, double coordinate, mhs_rect2d_t region);

/** Set the default boundary applied to exposed faces that no explicit
 *  boundary patch matches.  Last-call-wins (call only one).
 *  If never called, the default is Neumann(heat_flux="0") — i.e. adiabatic. */
MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature);
MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux);
MHS_API mhs_status_t mhs_model_set_default_convection(
    mhs_model_t* m, const char* coefficient, const char* ambient_temperature);

/* ------------------------------------------------------------------ */
/*  Model construction  —  function library                            */
/* ------------------------------------------------------------------ */

/** Register a named function using an expression formula.
 *  Returns MHS_FUNCTION_ID_INVALID on error. */
MHS_API mhs_function_id_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression);

/** Register a named Gaussian function: amplitude * exp(-(t - center)^2 / tau^2). */
MHS_API mhs_function_id_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center);

/** Register a named sine function: amplitude * sin(omega * t + phase). */
MHS_API mhs_function_id_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase);

/** Register a named double-exponential pulse. */
MHS_API mhs_function_id_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta);

/** Register a named piecewise-linear function from knot points. */
MHS_API mhs_function_id_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, int32_t count);

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

/** Add a temperature observation point.
 *  Coordinates are in user-unit (converted per length_unit at compile time).
 *  Returns MHS_PROBE_ID_INVALID on error. */
MHS_API mhs_probe_id_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z);

/** Add a fluid (inlet/outlet) boundary condition on a rectangular face
 *  region.  inlet_temperature is set to NaN to omit (use the default). */
MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_axis_t axis, double coordinate,
    mhs_rect2d_t region, mhs_fluid_bc_t kind, double value, double inlet_temperature);

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

/** Compile a model into its runtime representation.
 *
 *  The model handle m is not modified — pending boundary slots are NOT
 *  consumed and remain valid for further editing.  The returned
 *  mhs_compiled_t is a snapshot of the current model state and is
 *  independent of m (destroying m does not invalidate the compiled model).
 *
 *  Call again after any modifications to produce an updated snapshot. */
MHS_API mhs_status_t mhs_model_compile(mhs_model_t* m, mhs_compiled_t** out);

/** Destroy a compiled model handle.  Passing NULL is a no-op. */
MHS_API mhs_status_t mhs_compiled_destroy(mhs_compiled_t* c);

/** Query the compiler output without solving.
 *  Useful for pre-allocating arrays in scripting languages. */
MHS_API int32_t mhs_compiled_cell_count(const mhs_compiled_t* c);
MHS_API int32_t mhs_compiled_state_count(const mhs_compiled_t* c);
MHS_API int32_t mhs_compiled_node_count(const mhs_compiled_t* c);
MHS_API double mhs_compiled_initial_temperature(const mhs_compiled_t* c);
MHS_API mhs_study_t mhs_compiled_study_type(const mhs_compiled_t* c);

/** Number of layers in the compiled model. */
MHS_API uint32_t mhs_compiled_layer_count(const mhs_compiled_t* c);

/** Number of blocks in a given layer (0-based).
 *  Returns 0 if layer is out of range. */
MHS_API uint32_t mhs_compiled_block_count(const mhs_compiled_t* c, uint32_t layer);

/** Const access to per-cell layer IDs, length cell_count().
 *  layer_id[i] == L selects cells belonging to the L-th layer (0-based).
 *  Pointer valid until the compiled model is destroyed. */
MHS_API const uint32_t* mhs_compiled_layer_ids(const mhs_compiled_t* c);

/** Const access to per-cell block IDs, length cell_count().
 *  block_id[i] == B selects cells belonging to the B-th block within
 *  their layer (0-based).  Combined with layer_id for unique selection.
 *  Pointer valid until the compiled model is destroyed. */
MHS_API const uint32_t* mhs_compiled_block_ids(const mhs_compiled_t* c);

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

/** Solve a compiled model.  c may be reused for multiple solves.
 *  opts may be NULL (sensible defaults are used).
 *  The returned mhs_solution_t must be freed with mhs_solution_destroy(). */
MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const mhs_solver_opts_t* opts, mhs_solution_t** out);

/** Single-step convenience: compile then solve.
 *  m remains valid and may be reused. */
MHS_API mhs_status_t mhs_solve(mhs_model_t* m, const mhs_solver_opts_t* opts, mhs_solution_t** out);

/** Destroy a solution handle.  Passing NULL is a no-op. */
MHS_API mhs_status_t mhs_solution_destroy(mhs_solution_t* s);

/* ------------------------------------------------------------------ */
/*  Assembly (matrix + RHS extraction)                                */
/* ------------------------------------------------------------------ */

/** Opaque handle for assembled C * dx/dt + K * x = f operators.
 *  Both matrices are stored in CSC (compressed sparse column) format.
 *  Must be freed with mhs_assembly_destroy(). */
typedef struct mhs_assembly_t mhs_assembly_t;

/** Assemble K, C, f at a given state and time.
 *  The compiled model must have been produced by mhs_model_compile().
 *  state may be NULL to use the model's initial state.
 *  The returned handle is independent — the compiled model can be modified
 *  or destroyed without affecting this handle. */
MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* state, double time, mhs_assembly_t** out);

/** Destroy an assembly handle.  Passing NULL is a no-op. */
MHS_API mhs_status_t mhs_assembly_destroy(mhs_assembly_t* a);

/** Operator dimension (number of global states). */
MHS_API int32_t mhs_assembly_n(const mhs_assembly_t* a);

/** Number of non-zero entries in K. */
MHS_API int32_t mhs_assembly_stiffness_nnz(const mhs_assembly_t* a);

/** K CSC column pointers, length n+1.
 *  outer_indices[i] is the start of column i in inner_indices / values.
 *  Pointer valid until the assembly handle is destroyed. */
MHS_API const int32_t* mhs_assembly_stiffness_outer_indices(const mhs_assembly_t* a);

/** K CSC row indices for each non-zero, length stiffness_nnz().
 *  Pointer valid until the assembly handle is destroyed. */
MHS_API const int32_t* mhs_assembly_stiffness_inner_indices(const mhs_assembly_t* a);

/** K CSC values, length stiffness_nnz(), parallel to inner_indices.
 *  Pointer valid until the assembly handle is destroyed. */
MHS_API const double* mhs_assembly_stiffness_values(const mhs_assembly_t* a);

/** Number of non-zero entries in C. */
MHS_API int32_t mhs_assembly_capacity_nnz(const mhs_assembly_t* a);

/** C CSC column pointers, length n+1. */
MHS_API const int32_t* mhs_assembly_capacity_outer_indices(const mhs_assembly_t* a);

/** C CSC row indices, length capacity_nnz(). */
MHS_API const int32_t* mhs_assembly_capacity_inner_indices(const mhs_assembly_t* a);

/** C CSC values, length capacity_nnz(). */
MHS_API const double* mhs_assembly_capacity_values(const mhs_assembly_t* a);

/** Right-hand side vector, length n.
 *  Pointer valid until the assembly handle is destroyed. */
MHS_API const double* mhs_assembly_rhs(const mhs_assembly_t* a);

/* ------------------------------------------------------------------ */
/*  Solution accessors                                                 */
/* ------------------------------------------------------------------ */

/** Number of global system states. */
MHS_API int32_t mhs_solution_state_count(const mhs_solution_t* s);

/** Number of active cells in the mesh. */
MHS_API int32_t mhs_solution_cell_count(const mhs_solution_t* s);

/** Number of nodes (vertices) in the mesh: (nx+1)*(ny+1)*(nz+1). */
MHS_API int32_t mhs_solution_node_count(const mhs_solution_t* s);

/** Final simulation time (zero for steady-state). */
MHS_API double mhs_solution_time(const mhs_solution_t* s);

/** Complete system state, length = state_count().
 *  Entries are not necessarily temperatures. */
MHS_API const double* mhs_solution_states(const mhs_solution_t* s);

/** Cell-centroid temperature field, length = cell_count().
 *  Pointer is valid until the solution handle is destroyed. */
MHS_API const double* mhs_solution_cell_temperatures(const mhs_solution_t* s);

/** Node (vertex) temperature field computed from cell-to-node
 *  interpolation, length = node_count().
 *  Pointer is valid until the solution handle is destroyed. */
MHS_API const double* mhs_solution_node_temperatures(const mhs_solution_t* s);

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

/** Number of recorded probe traces (observation points). */
MHS_API int32_t mhs_solution_probe_count(const mhs_solution_t* s);

/** Name of the probe at the given index.  0 ≤ index < probe_count(). */
MHS_API const char* mhs_solution_probe_name(const mhs_solution_t* s, int32_t index);

/** Number of recorded time-steps for a given probe
 *  (1 for steady-state, N for transient). */
MHS_API int32_t mhs_solution_probe_record_count(const mhs_solution_t* s, int32_t probe_index);

/** Time vector for a probe, length = probe_record_count().
 *  NULL for steady-state (single record). */
MHS_API const double* mhs_solution_probe_times(const mhs_solution_t* s, int32_t probe_index);

/** Temperature values for a probe, length = probe_record_count(). */
MHS_API const double* mhs_solution_probe_values(const mhs_solution_t* s, int32_t probe_index);

/* ------------------------------------------------------------------ */
/*  Model introspection (before compile)                               */
/* ------------------------------------------------------------------ */

/** Query the material name for a given material index.  Returns NULL if
 *  index is out of range.  Pointer is valid until the model is destroyed
 *  or a new material is added. */
MHS_API const char* mhs_model_material_name(const mhs_model_t* m, int32_t index);

/** Number of materials currently registered. */
MHS_API int32_t mhs_model_material_count(const mhs_model_t* m);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* METAHOTSPOT_H */
