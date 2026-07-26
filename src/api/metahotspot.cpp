/* Implementation of the MetaHotspot C API. */
#include "api/metahotspot.h"

#include "compiler/model_compiler.hpp"
#include "io/model_io.hpp"
#include "io/result_io.hpp"
#include "model/model_definition.hpp"
#include "solver/assembler.hpp"
#include "solver/scheduler.hpp" // take_step
#include "solver/solution_history.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Internal opaque handle definitions (hidden from the header)        */
/* ------------------------------------------------------------------ */

struct BlockLocation {
    uint32_t layer;
    uint32_t block;
};

struct mhs_model_t {
    mhs::model::ModelDefinition def;
    std::vector<BlockLocation> block_locations;
};

struct CscMatrixData {
    int32_t n = 0;
    int32_t nnz = 0;
    std::vector<int32_t> outer_indices;
    std::vector<int32_t> inner_indices;
    std::vector<double> values;
};

struct mhs_assembly_t {
    CscMatrixData stiffness;
    CscMatrixData capacity;
    std::vector<double> rhs;
};

struct mhs_compiled_t {
    mhs::core::Model model;
};

struct mhs_solution_t {
    mhs::core::Solution solution;
};

/* ------------------------------------------------------------------ */
/*  Thread-local error buffer                                          */
/* ------------------------------------------------------------------ */
static thread_local std::string tls_err;

#define SET_ERR(msg)                                                                                                   \
    do {                                                                                                               \
        std::ostringstream _oss;                                                                                       \
        _oss << msg;                                                                                                   \
        tls_err = _oss.str();                                                                                          \
    } while (0)

#define CHECK_NULL(p)                                                                                                  \
    do {                                                                                                               \
        if (!(p)) {                                                                                                    \
            SET_ERR("NULL pointer: " #p);                                                                              \
            return MHS_ERR_NULL_PTR;                                                                                   \
        }                                                                                                              \
    } while (0)

// Unified try/catch wrapper for mhs_status_t-returning functions.
#define MHS_TRY(err_code, ...)                                                                                         \
    try {                                                                                                              \
        tls_err.clear();                                                                                               \
        __VA_ARGS__;                                                                                                   \
        tls_err.clear();                                                                                               \
        return MHS_OK;                                                                                                 \
    }                                                                                                                  \
    catch (const std::exception& e) {                                                                                  \
        SET_ERR(e.what());                                                                                             \
        return err_code;                                                                                               \
    }

// Unified try/catch wrapper for uint32_t ID-returning functions.
// The block must contain a 'return <id_value>;' statement.
#define MHS_TRY_ID(invalid, ...)                                                                                       \
    try {                                                                                                              \
        tls_err.clear();                                                                                               \
        __VA_ARGS__;                                                                                                   \
    }                                                                                                                  \
    catch (const std::exception& e) {                                                                                  \
        SET_ERR(e.what());                                                                                             \
        return invalid;                                                                                                \
    }

/* ------------------------------------------------------------------ */
/*  Enum conversions                                                   */
/* ------------------------------------------------------------------ */
static mhs::model::Axis _to_axis(mhs_axis_t a)
{
    switch (a) {
    case MHS_AXIS_X:
        return mhs::model::Axis::X;
    case MHS_AXIS_Y:
        return mhs::model::Axis::Y;
    case MHS_AXIS_Z:
        return mhs::model::Axis::Z;
    default:
        throw std::invalid_argument("invalid axis value: " + std::to_string(a));
    }
}

static mhs::model::StudyType _to_model_study(mhs_study_t s)
{
    switch (s) {
    case MHS_STUDY_STEADY:
        return mhs::model::StudyType::Steady;
    case MHS_STUDY_TRANSIENT:
        return mhs::model::StudyType::Transient;
    default:
        throw std::invalid_argument("invalid study type: " + std::to_string(s));
    }
}

static mhs::model::LengthUnit _to_unit(mhs_length_unit_t u)
{
    switch (u) {
    case MHS_UNIT_METER:
        return mhs::model::LengthUnit::Meter;
    case MHS_UNIT_MILLIMETER:
        return mhs::model::LengthUnit::Millimeter;
    case MHS_UNIT_MICROMETER:
        return mhs::model::LengthUnit::Micrometer;
    case MHS_UNIT_NANOMETER:
        return mhs::model::LengthUnit::Nanometer;
    case MHS_UNIT_INCH:
        return mhs::model::LengthUnit::Inch;
    case MHS_UNIT_MIL:
        return mhs::model::LengthUnit::Mil;
    default:
        throw std::invalid_argument("invalid length unit: " + std::to_string(u));
    }
}

static mhs_study_t _from_core_study(mhs::core::StudyType s)
{
    switch (s) {
    case mhs::core::StudyType::Steady:
        return MHS_STUDY_STEADY;
    case mhs::core::StudyType::Transient:
        return MHS_STUDY_TRANSIENT;
    default:
        return MHS_STUDY_STEADY;
    }
}

static mhs::model::FaceRegion _make_face_region(mhs_axis_t axis, double coord, mhs_rect2d_t r)
{
    return {_to_axis(axis), coord, {{r.a_min, r.a_max, r.b_min, r.b_max}}};
}

static mhs::sim::SolverType _to_solver_type(mhs_solver_type_t t)
{
    switch (t) {
    case MHS_SOLVER_PARDISO:
        return mhs::sim::SolverType::Pardiso;
    case MHS_SOLVER_EIGEN_SPARSE_LU:
        return mhs::sim::SolverType::EigenSparseLU;
    case MHS_SOLVER_EIGEN_BICGSTAB:
        return mhs::sim::SolverType::EigenBiCGSTAB;
    }
    throw std::invalid_argument("invalid solver type: " + std::to_string(t));
}

static mhs::model::FluidBoundaryKind _to_fluid_kind(mhs_fluid_bc_t k)
{
    switch (k) {
    case MHS_FLUID_NONE:
        return mhs::model::FluidBoundaryKind::None;
    case MHS_FLUID_PRESSURE:
        return mhs::model::FluidBoundaryKind::Pressure;
    case MHS_FLUID_MASS_FLOW:
        return mhs::model::FluidBoundaryKind::MassFlowRate;
    case MHS_FLUID_VELOCITY:
        return mhs::model::FluidBoundaryKind::Velocity;
    }
    throw std::invalid_argument("invalid fluid boundary kind: " + std::to_string(k));
}

/* ------------------------------------------------------------------ */
/*  Global helpers                                                     */
/* ------------------------------------------------------------------ */

MHS_API void mhs_solver_opts_default(mhs_solver_opts_t* opts)
{
    if (!opts)
        return;
    opts->solver_type = MHS_SOLVER_PARDISO;
    opts->linear_tolerance = 1e-8;
    opts->linear_max_iterations = 1000;
    opts->underrelaxation = 1.0;
    opts->nonlinear_max_iterations = 200;
    opts->nonlinear_relative_tolerance = 1e-6;
    opts->nonlinear_absolute_tolerance = 1e-12;
}

MHS_API const char* mhs_status_string(mhs_status_t status)
{
    switch (status) {
    case MHS_OK:
        return "OK";
    case MHS_ERR_NULL_PTR:
        return "NULL pointer";
    case MHS_ERR_INVALID_ARG:
        return "invalid argument";
    case MHS_ERR_COMPILE:
        return "compilation error";
    case MHS_ERR_ASSEMBLE:
        return "assemble error";
    case MHS_ERR_SOLVE:
        return "solver did not converge";
    case MHS_ERR_IO:
        return "I/O error";
    case MHS_ERR_OOM:
        return "out of memory";
    case MHS_ERR_UNSET:
        return "unset required field";
    case MHS_ERR_RUNTIME:
        return "internal runtime error";
    default:
        return "unknown error";
    }
}

MHS_API const char* mhs_last_error(void) { return tls_err.c_str(); }

/* ------------------------------------------------------------------ */
/*  Model life-cycle                                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_create(mhs_model_t** out)
{
    CHECK_NULL(out);
    try {
        *out = new mhs_model_t {};
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::bad_alloc&) {
        *out = nullptr;
        SET_ERR("memory allocation failed");
        return MHS_ERR_OOM;
    }
}

MHS_API mhs_status_t mhs_model_destroy(mhs_model_t* m)
{
    delete m;
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_model_read_xml(mhs_model_t* m, const char* path)
{
    CHECK_NULL(m);
    CHECK_NULL(path);
    MHS_TRY(MHS_ERR_IO, {
        m->def = mhs::io::read_xml(path);
        m->block_locations.clear();
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_settings(mhs_model_t* m, mhs_study_t study, mhs_length_unit_t length_unit,
    double initial_temperature_K, double duration, double output_interval)
{
    CHECK_NULL(m);
    MHS_TRY(MHS_ERR_RUNTIME, {
        m->def.settings.study_type = _to_model_study(study);
        m->def.settings.length_unit = _to_unit(length_unit);
        m->def.settings.initial_temperature = initial_temperature_K;
        m->def.settings.transient_duration = duration;
        m->def.settings.transient_output_interval = output_interval;
    });
}

MHS_API mhs_status_t mhs_model_set_mesh(
    mhs_model_t* m, size_t nx, const double* x, size_t ny, const double* y, size_t nz, const double* z)
{
    CHECK_NULL(m);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        m->def.mesh.x_vertices.clear();
        m->def.mesh.y_vertices.clear();
        m->def.mesh.z_vertices.clear();
        if (nx >= 2) {
            CHECK_NULL(x);
            m->def.mesh.x_vertices.assign(x, x + nx);
        }
        if (ny >= 2) {
            CHECK_NULL(y);
            m->def.mesh.y_vertices.assign(y, y + ny);
        }
        if (nz >= 2) {
            CHECK_NULL(z);
            m->def.mesh.z_vertices.assign(z, z + nz);
        }
    });
}

MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(expression);
    MHS_TRY(MHS_ERR_INVALID_ARG, { m->def.variables.push_back({name, expression}); });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, layers, blocks, rects           */
/* ------------------------------------------------------------------ */

MHS_API mhs_material_id_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity)
{
    if (!m || !name) {
        SET_ERR("NULL pointer");
        return MHS_MATERIAL_ID_INVALID;
    }
    MHS_TRY_ID(MHS_MATERIAL_ID_INVALID, {
        mhs::model::MaterialSpec spec;
        if (kx)
            spec.conductivity_x = kx;
        if (ky)
            spec.conductivity_y = ky;
        if (kz)
            spec.conductivity_z = kz;
        if (rho)
            spec.density = rho;
        if (c)
            spec.specific_heat = c;
        if (dynamic_viscosity)
            spec.dynamic_viscosity = std::string(dynamic_viscosity);

        m->def.materials.push_back({name, std::move(spec)});
        return static_cast<mhs_material_id_t>(m->def.materials.size() - 1);
    });
}

MHS_API mhs_layer_id_t mhs_model_add_layer(
    mhs_model_t* m, const char* thickness, const char* x_offset, const char* y_offset)
{
    if (!m) {
        SET_ERR("NULL pointer");
        return MHS_LAYER_ID_INVALID;
    }
    if (!thickness || !x_offset || !y_offset) {
        SET_ERR("NULL pointer in layer params");
        return MHS_LAYER_ID_INVALID;
    }
    MHS_TRY_ID(MHS_LAYER_ID_INVALID, {
        m->def.layers.push_back({thickness, x_offset, y_offset, {}});
        return static_cast<mhs_layer_id_t>(m->def.layers.size() - 1);
    });
}

MHS_API mhs_block_id_t mhs_model_add_block(mhs_model_t* m, mhs_layer_id_t layer, const char* material_name,
    const char* heat_source, const char* x_offset, const char* y_offset, const char* thickness)
{
    if (!m) {
        SET_ERR("NULL pointer");
        return MHS_BLOCK_ID_INVALID;
    }
    if (layer == MHS_LAYER_ID_INVALID) {
        SET_ERR("invalid layer ID");
        return MHS_BLOCK_ID_INVALID;
    }
    if (!material_name) {
        SET_ERR("NULL pointer: material_name");
        return MHS_BLOCK_ID_INVALID;
    }
    MHS_TRY_ID(MHS_BLOCK_ID_INVALID, {
        mhs::model::BlockSpec block;
        block.material = material_name;
        block.volumetric_heat_source = heat_source ? heat_source : "0.0";
        block.x_offset = x_offset ? x_offset : "0.0";
        block.y_offset = y_offset ? y_offset : "0.0";
        if (thickness)
            block.thickness = std::string(thickness);

        m->def.layers[layer].blocks.push_back(std::move(block));
        const auto block_idx = static_cast<uint32_t>(m->def.layers[layer].blocks.size() - 1);
        m->block_locations.push_back({layer, block_idx});
        return static_cast<mhs_block_id_t>(m->block_locations.size() - 1);
    });
}

MHS_API mhs_status_t mhs_model_add_rect(mhs_model_t* m, mhs_block_id_t block, mhs_geometry_op_t op, const char* x,
    const char* y, const char* width, const char* height)
{
    CHECK_NULL(m);
    if (block == MHS_BLOCK_ID_INVALID) {
        SET_ERR("invalid block ID");
        return MHS_ERR_INVALID_ARG;
    }
    if (!x || !y || !width || !height) {
        SET_ERR("NULL pointer in rect params");
        return MHS_ERR_NULL_PTR;
    }
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        const auto loc = m->block_locations[block];
        mhs::model::RectOperation rect_op;
        rect_op.operation
            = (op == MHS_GEOM_SUB) ? mhs::model::GeometryOperation::Subtract : mhs::model::GeometryOperation::Add;
        rect_op.rect = {x, y, width, height};
        m->def.layers[loc.layer].blocks[loc.block].geometry.push_back(std::move(rect_op));
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  atomic boundary conditions                 */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_dirichlet(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(regions);
    CHECK_NULL(temperature);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        mhs::model::BoundaryPatch bp;
        bp.condition = mhs::model::DirichletBoundary {temperature};
        bp.regions.reserve(n_regions);
        for (size_t i = 0; i < n_regions; ++i)
            bp.regions.push_back(_make_face_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
        m->def.boundaries.push_back(std::move(bp));
    });
}

MHS_API mhs_status_t mhs_model_add_neumann(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* heat_flux)
{
    CHECK_NULL(m);
    CHECK_NULL(regions);
    CHECK_NULL(heat_flux);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        mhs::model::BoundaryPatch bp;
        bp.condition = mhs::model::NeumannBoundary {heat_flux};
        bp.regions.reserve(n_regions);
        for (size_t i = 0; i < n_regions; ++i)
            bp.regions.push_back(_make_face_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
        m->def.boundaries.push_back(std::move(bp));
    });
}

MHS_API mhs_status_t mhs_model_add_convection(mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions,
    const char* coefficient, const char* ambient_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(regions);
    CHECK_NULL(coefficient);
    CHECK_NULL(ambient_temperature);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        mhs::model::BoundaryPatch bp;
        bp.condition = mhs::model::ConvectionBoundary {coefficient, ambient_temperature};
        bp.regions.reserve(n_regions);
        for (size_t i = 0; i < n_regions; ++i)
            bp.regions.push_back(_make_face_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
        m->def.boundaries.push_back(std::move(bp));
    });
}

MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(temperature);
    MHS_TRY(MHS_ERR_RUNTIME, { m->def.default_boundary = mhs::model::DirichletBoundary {temperature}; });
}

MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux)
{
    CHECK_NULL(m);
    CHECK_NULL(heat_flux);
    MHS_TRY(MHS_ERR_RUNTIME, { m->def.default_boundary = mhs::model::NeumannBoundary {heat_flux}; });
}

MHS_API mhs_status_t mhs_model_set_default_convection(
    mhs_model_t* m, const char* coefficient, const char* ambient_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(coefficient);
    CHECK_NULL(ambient_temperature);
    MHS_TRY(MHS_ERR_RUNTIME,
        { m->def.default_boundary = mhs::model::ConvectionBoundary {coefficient, ambient_temperature}; });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  function library                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_function_id_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression)
{
    if (!m || !name || !expression) {
        SET_ERR("NULL pointer");
        return MHS_FUNCTION_ID_INVALID;
    }
    MHS_TRY_ID(MHS_FUNCTION_ID_INVALID, {
        m->def.functions.push_back({name, mhs::model::ExpressionFunctionSpec {expression}});
        return static_cast<mhs_function_id_t>(m->def.functions.size() - 1);
    });
}

MHS_API mhs_function_id_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center)
{
    if (!m || !name) {
        SET_ERR("NULL pointer");
        return MHS_FUNCTION_ID_INVALID;
    }
    MHS_TRY_ID(MHS_FUNCTION_ID_INVALID, {
        m->def.functions.push_back({name, mhs::model::GaussFunctionSpec {amplitude, tau, center}});
        return static_cast<mhs_function_id_t>(m->def.functions.size() - 1);
    });
}

MHS_API mhs_function_id_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase)
{
    if (!m || !name) {
        SET_ERR("NULL pointer");
        return MHS_FUNCTION_ID_INVALID;
    }
    MHS_TRY_ID(MHS_FUNCTION_ID_INVALID, {
        m->def.functions.push_back({name, mhs::model::SineFunctionSpec {amplitude, angular_frequency, phase}});
        return static_cast<mhs_function_id_t>(m->def.functions.size() - 1);
    });
}

MHS_API mhs_function_id_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta)
{
    if (!m || !name) {
        SET_ERR("NULL pointer");
        return MHS_FUNCTION_ID_INVALID;
    }
    MHS_TRY_ID(MHS_FUNCTION_ID_INVALID, {
        m->def.functions.push_back({name, mhs::model::DoubleExponentialFunctionSpec {amplitude, alpha, beta}});
        return static_cast<mhs_function_id_t>(m->def.functions.size() - 1);
    });
}

MHS_API mhs_function_id_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, size_t count)
{
    if (!m || !name || !points) {
        SET_ERR("NULL pointer");
        return MHS_FUNCTION_ID_INVALID;
    }
    if (count < 2) {
        SET_ERR("piecewise requires count >= 2");
        return MHS_FUNCTION_ID_INVALID;
    }
    MHS_TRY_ID(MHS_FUNCTION_ID_INVALID, {
        mhs::model::PiecewiseFunctionSpec spec;
        for (size_t i = 0; i < count; ++i)
            spec.points.push_back({points[i].x, points[i].y});
        m->def.functions.push_back({name, std::move(spec)});
        return static_cast<mhs_function_id_t>(m->def.functions.size() - 1);
    });
}

MHS_API mhs_function_id_t mhs_model_add_function_periodic_piecewise_constant(
    mhs_model_t* m, const char* name, const double* values, size_t count, double period)
{
    if (!m || !name || !values) {
        SET_ERR("NULL pointer");
        return MHS_FUNCTION_ID_INVALID;
    }
    if (count < 1) {
        SET_ERR("periodic_piecewise_constant requires count >= 1");
        return MHS_FUNCTION_ID_INVALID;
    }
    if (period <= 0.0) {
        SET_ERR("period must be positive");
        return MHS_FUNCTION_ID_INVALID;
    }
    MHS_TRY_ID(MHS_FUNCTION_ID_INVALID, {
        mhs::model::PeriodicPiecewiseConstantFunctionSpec spec;
        spec.period = period;
        spec.values.assign(values, values + count);
        m->def.functions.push_back({name, std::move(spec)});
        return static_cast<mhs_function_id_t>(m->def.functions.size() - 1);
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_probe_id_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z)
{
    if (!m || !name) {
        SET_ERR("NULL pointer");
        return MHS_PROBE_ID_INVALID;
    }
    MHS_TRY_ID(MHS_PROBE_ID_INVALID, {
        m->def.observation_points.push_back({name, std::to_string(x), std::to_string(y), std::to_string(z)});
        return static_cast<mhs_probe_id_t>(m->def.observation_points.size() - 1);
    });
}

MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_axis_t axis, double coordinate,
    mhs_rect2d_t region, mhs_fluid_bc_t kind, double value, double inlet_temperature)
{
    CHECK_NULL(m);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        mhs::model::FluidBoundarySpec fb;
        fb.regions.push_back(_make_face_region(axis, coordinate, region));
        fb.kind = _to_fluid_kind(kind);
        fb.value = value;
        fb.inlet_temperature = inlet_temperature;
        m->def.fluid_boundaries.push_back(std::move(fb));
    });
}

/* ------------------------------------------------------------------ */
/*  Model introspection                                                */
/* ------------------------------------------------------------------ */

MHS_API const char* mhs_model_material_name(const mhs_model_t* m, size_t index)
{
    if (!m)
        return nullptr;
    if (index >= m->def.materials.size())
        return nullptr;
    return m->def.materials[index].name.c_str();
}

MHS_API size_t mhs_model_material_count(const mhs_model_t* m)
{
    if (!m)
        return 0;
    return m->def.materials.size();
}

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(const mhs_model_t* m, mhs_compiled_t** out)
{
    CHECK_NULL(m);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_COMPILE, {
        auto core_model = mhs::sim::build_model(m->def);

        auto* c = new (std::nothrow) mhs_compiled_t {};
        if (!c) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        c->model = std::move(core_model);
        *out = c;
    });
}

MHS_API mhs_status_t mhs_compiled_destroy(mhs_compiled_t* c)
{
    delete c;
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Compiled metadata view                                             */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_metadata(const mhs_compiled_t* c, mhs_compiled_metadata_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    out->cell_count = c->model.cells.cell_to_grid.size();
    out->state_count = c->model.dofs.total_count;
    out->node_count = (c->model.mesh.nx + 1) * (c->model.mesh.ny + 1) * (c->model.mesh.nz + 1);
    out->grid_count = c->model.mesh.nx * c->model.mesh.ny * c->model.mesh.nz;
    out->study_type = _from_core_study(c->model.study_type);
    out->initial_temperature = c->model.initial_temperature;
    out->layer_ids = c->model.cells.layer_id.data();
    out->block_ids = c->model.cells.block_id.data();
    out->grid_to_cell = c->model.cells.grid_to_cell.data();
    out->nx = c->model.mesh.nx;
    out->ny = c->model.mesh.ny;
    out->nz = c->model.mesh.nz;
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Single transient step (exposes take_step kernel to C API)         */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_step(const mhs_compiled_t* c, const double* state, double time, double dt,
    double* out_state, mhs_step_info_t* info, const mhs_solver_opts_t* opts)
{
    CHECK_NULL(c);
    CHECK_NULL(state);
    CHECK_NULL(out_state);

    if (c->model.study_type != mhs::core::StudyType::Transient) {
        SET_ERR("step requires a transient model");
        return MHS_ERR_INVALID_ARG;
    }

    MHS_TRY(MHS_ERR_SOLVE, {
        const auto n = static_cast<std::size_t>(c->model.dofs.total_count);

        // Local state for this step — no mutable cache on the compiled model.
        mhs::sim::Assembler assembler(c->model);
        auto solver = mhs::sim::LinearSolver::create();
        mhs::core::SolutionHistory history(n, 2);

        // Copy input state into work buffer.
        std::vector<double> work_state(state, state + n);

        // Initialise history for this step.
        history.initialize(work_state, time);

        // Solver options.
        mhs::sim::NonLinearConfig nl_cfg;
        if (opts) {
            nl_cfg.underrelaxation = opts->underrelaxation;
            nl_cfg.max_iterations = opts->nonlinear_max_iterations;
            nl_cfg.relative_tolerance = opts->nonlinear_relative_tolerance;
            nl_cfg.absolute_tolerance = opts->nonlinear_absolute_tolerance;
        }

        // Execute the shared kernel.
        auto result = mhs::sim::take_step(assembler, *solver, history, work_state, time, dt, nl_cfg);

        // Copy result out.
        std::copy(work_state.begin(), work_state.end(), out_state);

        if (info) {
            info->accepted = result.accepted ? 1 : 0;
            info->error_ratio = result.error_ratio;
            info->suggested_dt_factor = result.suggested_dt_factor;
            info->nonlinear_iterations = static_cast<int32_t>(result.nonlinear_iterations);
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Assembly (K, C, f in CSC format)                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* state, double time, mhs_assembly_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_ASSEMBLE, {
        mhs::sim::Assembler assembler(c->model);

        const auto n = c->model.dofs.total_count;
        std::vector<double> current_state(n);
        if (state) {
            std::copy_n(state, n, current_state.begin());
        }
        else {
            current_state = c->model.initial_state;
        }

        mhs::sim::AssembleContext ctx {current_state, time};
        auto result = assembler.assemble(ctx);

        auto h = std::make_unique<mhs_assembly_t>();
        const auto copy_matrix = [](Eigen::SparseMatrix<double>& source, CscMatrixData& destination) {
            source.makeCompressed();
            destination.n = static_cast<int32_t>(source.rows());
            destination.nnz = static_cast<int32_t>(source.nonZeros());
            destination.outer_indices.assign(source.outerIndexPtr(), source.outerIndexPtr() + destination.n + 1);
            destination.inner_indices.assign(source.innerIndexPtr(), source.innerIndexPtr() + destination.nnz);
            destination.values.assign(source.valuePtr(), source.valuePtr() + destination.nnz);
        };
        copy_matrix(result.K, h->stiffness);
        copy_matrix(result.C, h->capacity);
        int32_t dim = h->stiffness.n;
        h->rhs.assign(result.f.data(), result.f.data() + dim);

        *out = h.release();
    });
}

MHS_API mhs_status_t mhs_assembly_destroy(mhs_assembly_t* a)
{
    delete a;
    tls_err.clear();
    return MHS_OK;
}

MHS_API size_t mhs_assembly_n(const mhs_assembly_t* a)
{
    if (!a)
        return 0;
    return a->stiffness.n;
}

MHS_API const double* mhs_assembly_rhs(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->rhs.data();
}

MHS_API mhs_status_t mhs_assembly_matrix(const mhs_assembly_t* a, mhs_operator_t which, mhs_csc_view_t* out)
{
    CHECK_NULL(a);
    CHECK_NULL(out);

    const CscMatrixData* matrix = nullptr;
    switch (which) {
    case MHS_OPERATOR_STIFFNESS:
        matrix = &a->stiffness;
        break;
    case MHS_OPERATOR_CAPACITY:
        matrix = &a->capacity;
        break;
    default:
        SET_ERR("invalid operator");
        return MHS_ERR_INVALID_ARG;
    }

    *out = {matrix->n, matrix->n, matrix->nnz, matrix->outer_indices.data(), matrix->inner_indices.data(),
        matrix->values.data()};
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_set_initial_state(mhs_compiled_t* c, const double* state, size_t count)
{
    CHECK_NULL(c);
    CHECK_NULL(state);
    MHS_TRY(MHS_ERR_RUNTIME, {
        if (count != static_cast<size_t>(c->model.dofs.total_count)) {
            SET_ERR("set_initial_state: expected " << c->model.dofs.total_count << " values, got " << count);
            return MHS_ERR_INVALID_ARG;
        }
        c->model.initial_state.assign(state, state + count);
    });
}

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const mhs_solver_opts_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_SOLVE, {
        mhs::sim::SolveOptions so;
        if (opts) {
            so.solver.type = _to_solver_type(opts->solver_type);
            so.solver.config.tolerance = opts->linear_tolerance;
            so.solver.config.max_iterations = opts->linear_max_iterations;
            so.nonlinear.underrelaxation = opts->underrelaxation;
            so.nonlinear.max_iterations = opts->nonlinear_max_iterations;
            so.nonlinear.relative_tolerance = opts->nonlinear_relative_tolerance;
            so.nonlinear.absolute_tolerance = opts->nonlinear_absolute_tolerance;
        }

        auto sol = mhs::sim::solve(c->model, so);

        auto* s = new (std::nothrow) mhs_solution_t {std::move(sol)};
        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        *out = s;
    });
}

/* ------------------------------------------------------------------ */
/*  VTU export                                                         */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_write_vtu(const mhs_compiled_t* c, const mhs_solution_t* s, const char* path)
{
    CHECK_NULL(c);
    CHECK_NULL(s);
    CHECK_NULL(path);
    MHS_TRY(MHS_ERR_IO, { mhs::io::write_vtu(path, c->model, s->solution.cell_temperature); });
}

/* ------------------------------------------------------------------ */
/*  Solution life-cycle                                                */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_destroy(mhs_solution_t* s)
{
    delete s;
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Solution view                                                      */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_view(const mhs_solution_t* s, mhs_solution_view_t* out)
{
    CHECK_NULL(s);
    CHECK_NULL(out);
    out->cell_count = s->solution.cell_temperature.size();
    out->state_count = s->solution.state.size();
    out->time = s->solution.time;
    out->cell_temperatures = s->solution.cell_temperature.data();
    out->states = s->solution.state.data();
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API size_t mhs_solution_probe_count(const mhs_solution_t* s)
{
    if (!s)
        return 0;
    return s->solution.probe_traces.size();
}

MHS_API const char* mhs_solution_probe_name(const mhs_solution_t* s, size_t index)
{
    if (!s)
        return nullptr;
    if (index >= s->solution.probe_traces.size())
        return nullptr;
    return s->solution.probe_traces[index].name.c_str();
}

MHS_API size_t mhs_solution_probe_record_count(const mhs_solution_t* s, size_t probe_index)
{
    if (!s)
        return 0;
    if (probe_index >= s->solution.probe_traces.size())
        return 0;
    return s->solution.probe_traces[probe_index].values.size();
}

MHS_API const double* mhs_solution_probe_times(const mhs_solution_t* s, size_t probe_index)
{
    if (!s)
        return nullptr;
    if (probe_index >= s->solution.probe_traces.size())
        return nullptr;
    const auto& tr = s->solution.probe_traces[probe_index];
    return tr.times.empty() ? nullptr : tr.times.data();
}

MHS_API const double* mhs_solution_probe_values(const mhs_solution_t* s, size_t probe_index)
{
    if (!s)
        return nullptr;
    if (probe_index >= s->solution.probe_traces.size())
        return nullptr;
    return s->solution.probe_traces[probe_index].values.data();
}

MHS_API mhs_status_t mhs_solution_probe_metadata(const mhs_solution_t* s, mhs_probe_metadata_t* out)
{
    CHECK_NULL(s);
    CHECK_NULL(out);
    out->count = s->solution.probe_traces.size();
    // Build arrays of C string pointers and record counts.
    // These are heap-allocated and freed by the caller via mhs_solution_probe_metadata_free().
    auto** names = new const char*[out->count];
    auto* record_counts = new size_t[out->count];
    for (size_t i = 0; i < out->count; ++i) {
        names[i] = s->solution.probe_traces[i].name.c_str();
        record_counts[i] = s->solution.probe_traces[i].values.size();
    }
    out->names = names;
    out->record_counts = record_counts;
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_solution_probe_metadata_free(mhs_probe_metadata_t* meta)
{
    CHECK_NULL(meta);
    delete[] meta->names;
    delete[] meta->record_counts;
    meta->names = nullptr;
    meta->record_counts = nullptr;
    meta->count = 0;
    tls_err.clear();
    return MHS_OK;
}
