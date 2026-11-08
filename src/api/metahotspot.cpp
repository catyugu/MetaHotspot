/* Implementation of the MetaHotspot C API. */
#include "metahotspot_c_api.h"

#include "compiler/model_compiler.hpp"
#include "logging/logger.hpp"
#include "model/model_definition.hpp"
#include "solver/assembler.hpp"
#include "solver/solver.hpp"

#include "core/vector.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>

/* ------------------------------------------------------------------ */
/*  Shared error-buffer helpers                                        */
/* ------------------------------------------------------------------ */

void mhs_detail_set_last_error(const std::string& msg);
void mhs_detail_clear_last_error();
const char* mhs_detail_last_error();

/* ------------------------------------------------------------------ */
/*  Shared error-handling macros                                       */
/* ------------------------------------------------------------------ */

#define SET_ERR(msg)                                                                                                   \
    do {                                                                                                               \
        std::ostringstream _oss;                                                                                       \
        _oss << msg;                                                                                                   \
        mhs_detail_set_last_error(_oss.str());                                                                         \
    } while (0)

#define CHECK_NULL(p)                                                                                                  \
    do {                                                                                                               \
        if (!(p)) {                                                                                                    \
            SET_ERR("NULL pointer: " #p);                                                                              \
            return MHS_ERROR;                                                                                          \
        }                                                                                                              \
    } while (0)

#define MHS_TRY(...)                                                                                                   \
    try {                                                                                                              \
        __VA_ARGS__;                                                                                                   \
        mhs_detail_clear_last_error();                                                                                 \
        return MHS_OK;                                                                                                 \
    }                                                                                                                  \
    catch (const std::exception& e) {                                                                                  \
        SET_ERR(e.what());                                                                                             \
        return MHS_ERROR;                                                                                              \
    }

/* ------------------------------------------------------------------ */
/*  Shared helper declarations                                         */
/* ------------------------------------------------------------------ */

/** Convert C-level solver opts to SolveOptions. */
mhs::solver::SolveOptions to_solve_options(const mhs_solve_options_t* opts);

/* ------------------------------------------------------------------ */
/*  Opaque handle structs  (exposed to both core and macromodel TUs)   */
/* ------------------------------------------------------------------ */

struct mhs_model_t {
    mhs::model::ModelDefinition def;
};

struct mhs_compiled_t {
    mhs::model::Model model;
};

struct mhs_operators_t {
    mhs::solver::Operators operators;
};

struct mhs_solution_t {
    mhs::solver::Solution sol;
};

/* ------------------------------------------------------------------ */
/*  Thread-local error buffer                                          */
/* ------------------------------------------------------------------ */
static thread_local std::string tls_err;

void mhs_detail_set_last_error(const std::string& msg) { tls_err = msg; }
void mhs_detail_clear_last_error() { tls_err.clear(); }
const char* mhs_detail_last_error() { return tls_err.c_str(); }

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

static mhs::model::FaceRegion _make_face_region(mhs_axis_t axis, double coord, mhs_rect2d_t r)
{
    if (!std::isfinite(coord) || !std::isfinite(r.a_min) || !std::isfinite(r.a_max) || !std::isfinite(r.b_min)
        || !std::isfinite(r.b_max) || r.a_min >= r.a_max || r.b_min >= r.b_max)
        throw std::invalid_argument("face region bounds must be finite and increasing");
    return {_to_axis(axis), coord, {{r.a_min, r.a_max, r.b_min, r.b_max}}};
}

template <typename Boundary>
static void _add_boundary_patch(mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, Boundary condition)
{
    mhs::model::BoundaryPatch bp;
    bp.condition = std::move(condition);
    bp.regions.reserve(n_regions);
    for (size_t i = 0; i < n_regions; ++i)
        bp.regions.push_back(_make_face_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
    m->def.boundaries.push_back(std::move(bp));
}

static mhs::solver::SolveOptions::LinearSolverType _to_solver_type(mhs_solver_type_t t)
{
    switch (t) {
    case MHS_SOLVER_PARDISO:
        return mhs::solver::SolveOptions::LinearSolverType::Pardiso;
    case MHS_SOLVER_AMG:
        return mhs::solver::SolveOptions::LinearSolverType::Amg;
    }
    throw std::invalid_argument("invalid solver type: " + std::to_string(t));
}

/* ------------------------------------------------------------------ */
/*  Global helpers                                                     */
/* ------------------------------------------------------------------ */

MHS_API void mhs_solve_options_default(mhs_solve_options_t* opts)
{
    if (!opts)
        return;
    const mhs::solver::SolveOptions defaults {};
    opts->solver_type = defaults.linear_solver == mhs::solver::SolveOptions::LinearSolverType::Pardiso
        ? MHS_SOLVER_PARDISO
        : MHS_SOLVER_AMG;
    opts->linear_tolerance = defaults.linear_tolerance;
    opts->linear_max_iterations = defaults.linear_max_iterations;
    opts->underrelaxation = defaults.underrelaxation;
    opts->nonlinear_max_iterations = defaults.nonlinear_max_iterations;
    opts->nonlinear_relative_tolerance = defaults.nonlinear_relative_tolerance;
    opts->nonlinear_absolute_tolerance = defaults.nonlinear_absolute_tolerance;
    opts->time_tolerance = defaults.time_tolerance;
    opts->max_time_step = defaults.max_time_step.value_or(0.0);
}

MHS_API const char* mhs_last_error(void) { return mhs_detail_last_error(); }

MHS_API const char* mhs_version_string(void) { return MHS_VERSION_STRING; }

/* ------------------------------------------------------------------ */
/*  Model life-cycle                                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_create(mhs_model_t** out)
{
    CHECK_NULL(out);
    *out = nullptr;
    try {
        *out = new mhs_model_t {};
        mhs_detail_clear_last_error();
        return MHS_OK;
    }
    catch (const std::bad_alloc&) {
        *out = nullptr;
        SET_ERR("memory allocation failed");
        return MHS_ERROR;
    }
}

MHS_API void mhs_model_destroy(mhs_model_t* m) { delete m; }

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_initial_temperature(mhs_model_t* m, double temperature_K)
{
    CHECK_NULL(m);
    MHS_TRY({
        if (!std::isfinite(temperature_K))
            throw std::invalid_argument("initial temperature must be finite");
        m->def.settings.initial_temperature = temperature_K;
    });
}

MHS_API mhs_status_t mhs_model_set_mesh(
    mhs_model_t* m, size_t nx, const double* x, size_t ny, const double* y, size_t nz, const double* z)
{
    CHECK_NULL(m);
    MHS_TRY({
        auto axis = [](const double* data, size_t count) {
            if (!data || count < 2)
                throw std::invalid_argument("mesh axes require at least two vertices");
            mhs::core::Vector<double> result(data, data + count);
            for (size_t i = 0; i < count; ++i)
                if (!std::isfinite(result[i]) || (i && result[i] <= result[i - 1]))
                    throw std::invalid_argument("mesh vertices must be finite and strictly increasing");
            return result;
        };
        mhs::model::MeshSpec mesh {axis(x, nx), axis(y, ny), axis(z, nz)};
        m->def.mesh = std::move(mesh);
    });
}

MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(expression);
    MHS_TRY({ m->def.variables.push_back({name, expression}); });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, sources, volumes           */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(out_id);
    CHECK_NULL(kx);
    CHECK_NULL(ky);
    CHECK_NULL(kz);
    CHECK_NULL(rho);
    CHECK_NULL(c);
    MHS_TRY({
        mhs::model::MaterialSpec spec;
        spec.conductivity_x = kx;
        spec.conductivity_y = ky;
        spec.conductivity_z = kz;
        spec.density = rho;
        spec.specific_heat = c;
        if (dynamic_viscosity && dynamic_viscosity[0] != '\0') {
            spec.dynamic_viscosity = std::string(dynamic_viscosity);
            spec.is_fluid = true;
        }
        *out_id = static_cast<mhs_index_t>(m->def.materials.size());
        m->def.materials.push_back({name, std::move(spec)});
    });
}

MHS_API mhs_status_t mhs_model_add_source(mhs_model_t* m, const char* expression, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(expression);
    CHECK_NULL(out_id);
    MHS_TRY({
        *out_id = static_cast<mhs_index_t>(m->def.sources.size());
        m->def.sources.push_back(expression);
    });
}
MHS_API mhs_status_t mhs_model_add_volume(mhs_model_t* m, mhs_index_t material, mhs_index_t source, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(out_id);
    MHS_TRY({
        if (material >= m->def.materials.size())
            throw std::out_of_range("material ID out of range");
        if (source != MHS_INVALID_INDEX && source >= m->def.sources.size())
            throw std::out_of_range("source ID out of range");
        *out_id = static_cast<mhs_index_t>(m->def.volumes.size());
        m->def.volumes.push_back({material, source, {}});
    });
}
MHS_API mhs_status_t mhs_model_volume_add_box(
    mhs_model_t* m, mhs_index_t volume, mhs_geometry_op_t op, const double min_xyz[3], const double max_xyz[3])
{
    CHECK_NULL(m);
    CHECK_NULL(min_xyz);
    CHECK_NULL(max_xyz);
    MHS_TRY({
        if (volume >= m->def.volumes.size())
            throw std::out_of_range("volume ID out of range");
        if (op != MHS_GEOM_ADD && op != MHS_GEOM_SUB)
            throw std::invalid_argument("invalid geometry operation");
        mhs::model::BoxOperation box;
        box.operation
            = op == MHS_GEOM_ADD ? mhs::model::GeometryOperation::Add : mhs::model::GeometryOperation::Subtract;
        for (int a = 0; a < 3; ++a) {
            box.min[a] = min_xyz[a];
            box.max[a] = max_xyz[a];
        }
        m->def.volumes[volume].geometry.push_back(box);
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  atomic boundary conditions                 */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_dirichlet(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* temperature)
{
    CHECK_NULL(m);
    if (n_regions == 0) {
        SET_ERR("boundary requires at least one face region");
        return MHS_ERROR;
    }
    CHECK_NULL(regions);
    CHECK_NULL(temperature);
    MHS_TRY({ _add_boundary_patch(m, regions, n_regions, mhs::model::DirichletBoundary {temperature}); });
}

MHS_API mhs_status_t mhs_model_add_neumann(
    mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions, const char* heat_flux)
{
    CHECK_NULL(m);
    if (n_regions == 0) {
        SET_ERR("boundary requires at least one face region");
        return MHS_ERROR;
    }
    CHECK_NULL(regions);
    CHECK_NULL(heat_flux);
    MHS_TRY({ _add_boundary_patch(m, regions, n_regions, mhs::model::NeumannBoundary {heat_flux}); });
}

MHS_API mhs_status_t mhs_model_add_convection(mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions,
    const char* coefficient, const char* ambient_temperature)
{
    CHECK_NULL(m);
    if (n_regions == 0) {
        SET_ERR("boundary requires at least one face region");
        return MHS_ERROR;
    }
    CHECK_NULL(regions);
    CHECK_NULL(coefficient);
    CHECK_NULL(ambient_temperature);
    MHS_TRY({
        _add_boundary_patch(m, regions, n_regions, mhs::model::ConvectionBoundary {coefficient, ambient_temperature});
    });
}

MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(temperature);
    MHS_TRY({ m->def.default_boundary = mhs::model::DirichletBoundary {temperature}; });
}

MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux)
{
    CHECK_NULL(m);
    CHECK_NULL(heat_flux);
    MHS_TRY({ m->def.default_boundary = mhs::model::NeumannBoundary {heat_flux}; });
}

MHS_API mhs_status_t mhs_model_set_default_convection(
    mhs_model_t* m, const char* coefficient, const char* ambient_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(coefficient);
    CHECK_NULL(ambient_temperature);
    MHS_TRY({ m->def.default_boundary = mhs::model::ConvectionBoundary {coefficient, ambient_temperature}; });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  function library                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(expression);
    MHS_TRY({ m->def.functions.push_back({name, mhs::model::ExpressionFunctionSpec {expression}}); });
}

MHS_API mhs_status_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY({ m->def.functions.push_back({name, mhs::model::GaussFunctionSpec {amplitude, tau, center}}); });
}

MHS_API mhs_status_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(
        { m->def.functions.push_back({name, mhs::model::SineFunctionSpec {amplitude, angular_frequency, phase}}); });
}

MHS_API mhs_status_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(
        { m->def.functions.push_back({name, mhs::model::DoubleExponentialFunctionSpec {amplitude, alpha, beta}}); });
}

MHS_API mhs_status_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, size_t count)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(points);
    MHS_TRY({
        mhs::model::PiecewiseFunctionSpec spec;
        for (size_t i = 0; i < count; ++i)
            spec.points.push_back({points[i].x, points[i].y});
        m->def.functions.push_back({name, std::move(spec)});
    });
}

MHS_API mhs_status_t mhs_model_add_function_periodic_piecewise_constant(
    mhs_model_t* m, const char* name, const double* values, size_t count, double period)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(values);
    MHS_TRY({
        mhs::model::PeriodicPiecewiseConstantFunctionSpec spec;
        spec.period = period;
        spec.values.assign(values, values + count);
        m->def.functions.push_back({name, std::move(spec)});
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                 */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY({ m->def.observation_points.push_back({name, x, y, z}); });
}

MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, const mhs_face_region_t* regions, size_t n_regions,
    mhs_fluid_bc_t kind, double value, const double* inlet_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(regions);
    MHS_TRY({
        if (!n_regions || !std::isfinite(value) || (inlet_temperature && !std::isfinite(*inlet_temperature)))
            throw std::invalid_argument("fluid boundary requires regions and finite values");
        mhs::model::FluidBoundarySpec fb;
        for (size_t i = 0; i < n_regions; ++i)
            fb.regions.push_back(_make_face_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
        const auto temperature = inlet_temperature ? std::optional<double>(*inlet_temperature) : std::nullopt;
        switch (kind) {
        case MHS_FLUID_PRESSURE_INLET:
            fb.condition = mhs::model::PressureInlet {value, temperature};
            break;
        case MHS_FLUID_MASS_FLOW_INLET:
            fb.condition = mhs::model::MassFlowInlet {value, temperature};
            break;
        case MHS_FLUID_OUTLET:
            fb.condition = mhs::model::PressureOutlet {value};
            break;
        default:
            throw std::invalid_argument("invalid fluid boundary kind");
        }
        m->def.fluid_boundaries.push_back(std::move(fb));
    });
}

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(const mhs_model_t* m, mhs_compiled_t** out)
{
    CHECK_NULL(out);
    *out = nullptr;
    CHECK_NULL(m);
    MHS_TRY({
        auto c = std::make_unique<mhs_compiled_t>();
        c->model = mhs::compiler::build_model(m->def);
        *out = c.release();
    });
}

MHS_API void mhs_compiled_destroy(mhs_compiled_t* c) { delete c; }

/* ------------------------------------------------------------------ */
/*  Compiled metadata                                                  */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_get_info(const mhs_compiled_t* c, mhs_compiled_info_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    out->cell_count = c->model.discrete.cells.size();
    out->face_count = c->model.discrete.faces.size();
    out->projection_count = c->model.source_terms.size();
    out->source_count = c->model.heat_source_table.size();
    out->initial_temperature = c->model.initial_temperature;

    mhs_detail_clear_last_error();
    return MHS_OK;
}

namespace {
    template <typename T>
    mhs_status_t copy_vector(const mhs::core::Vector<T>& source, T* out, size_t count, const char* label)
    {
        if (count != source.size()) {
            SET_ERR(label << " count must equal " << source.size());
            return MHS_ERROR;
        }
        if (count > 0 && !out) {
            SET_ERR("NULL pointer: out");
            return MHS_ERROR;
        }
        std::copy(source.begin(), source.end(), out);
        mhs_detail_clear_last_error();
        return MHS_OK;
    }

    mhs_status_t copy_csr_matrix(const mhs::core::CSRMatrix<double>& matrix, mhs_sparse_index_t* row_ptr,
        size_t row_ptr_count, mhs_sparse_index_t* col_idx, size_t col_idx_count, double* values, size_t value_count)
    {
        if (row_ptr_count != matrix.rows() + 1 || col_idx_count != matrix.nnz() || value_count != matrix.nnz()) {
            SET_ERR("CSR buffer sizes do not match operator dimensions");
            return MHS_ERROR;
        }
        if (!row_ptr || (matrix.nnz() && (!col_idx || !values))) {
            SET_ERR("NULL CSR output buffer");
            return MHS_ERROR;
        }
        std::copy(matrix.row_ptr().begin(), matrix.row_ptr().end(), row_ptr);
        std::copy(matrix.col_idx().begin(), matrix.col_idx().end(), col_idx);
        std::copy(matrix.values().begin(), matrix.values().end(), values);
        mhs_detail_clear_last_error();
        return MHS_OK;
    }
} // namespace

MHS_API mhs_status_t mhs_compiled_copy_cell_fields(const mhs_compiled_t* c, mhs_cell_fields_t* fields)
{
    CHECK_NULL(c);
    CHECK_NULL(fields);
    const auto& model = c->model;
    const auto& mesh = model.discrete;
    if (fields->cell_count != mesh.cells.size() || fields->face_count != mesh.faces.size()
        || fields->projection_count != model.source_terms.size()) {
        SET_ERR("CellFields buffer sizes do not match compiled model");
        return MHS_ERROR;
    }
    if (fields->cell_count && (!fields->cell_min || !fields->cell_max || !fields->material_id)) {
        SET_ERR("NULL cell field buffer");
        return MHS_ERROR;
    }
    if (fields->face_count
        && (!fields->face_minus || !fields->face_plus || !fields->face_axis || !fields->face_bounds)) {
        SET_ERR("NULL face field buffer");
        return MHS_ERROR;
    }
    if (fields->projection_count
        && (!fields->projection_cell || !fields->projection_source || !fields->projection_volume
            || !fields->projection_center)) {
        SET_ERR("NULL projection field buffer");
        return MHS_ERROR;
    }
    for (size_t cidx = 0; cidx < mesh.cells.size(); ++cidx) {
        for (size_t a = 0; a < 3; ++a) {
            fields->cell_min[3 * cidx + a] = mesh.cells[cidx].min[a];
            fields->cell_max[3 * cidx + a] = mesh.cells[cidx].max[a];
        }
        fields->material_id[cidx] = model.cell_material[cidx];
    }
    for (size_t id = 0; id < mesh.faces.size(); ++id) {
        const auto& face = mesh.faces[id];
        fields->face_minus[id] = face.minus;
        fields->face_plus[id] = face.plus;
        fields->face_axis[id] = static_cast<mhs_axis_t>(face.axis);
        for (size_t a = 0; a < 4; ++a)
            fields->face_bounds[4 * id + a] = face.bounds[a];
    }
    for (size_t id = 0; id < model.source_terms.size(); ++id) {
        const auto& term = model.source_terms[id];
        fields->projection_cell[id] = term.cell;
        fields->projection_source[id] = term.source;
        fields->projection_volume[id] = term.weight;
        for (size_t a = 0; a < 3; ++a)
            fields->projection_center[3 * id + a] = term.point[a];
    }
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_compiled_eval_source_terms(const mhs_compiled_t* c, const double* state, size_t state_count,
    double time, double* powers, size_t projection_count)
{
    CHECK_NULL(c);
    CHECK_NULL(state);
    if (projection_count) {
        CHECK_NULL(powers);
    }
    MHS_TRY({
        const auto& model = c->model;
        if (state_count != model.discrete.cells.size() || projection_count != model.source_terms.size())
            throw std::invalid_argument("source projection buffer size mismatch");
        for (size_t i = 0; i < projection_count; ++i) {
            const auto& term = model.source_terms[i];
            const auto& p = term.point;
            powers[i]
                = term.weight * model.heat_source_table[term.source].eval({p[0], p[1], p[2], state[term.cell], time});
        }
    });
}

MHS_API mhs_status_t mhs_compiled_eval_materials(const mhs_compiled_t* c, const double* temperature,
    size_t temperature_count, double time, mhs_material_values_t* values)
{
    CHECK_NULL(c);
    CHECK_NULL(temperature);
    CHECK_NULL(values);
    const auto& model = c->model;

    const auto& mesh = model.discrete;
    if (temperature_count != model.discrete.cells.size() || values->count != temperature_count) {
        SET_ERR("material evaluation buffer sizes do not match compiled model");
        return MHS_ERROR;
    }
    if (!values->conductivity_x || !values->conductivity_y || !values->conductivity_z || !values->density
        || !values->specific_heat) {
        SET_ERR("NULL material evaluation output buffer");
        return MHS_ERROR;
    }
    MHS_TRY({
        for (std::size_t cell = 0; cell < mesh.cells.size(); ++cell) {
            const auto p = mesh.cells[cell].center();
            const auto& props = model.material_table[model.cell_material[cell]];
            const auto dof = cell;
            const mhs::expression::FieldContext context {p[0], p[1], p[2], temperature[dof], time};
            values->conductivity_x[cell] = props.kx.eval(context);
            values->conductivity_y[cell] = props.ky.eval(context);
            values->conductivity_z[cell] = props.kz.eval(context);
            values->density[cell] = props.rho.eval(context);
            values->specific_heat[cell] = props.c.eval(context);
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Assembly                                                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* temperature, size_t temperature_count, double time, mhs_operators_t** out)
{
    CHECK_NULL(out);
    *out = nullptr;
    CHECK_NULL(c);
    CHECK_NULL(temperature);
    MHS_TRY({
        const auto cell_count = c->model.discrete.cells.size();
        if (temperature_count != cell_count) {
            SET_ERR("temperature_count must equal cell_count");
            return MHS_ERROR;
        }

        auto result = std::make_unique<mhs_operators_t>();
        result->operators
            = mhs::solver::assemble_thermal(c->model, std::span<const double>(temperature, temperature_count), time);
        *out = result.release();
    });
}

MHS_API void mhs_operators_destroy(mhs_operators_t* operators) { delete operators; }

MHS_API mhs_status_t mhs_operators_get_info(const mhs_operators_t* operators, mhs_operators_info_t* out)
{
    CHECK_NULL(operators);
    CHECK_NULL(out);
    out->state_count = operators->operators.f.size();
    out->k_nnz = operators->operators.K.nnz();
    out->c_nnz = operators->operators.C.nnz();
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_operators_copy_k(const mhs_operators_t* operators, mhs_sparse_index_t* row_ptr,
    size_t row_ptr_count, mhs_sparse_index_t* col_idx, size_t col_idx_count, double* values, size_t value_count)
{
    CHECK_NULL(operators);
    return copy_csr_matrix(operators->operators.K, row_ptr, row_ptr_count, col_idx, col_idx_count, values, value_count);
}

MHS_API mhs_status_t mhs_operators_copy_c(const mhs_operators_t* operators, mhs_sparse_index_t* row_ptr,
    size_t row_ptr_count, mhs_sparse_index_t* col_idx, size_t col_idx_count, double* values, size_t value_count)
{
    CHECK_NULL(operators);
    return copy_csr_matrix(operators->operators.C, row_ptr, row_ptr_count, col_idx, col_idx_count, values, value_count);
}

MHS_API mhs_status_t mhs_operators_copy_rhs(const mhs_operators_t* operators, double* out, size_t count)
{
    CHECK_NULL(operators);
    const auto expected_count = operators->operators.f.size();
    if (count != expected_count) {
        SET_ERR("rhs count must equal " << expected_count);
        return MHS_ERROR;
    }
    if (count > 0 && !out) {
        SET_ERR("NULL pointer: out");
        return MHS_ERROR;
    }
    std::copy_n(operators->operators.f.data(), count, out);
    mhs_detail_clear_last_error();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  SolveOptions conversion helper                                     */
/* ------------------------------------------------------------------ */

mhs::solver::SolveOptions to_solve_options(const mhs_solve_options_t* opts)
{
    mhs::solver::SolveOptions so;
    if (!opts)
        return so;
    so.linear_solver = _to_solver_type(opts->solver_type);
    so.linear_tolerance = opts->linear_tolerance;
    so.linear_max_iterations = opts->linear_max_iterations;
    so.underrelaxation = opts->underrelaxation;
    so.nonlinear_max_iterations = opts->nonlinear_max_iterations;
    so.nonlinear_relative_tolerance = opts->nonlinear_relative_tolerance;
    so.nonlinear_absolute_tolerance = opts->nonlinear_absolute_tolerance;
    so.time_tolerance = opts->time_tolerance;
    if (!std::isfinite(opts->max_time_step) || opts->max_time_step < 0.0)
        throw std::invalid_argument("max_time_step must be finite and nonnegative (zero selects automatic)");
    if (opts->max_time_step > 0.0)
        so.max_time_step = opts->max_time_step;
    return so;
}

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API void mhs_study_options_default(mhs_study_options_t* out)
{
    if (!out)
        return;
    const mhs::solver::StudyOptions defaults;
    *out = {MHS_STUDY_STEADY, defaults.duration, defaults.output_interval};
}

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const mhs_study_options_t* study, const double* state,
    size_t state_count, const mhs_solve_options_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(out);
    *out = nullptr;
    CHECK_NULL(c);
    MHS_TRY({
        auto so = to_solve_options(opts);

        if ((!state && state_count != 0) || (state && state_count != c->model.discrete.cells.size()))
            throw std::invalid_argument("initial state must be NULL/0 or have cell_count entries");
        const std::span<const double> init_span(state, state_count);

        mhs::solver::StudyOptions request;
        if (study)
            request = {_to_model_study(study->type), study->duration, study->output_interval};
        auto sol = mhs::solver::solve(c->model, request, init_span, so);

        auto* s = new (std::nothrow) mhs_solution_t;
        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERROR;
        }
        s->sol = std::move(sol);
        *out = s;
    });
}

MHS_API void mhs_solution_destroy(mhs_solution_t* s) { delete s; }

/* ------------------------------------------------------------------ */
/*  Solution copy-out accessors                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_get_info(const mhs_solution_t* s, mhs_solution_info_t* out)
{
    CHECK_NULL(s);
    CHECK_NULL(out);
    out->state_count = s->sol.state.size();
    out->record_count = s->sol.snapshot_times.size();
    out->probe_count = s->sol.probe_traces.size();
    out->time = s->sol.time;
    out->converged = s->sol.converged ? 1 : 0;
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_solution_copy_state(const mhs_solution_t* s, double* out, size_t count)
{
    CHECK_NULL(s);
    return copy_vector(s->sol.state, out, count, "state");
}

MHS_API mhs_status_t mhs_solution_copy_history_times(const mhs_solution_t* s, double* out, size_t count)
{
    CHECK_NULL(s);
    return copy_vector(s->sol.snapshot_times, out, count, "history times");
}

MHS_API mhs_status_t mhs_solution_copy_history_states(const mhs_solution_t* s, double* out, size_t count)
{
    CHECK_NULL(s);
    return copy_vector(s->sol.snapshot_states, out, count, "history states");
}

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_probe_get_info(
    const mhs_solution_t* s, size_t index, size_t* name_size, size_t* record_count)
{
    CHECK_NULL(s);
    CHECK_NULL(name_size);
    CHECK_NULL(record_count);
    if (index >= s->sol.probe_traces.size()) {
        SET_ERR("probe index out of range");
        return MHS_ERROR;
    }
    const auto& tr = s->sol.probe_traces[index];
    *name_size = tr.name.size() + 1;
    *record_count = s->sol.snapshot_times.size();
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_solution_copy_probe(
    const mhs_solution_t* s, size_t index, char* name, size_t name_size, double* values, size_t record_count)
{
    CHECK_NULL(s);
    CHECK_NULL(name);
    if (index >= s->sol.probe_traces.size()) {
        SET_ERR("probe index out of range");
        return MHS_ERROR;
    }
    const auto& tr = s->sol.probe_traces[index];
    if (name_size != tr.name.size() + 1 || record_count != s->sol.snapshot_times.size()) {
        SET_ERR("probe buffer sizes do not match probe data");
        return MHS_ERROR;
    }
    if (record_count > 0 && !values) {
        SET_ERR("NULL probe output buffer");
        return MHS_ERROR;
    }
    std::memcpy(name, tr.name.c_str(), name_size);
    std::copy(tr.values.begin(), tr.values.end(), values);
    mhs_detail_clear_last_error();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Logging                                                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_log_set_level(mhs_log_level_t level)
{
    if (level < MHS_LOG_LEVEL_TRACE || level > MHS_LOG_LEVEL_OFF) {
        SET_ERR("invalid logging level");
        return MHS_ERROR;
    }
    mhs::logger::set_log_level(static_cast<mhs::logger::Level>(level));
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_log_level_t mhs_log_get_level() { return static_cast<mhs_log_level_t>(mhs::logger::log_level()); }
