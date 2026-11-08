/* Implementation of the MetaHotspot C API. */
#include "metahotspot_c_api.h"

#include "compiler/model_compiler.hpp"
#include "logging/logger.hpp"
#include "model/model_definition.hpp"
#include "solver/assembler.hpp"
#include "solver/connection_flux.hpp"
#include "solver/solver.hpp"

#include "core/vector.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>

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
    mhs::model::CompiledModel model;
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

static mhs::model::SurfaceRegion _make_surface_region(mhs_axis_t axis, double coord, mhs_rect2d_t r)
{
    if (!std::isfinite(coord) || !std::isfinite(r.a_min) || !std::isfinite(r.a_max) || !std::isfinite(r.b_min)
        || !std::isfinite(r.b_max) || r.a_min >= r.a_max || r.b_min >= r.b_max)
        throw std::invalid_argument("surface region bounds must be finite and increasing");
    return {_to_axis(axis), coord, {r.a_min, r.a_max, r.b_min, r.b_max}};
}

template <typename Boundary> static void _add_thermal_boundary(
    mhs_model_t* m, mhs_index_t component, const mhs_surface_region_t* regions, size_t n_regions, Boundary condition)
{
    if (!n_regions)
        throw std::invalid_argument("surface regions must not be empty");
    mhs::model::ThermalBoundarySpec bp;
    bp.condition = std::move(condition);
    bp.regions.reserve(n_regions);
    for (size_t i = 0; i < n_regions; ++i)
        bp.regions.push_back(_make_surface_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
    m->def.component(component).thermal_boundaries.push_back(std::move(bp));
}

static mhs::solver::SolveOptions::LinearSolverType _to_solver_type(mhs_solver_type_t t)
{
    switch (t) {
    case MHS_SOLVER_PARDISO:
        return mhs::solver::SolveOptions::LinearSolverType::Pardiso;
    case MHS_SOLVER_AMG:
        return mhs::solver::SolveOptions::LinearSolverType::Amg;
    case MHS_SOLVER_SPARSE_LU:
        return mhs::solver::SolveOptions::LinearSolverType::SparseLu;
    }
    throw std::invalid_argument("invalid solver type: " + std::to_string(t));
}

/* ------------------------------------------------------------------ */
/*  Global helpers                                                     */
/* ------------------------------------------------------------------ */

MHS_API void mhs_study_default(mhs_study_t* study)
{
    if (!study)
        return;
    study->type = MHS_STUDY_STEADY;
    study->initial_temperature = 300.0;
    study->duration = 0.0;
    study->output_interval = 1.0;
}

MHS_API void mhs_solve_options_default(mhs_solve_options_t* opts)
{
    if (!opts)
        return;
    const mhs::solver::SolveOptions defaults {};
    opts->solver_type = defaults.linear_solver == mhs::solver::SolveOptions::LinearSolverType::Pardiso
        ? MHS_SOLVER_PARDISO
        : defaults.linear_solver == mhs::solver::SolveOptions::LinearSolverType::SparseLu ? MHS_SOLVER_SPARSE_LU
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

static mhs::model::Study to_study(const mhs_study_t& value)
{
    mhs::model::Study study;
    switch (value.type) {
    case MHS_STUDY_STEADY:
        study.type = mhs::model::StudyType::Steady;
        break;
    case MHS_STUDY_TRANSIENT:
        study.type = mhs::model::StudyType::Transient;
        break;
    default:
        throw std::invalid_argument("invalid study type");
    }
    if (!std::isfinite(value.initial_temperature) || !std::isfinite(value.duration)
        || !std::isfinite(value.output_interval) || value.output_interval < 0
        || (study.type == mhs::model::StudyType::Transient && value.duration <= 0))
        throw std::invalid_argument("invalid study temperature, duration or output interval");
    study.initial_temperature = value.initial_temperature;
    study.duration = value.duration;
    study.output_interval = value.output_interval;
    return study;
}

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

MHS_API mhs_status_t mhs_model_add_cartesian_component(mhs_model_t* m, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(out_id);
    MHS_TRY({
        const auto id = static_cast<mhs_index_t>(m->def.components.size());
        m->def.components.emplace_back();
        *out_id = id;
    });
}
MHS_API mhs_status_t mhs_model_add_reduced_component(mhs_model_t* m, size_t order, const double* stiffness,
    const double* capacity, const double* rhs, const double* initial, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(out_id);
    CHECK_NULL(stiffness);
    CHECK_NULL(capacity);
    CHECK_NULL(rhs);
    MHS_TRY({
        if (!order || order > static_cast<size_t>(std::numeric_limits<mhs_index_t>::max())
            || order > static_cast<size_t>(std::numeric_limits<mhs::core::SparseIndex>::max()) / order)
            throw std::invalid_argument("invalid ROM order");
        const auto id = static_cast<mhs_index_t>(m->def.components.size());
        mhs::model::RomDomainDefinition rom;
        rom.order = order;
        rom.K.assign(stiffness, stiffness + order * order);
        rom.C.assign(capacity, capacity + order * order);
        rom.f.assign(rhs, rhs + order);
        if (initial)
            rom.initial_state.assign(initial, initial + order);
        auto& component = m->def.components.emplace_back();
        component.physics.emplace<mhs::model::RomDomainDefinition>(std::move(rom));
        *out_id = id;
    });
}

MHS_API mhs_status_t mhs_model_add_reduced_affine_term(mhs_model_t* m, mhs_index_t component, const char* name,
    double coefficient, const double* stiffness, const double* rhs)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(stiffness);
    CHECK_NULL(rhs);
    MHS_TRY({
        auto& rom = m->def.rom(component);
        if (!std::isfinite(coefficient))
            throw std::invalid_argument("ROM parameter must have a finite value");
        mhs::model::RomAffineTerm term;
        term.name = name;
        term.value = coefficient;
        term.K.assign(stiffness, stiffness + rom.order * rom.order);
        term.f.assign(rhs, rhs + rom.order);
        rom.affine_terms.push_back(std::move(term));
    });
}

MHS_API mhs_status_t mhs_model_add_port(mhs_model_t* m, mhs_index_t component, const char* name,
    const mhs_surface_region_t* region, int32_t outward_sign, const mhs_rom_port_patch_t* patches, size_t patch_count,
    mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(region);
    CHECK_NULL(out_id);
    MHS_TRY({
        auto& owner = m->def.component(component);
        auto& ports = owner.ports;
        mhs::model::PortDefinition value;
        value.name = name;
        value.region = _make_surface_region(region->axis, region->coordinate, region->rectangle);
        if (const auto* rom = std::get_if<mhs::model::RomDomainDefinition>(&owner.physics)) {
            if (!patch_count || !patches || (outward_sign != 1 && outward_sign != -1))
                throw std::invalid_argument("reduced port requires patches and outward sign +/-1");
            value.outward_sign = static_cast<std::int8_t>(outward_sign);
            value.patches.reserve(patch_count);
            for (std::size_t i = 0; i < patch_count; ++i) {
                const auto& source = patches[i];
                if (!source.trace)
                    throw std::invalid_argument("NULL reduced port patch trace");
                mhs::model::RomPortPatch patch;
                patch.rectangle
                    = {source.rectangle.a_min, source.rectangle.a_max, source.rectangle.b_min, source.rectangle.b_max};
                patch.half_resistance = source.half_resistance;
                patch.center = {source.center[0], source.center[1]};
                patch.offset = source.offset;
                patch.trace.assign(source.trace, source.trace + rom->order);
                if (source.gradient_a)
                    patch.tangential_gradient[0].assign(source.gradient_a, source.gradient_a + rom->order);
                if (source.gradient_b)
                    patch.tangential_gradient[1].assign(source.gradient_b, source.gradient_b + rom->order);
                value.patches.push_back(std::move(patch));
            }
        }
        else if (outward_sign != 0 || patches || patch_count)
            throw std::invalid_argument("Cartesian ports derive their physical traces from geometry");
        if (ports.size() >= std::numeric_limits<mhs_index_t>::max())
            throw std::length_error("too many component ports");
        const auto id = static_cast<mhs_index_t>(ports.size());
        ports.push_back(std::move(value));
        *out_id = id;
    });
}

MHS_API mhs_status_t mhs_model_add_observation(
    mhs_model_t* m, mhs_index_t component, const char* name, const double xyz[3], const double* weights, double offset)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(xyz);
    MHS_TRY({
        auto& owner = m->def.component(component);
        if (!std::isfinite(offset) || !std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])
            || !*name)
            throw std::invalid_argument("observation must have finite values and a nonempty name");
        mhs::model::ObservationDefinition observation;
        observation.point = {name, xyz[0], xyz[1], xyz[2]};
        observation.offset = offset;
        if (const auto* rom = std::get_if<mhs::model::RomDomainDefinition>(&owner.physics)) {
            if (!weights)
                throw std::invalid_argument("reduced observation requires modal weights");
            observation.weights.assign(weights, weights + rom->order);
        }
        else if (weights || offset != 0.0)
            throw std::invalid_argument("Cartesian observation accepts only a physical point");
        owner.observations.push_back(std::move(observation));
    });
}

MHS_API mhs_status_t mhs_model_connect(mhs_model_t* m, mhs_index_t component_a, mhs_index_t port_a,
    mhs_index_t component_b, mhs_index_t port_b, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(out_id);
    MHS_TRY({
        const auto check_port = [&](mhs_index_t d, mhs_index_t p) { (void)m->def.component(d).ports.at(p); };
        check_port(component_a, port_a);
        check_port(component_b, port_b);
        const auto id = static_cast<mhs_index_t>(m->def.connections.size());
        m->def.connections.push_back({{component_a, port_a}, {component_b, port_b}});
        *out_id = id;
    });
}
MHS_API mhs_status_t mhs_compiled_connection_patch_count(const mhs_compiled_t* c, mhs_index_t connection, size_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    MHS_TRY({ *out = c->model.connections.at(connection).patches.size(); });
}
MHS_API mhs_status_t mhs_compiled_eval_connection_flux(const mhs_compiled_t* c, mhs_index_t connection,
    const double* state, size_t state_count, double time, double* outward_a, size_t patch_count)
{
    CHECK_NULL(c);
    CHECK_NULL(state);
    CHECK_NULL(outward_a);
    MHS_TRY({
        const auto& value = c->model.connections.at(connection);
        if (patch_count != value.patches.size())
            throw std::invalid_argument("connection flux buffer size mismatch");
        const auto flux = mhs::solver::evaluate_connection_flux(c->model, value, std::span(state, state_count), time);
        for (size_t p = 0; p < patch_count; ++p)
            outward_a[p] = flux[p].outward_a;
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_cartesian_mesh(mhs_model_t* m, mhs_index_t component, size_t nx, const double* x,
    size_t ny, const double* y, size_t nz, const double* z)
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
        m->def.fom(component).mesh = std::move(mesh);
    });
}

MHS_API mhs_status_t mhs_model_set_flow_mode(mhs_model_t* m, mhs_index_t component, mhs_flow_mode_t mode)
{
    CHECK_NULL(m);
    MHS_TRY({
        if (mode != MHS_FLOW_FROZEN && mode != MHS_FLOW_STOKES)
            throw std::invalid_argument("unknown flow mode");
        m->def.fom(component).flow_mode = static_cast<mhs::model::FlowMode>(mode);
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
        if (dynamic_viscosity) {
            if (!dynamic_viscosity[0])
                throw std::invalid_argument("dynamic viscosity expression must not be empty");
            spec.dynamic_viscosity = std::string(dynamic_viscosity);
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
MHS_API mhs_status_t mhs_model_add_volume(
    mhs_model_t* m, mhs_index_t component, mhs_index_t material, mhs_index_t source, mhs_index_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(out_id);
    MHS_TRY({
        if (material >= m->def.materials.size())
            throw std::out_of_range("material ID out of range");
        if (source != MHS_INVALID_INDEX && source >= m->def.sources.size())
            throw std::out_of_range("source ID out of range");
        *out_id = static_cast<mhs_index_t>(m->def.fom(component).volumes.size());
        m->def.fom(component).volumes.push_back({material, source, {}});
    });
}
MHS_API mhs_status_t mhs_model_volume_add_box(mhs_model_t* m, mhs_index_t component, mhs_index_t volume,
    mhs_geometry_op_t op, const double min_xyz[3], const double max_xyz[3])
{
    CHECK_NULL(m);
    CHECK_NULL(min_xyz);
    CHECK_NULL(max_xyz);
    MHS_TRY({
        if (volume >= m->def.fom(component).volumes.size())
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
        m->def.fom(component).volumes[volume].geometry.push_back(box);
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  atomic boundary conditions                 */
/* ------------------------------------------------------------------ */

namespace {
    mhs::model::ThermalBoundary to_thermal_boundary(const mhs_thermal_bc_t& condition)
    {
        if (!condition.expression)
            throw std::invalid_argument("thermal boundary expression is required");
        switch (condition.kind) {
        case MHS_THERMAL_DIRICHLET:
            return mhs::model::DirichletBoundary {condition.expression};
        case MHS_THERMAL_NEUMANN:
            return mhs::model::NeumannBoundary {condition.expression};
        case MHS_THERMAL_CONVECTION:
            if (!condition.ambient_temperature)
                throw std::invalid_argument("convection requires an ambient temperature");
            return mhs::model::ConvectionBoundary {condition.expression, condition.ambient_temperature};
        default:
            throw std::invalid_argument("unknown thermal boundary kind");
        }
    }
}

MHS_API mhs_status_t mhs_model_add_thermal_boundary(mhs_model_t* m, mhs_index_t component,
    const mhs_surface_region_t* regions, size_t count, const mhs_thermal_bc_t* condition)
{
    CHECK_NULL(m);
    CHECK_NULL(regions);
    CHECK_NULL(condition);
    MHS_TRY({ _add_thermal_boundary(m, component, regions, count, to_thermal_boundary(*condition)); });
}

MHS_API mhs_status_t mhs_model_set_default_thermal_boundary(
    mhs_model_t* m, mhs_index_t component, const mhs_thermal_bc_t* condition)
{
    CHECK_NULL(m);
    CHECK_NULL(condition);
    MHS_TRY({ m->def.fom(component).default_thermal_boundary = to_thermal_boundary(*condition); });
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
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_index_t component,
    const mhs_surface_region_t* regions, size_t n_regions, mhs_fluid_bc_t kind, double value,
    const double* inlet_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(regions);
    MHS_TRY({
        if (!n_regions || !std::isfinite(value) || (inlet_temperature && !std::isfinite(*inlet_temperature)))
            throw std::invalid_argument("fluid boundary requires regions and finite values");
        mhs::model::FluidBoundarySpec fb;
        for (size_t i = 0; i < n_regions; ++i)
            fb.regions.push_back(_make_surface_region(regions[i].axis, regions[i].coordinate, regions[i].rectangle));
        const auto temperature = inlet_temperature ? std::optional<double>(*inlet_temperature) : std::nullopt;
        switch (kind) {
        case MHS_FLUID_PRESSURE_INLET:
            fb.condition = mhs::model::PressureInlet {value, temperature};
            break;
        case MHS_FLUID_MASS_FLOW_INLET:
            fb.condition = mhs::model::MassFlowInlet {value, temperature};
            break;
        case MHS_FLUID_OUTLET:
            fb.condition = mhs::model::PressureOutlet {value, temperature};
            break;
        default:
            throw std::invalid_argument("invalid fluid boundary kind");
        }
        m->def.fom(component).fluid_boundaries.push_back(std::move(fb));
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

MHS_API mhs_status_t mhs_compiled_get_component_info(
    const mhs_compiled_t* c, mhs_index_t component, mhs_component_info_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    MHS_TRY({
        const auto& component_value = c->model.components.at(component);
        if (const auto* value = std::get_if<mhs::model::FomDomain>(&component_value.physics)) {
            out->kind = MHS_COMPONENT_CARTESIAN;
            out->cell_count = value->discrete.cells.size();
            out->face_count = value->discrete.faces.size();
        }
        else {
            out->kind = MHS_COMPONENT_REDUCED;
            out->cell_count = 0;
            out->face_count = 0;
        }
        out->field_count = static_cast<size_t>(std::count_if(c->model.layout.fields.begin(),
            c->model.layout.fields.end(), [&](const auto& f) { return f.domain == component; }));
    });
}

MHS_API mhs_status_t mhs_compiled_get_field_info(const mhs_compiled_t* c, mhs_index_t field, mhs_field_info_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    MHS_TRY({
        const auto& value = c->model.layout.field(field);
        *out = {value.domain, static_cast<mhs_field_quantity_t>(value.quantity),
            static_cast<mhs_entity_support_t>(value.support()), value.offset, value.count};
    });
}

MHS_API mhs_status_t mhs_compiled_copy_field_entities(
    const mhs_compiled_t* c, mhs_index_t field, mhs_index_t* entities, size_t count)
{
    CHECK_NULL(c);
    MHS_TRY({
        const auto& value = c->model.layout.field(field);
        if (count != value.count)
            throw std::invalid_argument("field entity count mismatch");
        if (count && !entities)
            throw std::invalid_argument("NULL field entity buffer");
        switch (value.quantity) {
        case mhs::model::FieldQuantity::Temperature:
        case mhs::model::FieldQuantity::ReducedCoordinate:
            for (std::size_t i = 0; i < count; ++i)
                entities[i] = static_cast<mhs_index_t>(i);
            break;
        case mhs::model::FieldQuantity::Pressure: {
            const auto& cells = c->model.fom(value.domain).fluid.fluid_to_global;
            std::copy(cells.begin(), cells.end(), entities);
            break;
        }
        case mhs::model::FieldQuantity::Velocity: {
            const auto& faces = c->model.fom(value.domain).fluid.velocity_faces;
            std::copy(faces.begin(), faces.end(), entities);
            break;
        }
        }
    });
}

MHS_API mhs_status_t mhs_compiled_get_info(const mhs_compiled_t* c, mhs_compiled_info_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    out->cell_count = out->face_count = out->projection_count = out->source_count = 0;
    for (const auto& field : c->model.layout.fields)
        if (field.quantity == mhs::model::FieldQuantity::Temperature) {
            const auto& fom = c->model.fom(field.domain);
            out->source_count = c->model.library.heat_source_table.size();
            out->cell_count += fom.discrete.cells.size();
            out->face_count += fom.discrete.faces.size();
            out->projection_count += fom.source_terms.size();
        }
    out->state_count = c->model.layout.state_count;
    out->component_count = c->model.components.size();
    out->field_count = c->model.layout.fields.size();

    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_compiled_default_state(
    const mhs_compiled_t* c, double initial_temperature, double* state, size_t state_count)
{
    CHECK_NULL(c);
    CHECK_NULL(state);
    MHS_TRY({
        mhs::solver::fill_default_state(c->model, initial_temperature, {state, state_count});
    });
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
    mhs_compiled_info_t info;
    mhs_compiled_get_info(c, &info);
    if (fields->cell_count != info.cell_count || fields->face_count != info.face_count
        || fields->projection_count != info.projection_count) {
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
    size_t cell_offset = 0, face_offset = 0, projection_offset = 0;
    for (const auto& field : c->model.layout.fields) {
        if (field.quantity != mhs::model::FieldQuantity::Temperature)
            continue;
        const auto& model = c->model.fom(field.domain);
        const auto& mesh = model.discrete;
        for (size_t cidx = 0; cidx < mesh.cells.size(); ++cidx) {
            for (size_t a = 0; a < 3; ++a) {
                fields->cell_min[3 * (cell_offset + cidx) + a] = mesh.cells[cidx].min[a];
                fields->cell_max[3 * (cell_offset + cidx) + a] = mesh.cells[cidx].max[a];
            }
            fields->material_id[cell_offset + cidx] = model.cell_material[cidx];
        }
        for (size_t id = 0; id < mesh.faces.size(); ++id) {
            const auto& face = mesh.faces[id];
            fields->face_minus[face_offset + id] = face.minus == MHS_INVALID_INDEX
                ? MHS_INVALID_INDEX
                : static_cast<mhs_index_t>(cell_offset + face.minus);
            fields->face_plus[face_offset + id] = face.plus == MHS_INVALID_INDEX
                ? MHS_INVALID_INDEX
                : static_cast<mhs_index_t>(cell_offset + face.plus);
            fields->face_axis[face_offset + id] = static_cast<mhs_axis_t>(face.axis);
            for (size_t a = 0; a < 4; ++a)
                fields->face_bounds[4 * (face_offset + id) + a] = face.bounds[a];
        }
        for (size_t id = 0; id < model.source_terms.size(); ++id) {
            const auto& term = model.source_terms[id];
            fields->projection_cell[projection_offset + id] = static_cast<mhs_index_t>(cell_offset + term.cell);
            fields->projection_source[projection_offset + id] = term.source;
            fields->projection_volume[projection_offset + id] = term.weight;
            for (size_t a = 0; a < 3; ++a)
                fields->projection_center[3 * (projection_offset + id) + a] = term.point[a];
        }
        cell_offset += mesh.cells.size();
        face_offset += mesh.faces.size();
        projection_offset += model.source_terms.size();
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
        mhs_compiled_info_t info;
        mhs_compiled_get_info(c, &info);
        if (state_count != c->model.layout.state_count || projection_count != info.projection_count)
            throw std::invalid_argument("source projection buffer size mismatch");
        size_t offset = 0;
        for (const auto& field : c->model.layout.fields) {
            if (field.quantity != mhs::model::FieldQuantity::Temperature)
                continue;
            const auto& model = c->model.fom(field.domain);
            for (const auto& term : model.source_terms) {
                const auto& p = term.point;
                const auto density
                    = c->model.library.heat_source_table[term.source].eval({p[0], p[1], p[2], state[field.dof(term.cell)], time});
                powers[offset++] = term.weight * density;
            }
        }
    });
}

MHS_API mhs_status_t mhs_compiled_eval_materials(
    const mhs_compiled_t* c, const double* state, size_t state_count, double time, mhs_material_values_t* values)
{
    CHECK_NULL(c);
    CHECK_NULL(state);
    CHECK_NULL(values);
    mhs_compiled_info_t info;
    mhs_compiled_get_info(c, &info);
    if (state_count != info.state_count || values->count != info.cell_count) {
        SET_ERR("material evaluation buffer sizes do not match compiled model");
        return MHS_ERROR;
    }
    if (!values->conductivity_x || !values->conductivity_y || !values->conductivity_z || !values->density
        || !values->specific_heat) {
        SET_ERR("NULL material evaluation output buffer");
        return MHS_ERROR;
    }
    MHS_TRY({
        size_t offset = 0;
        for (const auto& field : c->model.layout.fields) {
            if (field.quantity != mhs::model::FieldQuantity::Temperature)
                continue;
            const auto& model = c->model.fom(field.domain);
            const auto& mesh = model.discrete;
            for (std::size_t cell = 0; cell < mesh.cells.size(); ++cell) {
                const auto p = mesh.cells[cell].center();
                const auto& props = c->model.library.material_table[model.cell_material[cell]];
                const auto dof = field.dof(cell);
                const mhs::expression::FieldContext context {p[0], p[1], p[2], state[dof], time};
                values->conductivity_x[offset + cell] = props.kx.eval(context);
                values->conductivity_y[offset + cell] = props.ky.eval(context);
                values->conductivity_z[offset + cell] = props.kz.eval(context);
                values->density[offset + cell] = props.rho.eval(context);
                values->specific_heat[offset + cell] = props.c.eval(context);
            }
            offset += mesh.cells.size();
        }
    });
}

/* ------------------------------------------------------------------ */
/*  Assembly                                                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* state, size_t state_count, double time, mhs_operators_t** out)
{
    CHECK_NULL(out);
    *out = nullptr;
    CHECK_NULL(c);
    CHECK_NULL(state);
    MHS_TRY({
        if (state_count != c->model.layout.state_count) {
            SET_ERR("state_count must equal compiled state_count");
            return MHS_ERROR;
        }

        auto result = std::make_unique<mhs_operators_t>();
        result->operators = mhs::solver::assemble_thermal(c->model, std::span<const double>(state, state_count), time);
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

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const mhs_study_t* study, const double* state,
    size_t state_count, const mhs_solve_options_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(out);
    *out = nullptr;
    CHECK_NULL(c);
    MHS_TRY({
        auto so = to_solve_options(opts);
        const auto run = study ? to_study(*study) : mhs::model::Study {};

        if ((!state && state_count != 0) || (state && state_count != c->model.layout.state_count))
            throw std::invalid_argument("initial state must be NULL/0 or have state_count entries");
        const std::span<const double> init_span(state, state_count);

        auto sol = mhs::solver::solve(c->model, run, init_span, so);

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
