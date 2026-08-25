/* Implementation of the MetaHotspot C API. */
#include "api/metahotspot.h"
#include "api/internal.h"

#include "common/model_definition.hpp"
#include "common/mesh.hpp"
#include "common/solver.hpp"
#include "compiler/model_compiler.hpp"
#include "io/model_io.hpp"
#include "io/result_io.hpp"
#include "solver/assembler.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

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
{ return {_to_axis(axis), coord, {{r.a_min, r.a_max, r.b_min, r.b_max}}}; }

static mhs::sim::SolveOptions::LinearSolverType _to_solver_type(mhs_solver_type_t t)
{
    switch (t) {
    case MHS_SOLVER_PARDISO:
        return mhs::sim::SolveOptions::LinearSolverType::Pardiso;
    case MHS_SOLVER_AMG:
        return mhs::sim::SolveOptions::LinearSolverType::AmgCg;
    }
    throw std::invalid_argument("invalid solver type: " + std::to_string(t));
}

static mhs::sim::SolveOptions::Integrator _to_integrator(mhs_integrator_t integrator)
{
    switch (integrator) {
    case MHS_INTEGRATOR_BDF1:
        return mhs::sim::SolveOptions::Integrator::Bdf1;
    case MHS_INTEGRATOR_BDF2:
        return mhs::sim::SolveOptions::Integrator::Bdf2;
    }
    throw std::invalid_argument("invalid integrator: " + std::to_string(integrator));
}

static mhs::sim::SolveOptions::StepStrategy _to_step_strategy(mhs_step_strategy_t strategy)
{
    switch (strategy) {
    case MHS_STEP_ADAPTIVE:
        return mhs::sim::SolveOptions::StepStrategy::Adaptive;
    case MHS_STEP_FIXED:
        return mhs::sim::SolveOptions::StepStrategy::Fixed;
    }
    throw std::invalid_argument("invalid step strategy: " + std::to_string(strategy));
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

MHS_API void mhs_solve_options_default(mhs_solve_options_t* opts)
{
    if (!opts)
        return;
    opts->solver_type = MHS_SOLVER_AMG;
    opts->linear_tolerance = 1e-8;
    opts->linear_max_iterations = 1000;
    opts->underrelaxation = 1.0;
    opts->nonlinear_max_iterations = 200;
    opts->nonlinear_relative_tolerance = 1e-6;
    opts->nonlinear_absolute_tolerance = 1e-12;
    opts->integrator = MHS_INTEGRATOR_BDF1;
    opts->step_strategy = MHS_STEP_ADAPTIVE;
    opts->error_rel_tol = 1e-4;
    opts->error_safety = 0.9;
    opts->min_dt = 1e-12;
    opts->max_dt = 1.0;
    opts->fixed_dt = 1.0;
}

MHS_API const char* mhs_last_error(void) { return mhs_detail_last_error(); }

/* ------------------------------------------------------------------ */
/*  Model life-cycle                                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_create(mhs_model_t** out)
{
    CHECK_NULL(out);
    try {
        *out = new mhs_model_t {};
        mhs_detail_clear_last_error();
        return MHS_OK;
    }
    catch (const std::bad_alloc&) {
        *out = nullptr;
        SET_ERR("memory allocation failed");
        return MHS_ERR_OOM;
    }
}

MHS_API void mhs_model_destroy(mhs_model_t* m) { delete m; }

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

MHS_API mhs_status_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
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
    });
}

MHS_API mhs_status_t mhs_model_add_layer(
    mhs_model_t* m, const char* thickness, const char* x_offset, const char* y_offset, uint32_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(thickness);
    CHECK_NULL(x_offset);
    CHECK_NULL(y_offset);
    CHECK_NULL(out_id);
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        m->def.layers.push_back({thickness, x_offset, y_offset, {}});
        *out_id = static_cast<uint32_t>(m->def.layers.size() - 1);
    });
}

MHS_API mhs_status_t mhs_model_add_block(mhs_model_t* m, uint32_t layer, const char* material_name,
    const char* heat_source, const char* x_offset, const char* y_offset, const char* thickness, uint32_t* out_id)
{
    CHECK_NULL(m);
    CHECK_NULL(material_name);
    CHECK_NULL(out_id);
    if (layer >= m->def.layers.size()) {
        SET_ERR("layer ID out of range");
        return MHS_ERR_INVALID_ARG;
    }
    MHS_TRY(MHS_ERR_INVALID_ARG, {
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
        *out_id = static_cast<uint32_t>(m->block_locations.size() - 1);
    });
}

MHS_API mhs_status_t mhs_model_add_rect(mhs_model_t* m, uint32_t block, mhs_geometry_op_t op, const char* x,
    const char* y, const char* width, const char* height)
{
    CHECK_NULL(m);
    CHECK_NULL(x);
    CHECK_NULL(y);
    CHECK_NULL(width);
    CHECK_NULL(height);
    if (block >= m->block_locations.size()) {
        SET_ERR("block ID out of range");
        return MHS_ERR_INVALID_ARG;
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

MHS_API mhs_status_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(expression);
    MHS_TRY(
        MHS_ERR_INVALID_ARG, { m->def.functions.push_back({name, mhs::model::ExpressionFunctionSpec {expression}}); });
}

MHS_API mhs_status_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(MHS_ERR_INVALID_ARG,
        { m->def.functions.push_back({name, mhs::model::GaussFunctionSpec {amplitude, tau, center}}); });
}

MHS_API mhs_status_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(MHS_ERR_INVALID_ARG,
        { m->def.functions.push_back({name, mhs::model::SineFunctionSpec {amplitude, angular_frequency, phase}}); });
}

MHS_API mhs_status_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(MHS_ERR_INVALID_ARG,
        { m->def.functions.push_back({name, mhs::model::DoubleExponentialFunctionSpec {amplitude, alpha, beta}}); });
}

MHS_API mhs_status_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, size_t count)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(points);
    if (count < 2) {
        SET_ERR("piecewise requires count >= 2");
        return MHS_ERR_INVALID_ARG;
    }
    MHS_TRY(MHS_ERR_INVALID_ARG, {
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
    if (count < 1) {
        SET_ERR("periodic_piecewise_constant requires count >= 1");
        return MHS_ERR_INVALID_ARG;
    }
    if (period <= 0.0) {
        SET_ERR("period must be positive");
        return MHS_ERR_INVALID_ARG;
    }
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        mhs::model::PeriodicPiecewiseConstantFunctionSpec spec;
        spec.period = period;
        spec.values.assign(values, values + count);
        m->def.functions.push_back({name, std::move(spec)});
    });
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    MHS_TRY(MHS_ERR_INVALID_ARG,
        { m->def.observation_points.push_back({name, std::to_string(x), std::to_string(y), std::to_string(z)}); });
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
        c->model = std::make_shared<const mhs::core::Model>(std::move(core_model));
        *out = c;
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
    out->cell_count = c->model->cells.cell_to_grid.size();
    out->grid_count = c->model->cells.grid_to_cell.size();
    out->study_type = _from_core_study(c->model->study_type);
    out->initial_temperature = c->model->initial_temperature;

    out->nx = c->model->mesh.nx;
    out->ny = c->model->mesh.ny;
    out->nz = c->model->mesh.nz;
    mhs_detail_clear_last_error();
    return MHS_OK;
}

namespace {
    template <typename T>
    mhs_status_t copy_vector(const std::vector<T>& source, T* out, size_t count, const char* label)
    {
        if (count != source.size()) {
            SET_ERR(label << " count must equal " << source.size());
            return MHS_ERR_INVALID_ARG;
        }
        if (count > 0 && !out) {
            SET_ERR("NULL pointer: out");
            return MHS_ERR_NULL_PTR;
        }
        std::copy(source.begin(), source.end(), out);
        mhs_detail_clear_last_error();
        return MHS_OK;
    }

    Eigen::SparseMatrix<double> copy_csc(
        size_t n, const int32_t* outer, const int32_t* inner, const double* values, size_t nnz, const char* label)
    {
        if (!outer || (nnz > 0 && (!inner || !values)))
            throw std::invalid_argument(std::string("NULL pointer in ") + label);
        if (n > static_cast<size_t>(std::numeric_limits<int32_t>::max())
            || nnz > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
            throw std::invalid_argument(std::string(label) + " dimensions exceed int32 range");
        if (outer[0] != 0 || outer[n] != static_cast<int32_t>(nnz))
            throw std::invalid_argument(std::string("invalid ") + label + " outer-index range");
        std::vector<Eigen::Triplet<double>> entries;
        entries.reserve(nnz);
        for (size_t column = 0; column < n; ++column) {
            const auto begin = outer[column];
            const auto end = outer[column + 1];
            if (begin < 0 || end < begin || static_cast<size_t>(end) > nnz)
                throw std::invalid_argument(std::string("invalid ") + label + " column offsets");
            for (int32_t entry = begin; entry < end; ++entry) {
                const auto row = inner[entry];
                if (row < 0 || static_cast<size_t>(row) >= n)
                    throw std::invalid_argument(std::string("invalid ") + label + " row index");
                entries.emplace_back(row, static_cast<int32_t>(column), values[entry]);
            }
        }
        Eigen::SparseMatrix<double> matrix(static_cast<int32_t>(n), static_cast<int32_t>(n));
        matrix.setFromTriplets(entries.begin(), entries.end());
        matrix.makeCompressed();
        return matrix;
    }

    mhs_status_t copy_csc_matrix(const Eigen::SparseMatrix<double>& matrix, int32_t* outer, size_t outer_count,
        int32_t* inner, size_t inner_count, double* values, size_t value_count)
    {
        const auto expected_outer = static_cast<size_t>(matrix.cols()) + 1;
        const auto expected_nnz = static_cast<size_t>(matrix.nonZeros());
        if (outer_count != expected_outer || inner_count != expected_nnz || value_count != expected_nnz) {
            SET_ERR("CSC buffer sizes do not match operator dimensions");
            return MHS_ERR_INVALID_ARG;
        }
        if (!outer || (expected_nnz > 0 && (!inner || !values))) {
            SET_ERR("NULL CSC output buffer");
            return MHS_ERR_NULL_PTR;
        }
        std::copy_n(matrix.outerIndexPtr(), expected_outer, outer);
        std::copy_n(matrix.innerIndexPtr(), expected_nnz, inner);
        std::copy_n(matrix.valuePtr(), expected_nnz, values);
        mhs_detail_clear_last_error();
        return MHS_OK;
    }
}

MHS_API mhs_status_t mhs_compiled_copy_cell_fields(const mhs_compiled_t* c, mhs_cell_fields_t* fields)
{
    CHECK_NULL(c);
    CHECK_NULL(fields);
    const auto& cells = c->model->cells;
    const auto& mesh = c->model->mesh;
    if (fields->grid_count != cells.grid_to_cell.size() || fields->cell_count != cells.cell_to_grid.size()
        || fields->nx != mesh.nx || fields->ny != mesh.ny || fields->nz != mesh.nz) {
        SET_ERR("CellFields buffer sizes do not match compiled model");
        return MHS_ERR_INVALID_ARG;
    }
    auto status = copy_vector(cells.grid_to_cell, fields->grid_to_cell, fields->grid_count, "grid_to_cell");
    if (status != MHS_OK)
        return status;
    status = copy_vector(cells.cell_to_grid, fields->cell_to_grid, fields->cell_count, "cell_to_grid");
    if (status != MHS_OK)
        return status;
    status = copy_vector(mesh.dx, fields->dx, fields->nx, "dx");
    if (status != MHS_OK)
        return status;
    status = copy_vector(mesh.dy, fields->dy, fields->ny, "dy");
    if (status != MHS_OK)
        return status;
    status = copy_vector(mesh.dz, fields->dz, fields->nz, "dz");
    if (status != MHS_OK)
        return status;
    status = copy_vector(mesh.cx, fields->cx, fields->nx, "cx");
    if (status != MHS_OK)
        return status;
    status = copy_vector(mesh.cy, fields->cy, fields->ny, "cy");
    if (status != MHS_OK)
        return status;
    status = copy_vector(mesh.cz, fields->cz, fields->nz, "cz");
    if (status != MHS_OK)
        return status;
    status = copy_vector(cells.layer_id, fields->layer_id, fields->cell_count, "layer_id");
    if (status != MHS_OK)
        return status;
    status = copy_vector(cells.block_id, fields->block_id, fields->cell_count, "block_id");
    if (status != MHS_OK)
        return status;
    status = copy_vector(cells.material_id, fields->material_id, fields->cell_count, "material_id");
    if (status != MHS_OK)
        return status;
    status = copy_vector(cells.heat_source_idx, fields->heat_source_idx, fields->cell_count, "heat_source_idx");
    if (status != MHS_OK)
        return status;
    return MHS_OK;
}

MHS_API mhs_status_t mhs_compiled_eval_materials(const mhs_compiled_t* c, const double* temperature,
    size_t temperature_count, double time, mhs_material_values_t* values)
{
    CHECK_NULL(c);
    CHECK_NULL(temperature);
    CHECK_NULL(values);
    const auto& model = *c->model;
    const auto& cells = model.cells;
    const auto& mesh = model.mesh;
    if (temperature_count != cells.cell_to_grid.size() || values->count != temperature_count) {
        SET_ERR("material evaluation buffer sizes do not match compiled model");
        return MHS_ERR_INVALID_ARG;
    }
    if (!values->conductivity_x || !values->conductivity_y || !values->conductivity_z || !values->density
        || !values->specific_heat) {
        SET_ERR("NULL material evaluation output buffer");
        return MHS_ERR_NULL_PTR;
    }
    for (mhs::core::Index cell = 0; cell < cells.cell_to_grid.size(); ++cell) {
        const auto grid = cells.cell_to_grid[cell];
        mhs::core::Index ix, iy, iz;
        mhs::utils::decode_index(grid, mesh.ny, mesh.nz, ix, iy, iz);
        const auto& props = model.material_table[cells.material_id[cell]];
        const mhs::core::FieldContext context {
            mesh.cx[ix], mesh.cy[iy], mesh.cz[iz], temperature[cell], time};
        values->conductivity_x[cell] = props.kx.eval(context);
        values->conductivity_y[cell] = props.ky.eval(context);
        values->conductivity_z[cell] = props.kz.eval(context);
        values->density[cell] = props.rho.eval(context);
        values->specific_heat[cell] = props.c.eval(context);
    }
    mhs_detail_clear_last_error();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Assembly                                                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* temperature, size_t temperature_count, double time, mhs_operators_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(temperature);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_ASSEMBLE, {
        const auto cell_count = c->model->cells.cell_to_grid.size();
        if (temperature_count != cell_count) {
            SET_ERR("temperature_count must equal cell_count");
            return MHS_ERR_INVALID_ARG;
        }

        *out = nullptr;
        auto result = std::make_unique<mhs_operators_t>();
        result->operators
            = mhs::sim::assemble_thermal(*c->model, std::span<const double>(temperature, temperature_count), time);
        *out = result.release();
    });
}

MHS_API mhs_status_t mhs_operators_create(size_t state_count, const int32_t* k_outer, const int32_t* k_inner,
    const double* k_values, size_t k_nnz, const int32_t* c_outer, const int32_t* c_inner, const double* c_values,
    size_t c_nnz, const double* rhs, mhs_operators_t** out)
{
    CHECK_NULL(out);
    CHECK_NULL(rhs);
    *out = nullptr;
    MHS_TRY(MHS_ERR_INVALID_ARG, {
        auto result = std::make_unique<mhs_operators_t>();
        result->operators.K = copy_csc(state_count, k_outer, k_inner, k_values, k_nnz, "K");
        result->operators.C = copy_csc(state_count, c_outer, c_inner, c_values, c_nnz, "C");
        result->operators.f = Eigen::Map<const Eigen::VectorXd>(rhs, static_cast<Eigen::Index>(state_count));
        *out = result.release();
    });
}

MHS_API void mhs_operators_destroy(mhs_operators_t* operators) { delete operators; }

MHS_API mhs_status_t mhs_operators_get_info(const mhs_operators_t* operators, mhs_operators_info_t* out)
{
    CHECK_NULL(operators);
    CHECK_NULL(out);
    out->state_count = static_cast<size_t>(operators->operators.f.size());
    out->k_nnz = static_cast<size_t>(operators->operators.K.nonZeros());
    out->c_nnz = static_cast<size_t>(operators->operators.C.nonZeros());
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_operators_copy_k(const mhs_operators_t* operators, int32_t* outer, size_t outer_count,
    int32_t* inner, size_t inner_count, double* values, size_t value_count)
{
    CHECK_NULL(operators);
    return copy_csc_matrix(operators->operators.K, outer, outer_count, inner, inner_count, values, value_count);
}

MHS_API mhs_status_t mhs_operators_copy_c(const mhs_operators_t* operators, int32_t* outer, size_t outer_count,
    int32_t* inner, size_t inner_count, double* values, size_t value_count)
{
    CHECK_NULL(operators);
    return copy_csc_matrix(operators->operators.C, outer, outer_count, inner, inner_count, values, value_count);
}

MHS_API mhs_status_t mhs_operators_copy_rhs(const mhs_operators_t* operators, double* out, size_t count)
{
    CHECK_NULL(operators);
    return copy_vector(
        std::vector<double>(operators->operators.f.data(), operators->operators.f.data() + count), out, count, "rhs");
}

/* ------------------------------------------------------------------ */
/*  SolveOptions conversion helper                                     */
/* ------------------------------------------------------------------ */

mhs::sim::SolveOptions to_solve_options(const mhs_solve_options_t* opts, double transient_duration)
{
    mhs::sim::SolveOptions so;
    if (!opts)
        return so;
    so.linear_solver = _to_solver_type(opts->solver_type);
    so.linear_tolerance = opts->linear_tolerance;
    so.linear_max_iterations = opts->linear_max_iterations;
    so.underrelaxation = opts->underrelaxation;
    so.nonlinear_max_iterations = opts->nonlinear_max_iterations;
    so.nonlinear_relative_tolerance = opts->nonlinear_relative_tolerance;
    so.nonlinear_absolute_tolerance = opts->nonlinear_absolute_tolerance;
    so.integrator = _to_integrator(opts->integrator);
    so.step_strategy = _to_step_strategy(opts->step_strategy);
    so.error_rel_tol = opts->error_rel_tol;
    so.error_safety = opts->error_safety;
    so.min_dt = opts->min_dt;
    so.max_dt = (opts->max_dt > 0.0) ? opts->max_dt : (transient_duration > 0.0) ? transient_duration : 1.0;
    so.fixed_dt = opts->fixed_dt;
    return so;
}

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const double* state, size_t state_count,
    const mhs_solve_options_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_SOLVE, {
        auto so = to_solve_options(opts, c->model->transient_duration);

        // Build initial state span
        std::span<const double> init_span;
        std::vector<double> owned;
        if (state && state_count > 0) {
            owned.assign(state, state + state_count);
            init_span = owned;
        }

        auto sol = mhs::sim::solve(*c->model, init_span, so);

        auto* s = new (std::nothrow) mhs_solution_t;
        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        s->sol = std::move(sol);
        s->model = c->model;
        *out = s;
    });
}

MHS_API void mhs_solution_destroy(mhs_solution_t* s) { delete s; }

/* ------------------------------------------------------------------ */
/*  VTU export                                                         */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_write_vtu(const mhs_solution_t* s, const char* path)
{
    CHECK_NULL(s);
    CHECK_NULL(path);
    MHS_TRY(MHS_ERR_IO, {
        if (!s->model)
            throw std::invalid_argument("solution does not own a compiled runtime model");
        if (s->sol.fvm_count != s->model->cells.cell_to_grid.size())
            throw std::invalid_argument("solution FVM state does not match its runtime model");
        mhs::io::write_vtu(path, *s->model, std::span<const double>(s->sol.state.data(), s->sol.fvm_count));
    });
}

/* ------------------------------------------------------------------ */
/*  Solution copy-out accessors                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_get_info(const mhs_solution_t* s, mhs_solution_info_t* out)
{
    CHECK_NULL(s);
    CHECK_NULL(out);
    out->fvm_count = s->sol.fvm_count;
    out->state_count = s->sol.state.size();
    out->record_count = s->sol.snapshot_times.size();
    out->probe_count = s->sol.probe_traces.size();
    out->time = s->sol.time;
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
        return MHS_ERR_INVALID_ARG;
    }
    const auto& tr = s->sol.probe_traces[index];
    if (tr.times.size() != tr.values.size()) {
        SET_ERR("probe storage is inconsistent");
        return MHS_ERR_RUNTIME;
    }
    *name_size = tr.name.size() + 1;
    *record_count = tr.times.size();
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_solution_copy_probe(const mhs_solution_t* s, size_t index, char* name, size_t name_size,
    double* times, double* values, size_t record_count)
{
    CHECK_NULL(s);
    CHECK_NULL(name);
    if (index >= s->sol.probe_traces.size()) {
        SET_ERR("probe index out of range");
        return MHS_ERR_INVALID_ARG;
    }
    const auto& tr = s->sol.probe_traces[index];
    if (name_size != tr.name.size() + 1 || record_count != tr.times.size() || tr.times.size() != tr.values.size()) {
        SET_ERR("probe buffer sizes do not match probe data");
        return MHS_ERR_INVALID_ARG;
    }
    if (record_count > 0 && (!times || !values)) {
        SET_ERR("NULL probe output buffer");
        return MHS_ERR_NULL_PTR;
    }
    std::memcpy(name, tr.name.c_str(), name_size);
    std::copy(tr.times.begin(), tr.times.end(), times);
    std::copy(tr.values.begin(), tr.values.end(), values);
    mhs_detail_clear_last_error();
    return MHS_OK;
}
