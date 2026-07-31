/* Implementation of the MetaHotspot C API. */
#include "api/metahotspot.h"
#include "api/internal.h"

#include "compiler/model_compiler.hpp"
#include "io/model_io.hpp"
#include "io/result_io.hpp"
#include "common/model_definition.hpp"
#include "common/solver.hpp"
#include "solver/assembler.hpp"

#include "common/mesh.hpp"
#include <optional>
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
{
    return {_to_axis(axis), coord, {{r.a_min, r.a_max, r.b_min, r.b_max}}};
}

static mhs::sim::SolveOptions::LinearSolverType _to_solver_type(mhs_solver_type_t t)
{
    switch (t) {
    case MHS_SOLVER_PARDISO:
        return mhs::sim::SolveOptions::LinearSolverType::Pardiso;
    case MHS_SOLVER_EIGEN_SPARSE_LU:
        return mhs::sim::SolveOptions::LinearSolverType::EigenSparseLU;
    case MHS_SOLVER_EIGEN_BICGSTAB:
        return mhs::sim::SolveOptions::LinearSolverType::EigenBiCGSTAB;
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
    opts->solver_type = MHS_SOLVER_PARDISO;
    opts->linear_tolerance = 1e-8;
    opts->linear_max_iterations = 1000;
    opts->underrelaxation = 1.0;
    opts->nonlinear_max_iterations = 200;
    opts->nonlinear_relative_tolerance = 1e-6;
    opts->nonlinear_absolute_tolerance = 1e-12;
    opts->integrator = MHS_INTEGRATOR_BDF1;
    opts->step_strategy = MHS_STEP_ADAPTIVE;
    opts->error_abs_tol = 1e-4;
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
        c->model = std::move(core_model);
        *out = c;
    });
}

MHS_API void mhs_compiled_destroy(mhs_compiled_t* c) { delete c; }

/* ------------------------------------------------------------------ */
/*  Compiled metadata                                                  */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_metadata(const mhs_compiled_t* c, mhs_compiled_metadata_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    out->cell_count = c->model.cells.cell_to_grid.size();
    out->study_type = _from_core_study(c->model.study_type);
    out->initial_temperature = c->model.initial_temperature;

    out->nx = c->model.mesh.nx;
    out->ny = c->model.mesh.ny;
    out->nz = c->model.mesh.nz;
    out->grid_to_cell = c->model.cells.grid_to_cell.data();
    out->layer_ids = c->model.cells.layer_id.data();
    out->block_ids = c->model.cells.block_id.data();
    mhs_detail_clear_last_error();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Assembly                                                            */
/* ------------------------------------------------------------------ */

static void _eigen_to_csc_view(const Eigen::SparseMatrix<double>& mat, mhs_csc_view_t* out)
{
    out->rows = static_cast<int32_t>(mat.rows());
    out->columns = static_cast<int32_t>(mat.cols());
    out->nnz = static_cast<int32_t>(mat.nonZeros());
    out->outer_indices = mat.outerIndexPtr();
    out->inner_indices = mat.innerIndexPtr();
    out->values = mat.valuePtr();
}

MHS_API mhs_status_t mhs_compiled_half_conductance(const mhs_compiled_t* c, const size_t* cells, mhs_face_t face,
    double temperature, double time, double* out, size_t n)
{
    CHECK_NULL(c);
    CHECK_NULL(cells);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_ASSEMBLE, {
        if (static_cast<int>(face) < 0 || static_cast<int>(face) >= 6) {
            throw std::invalid_argument("invalid face direction");
        }
        const auto& model = c->model;
        const auto face_dir = static_cast<mhs::core::FaceDir>(static_cast<int>(face));
        const auto cell_count = model.cells.cell_to_grid.size();
        for (size_t i = 0; i < n; ++i) {
            const auto cell = cells[i];
            if (cell >= cell_count) {
                throw std::out_of_range("cell index " + std::to_string(cell)
                    + " out of range (cell_count=" + std::to_string(cell_count) + ")");
            }
            const auto grid = model.cells.cell_to_grid[cell];
            mhs::core::Index ix, iy, iz;
            mhs::utils::decode_index(grid, model.mesh.ny, model.mesh.nz, ix, iy, iz);
            const auto& material = model.material_table[model.cells.material_id[cell]];
            const mhs::core::FieldContext ctx {
                model.mesh.cx[ix], model.mesh.cy[iy], model.mesh.cz[iz], temperature, time};
            const double k
                = mhs::utils::k_along(face_dir, material.kx.eval(ctx), material.ky.eval(ctx), material.kz.eval(ctx));
            const double area
                = mhs::utils::face_area(face_dir, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            const double half_len
                = mhs::utils::half_length_along(face_dir, model.mesh.dx[ix], model.mesh.dy[iy], model.mesh.dz[iz]);
            out[i] = (k > 0.0 && half_len > 0.0) ? k * area / half_len : 0.0;
        }
    });
}

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* temperature, size_t temperature_count, double time, mhs_operators_t* out)
{
    CHECK_NULL(c);
    CHECK_NULL(temperature);
    CHECK_NULL(out);
    MHS_TRY(MHS_ERR_ASSEMBLE, {
        const auto cell_count = c->model.cells.cell_to_grid.size();
        if (temperature_count != cell_count) {
            SET_ERR("temperature_count must equal cell_count");
            return MHS_ERR_INVALID_ARG;
        }

        // Reuse scratch buffer for the assembly result
        auto& ops = const_cast<mhs_compiled_t*>(c)->assemble_scratch;
        ops = mhs::sim::assemble_thermal(c->model, std::span<const double>(temperature, temperature_count), time);

        _eigen_to_csc_view(ops.K, &out->K);
        _eigen_to_csc_view(ops.C, &out->C);
        out->rhs = ops.f.data();
        out->n = static_cast<size_t>(ops.f.size());
    });
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
    so.error_abs_tol = opts->error_abs_tol;
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
        auto so = to_solve_options(opts, c->model.transient_duration);

        // Build initial state span
        std::span<const double> init_span;
        std::vector<double> owned;
        if (state && state_count > 0) {
            owned.assign(state, state + state_count);
            init_span = owned;
        }

        auto sol = mhs::sim::solve(c->model, init_span, so);

        auto* s = new (std::nothrow) mhs_solution_t;
        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        s->sol = std::move(sol);
        *out = s;
    });
}

MHS_API void mhs_solution_destroy(mhs_solution_t* s) { delete s; }

/* ------------------------------------------------------------------ */
/*  VTU export                                                         */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_write_vtu(const mhs_compiled_t* c, const mhs_solution_t* s, const char* path)
{
    CHECK_NULL(c);
    CHECK_NULL(s);
    CHECK_NULL(path);
    MHS_TRY(MHS_ERR_IO, {
        if (s->sol.fvm_count != c->model.cells.cell_to_grid.size()) {
            throw std::invalid_argument("solution FVM state does not match compiled model");
        }
        mhs::io::write_vtu(path, c->model, std::span<const double>(s->sol.state.data(), s->sol.fvm_count));
    });
}

/* ------------------------------------------------------------------ */
/*  Solution views                                                     */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_solution_view(const mhs_solution_t* s, mhs_solution_view_t* out)
{
    CHECK_NULL(s);
    CHECK_NULL(out);
    out->fvm_count = s->sol.fvm_count;
    out->state_count = s->sol.state.size();
    out->time = s->sol.time;
    out->state = s->sol.state.data();
    mhs_detail_clear_last_error();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_solution_history_view(
    const mhs_solution_t* solution, mhs_solution_history_view_t* out)
{
    CHECK_NULL(solution);
    CHECK_NULL(out);

    const auto state_count = solution->sol.state.size();
    const auto record_count = solution->sol.snapshot_times.size();
    if (record_count > 0 && solution->sol.snapshot_states.size() != record_count * state_count) {
        SET_ERR("solution history storage is inconsistent");
        return MHS_ERR_RUNTIME;
    }

    out->times = record_count > 0 ? solution->sol.snapshot_times.data() : nullptr;
    out->states = record_count > 0 ? solution->sol.snapshot_states.data() : nullptr;
    out->record_count = record_count;
    out->state_count = state_count;
    mhs_detail_clear_last_error();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API size_t mhs_solution_probe_count(const mhs_solution_t* s)
{
    if (!s)
        return 0;
    return s->sol.probe_traces.size();
}

MHS_API mhs_status_t mhs_solution_probe_view(const mhs_solution_t* s, size_t index, mhs_probe_view_t* out)
{
    CHECK_NULL(s);
    CHECK_NULL(out);
    if (index >= s->sol.probe_traces.size()) {
        SET_ERR("probe index out of range");
        return MHS_ERR_INVALID_ARG;
    }
    const auto& tr = s->sol.probe_traces[index];
    out->name = tr.name.c_str();
    out->times = tr.times.empty() ? nullptr : tr.times.data();
    out->values = tr.values.empty() ? nullptr : tr.values.data();
    out->record_count = tr.times.size();
    mhs_detail_clear_last_error();
    return MHS_OK;
}
