/* Implementation of the MetaHotspot C API. */
#include "api/metahotspot.h"

#include "compiler/model_compiler.hpp"
#include "io/model_io.hpp"
#include "model/model_builder.hpp"
#include "model/model_definition.hpp"
#include "solver/assembler.hpp"
#include "solver/postprocessor.hpp"
#include "solver/scheduler.hpp"

#include <algorithm>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

/* ------------------------------------------------------------------ */
/*  Internal opaque handle definitions (hidden from the header)        */
/* ------------------------------------------------------------------ */

/* Bridge types. */
using MhsModelAxis = mhs::model::Axis;

struct PendingBoundary {
    enum Type : uint8_t { Unset, Dirichlet, Neumann, Convection };
    Type type = Unset;
    mhs::model::DirichletBoundary dirichlet;
    mhs::model::NeumannBoundary neumann;
    mhs::model::ConvectionBoundary convection;
    std::vector<mhs::model::FaceRegion> regions;
};

struct mhs_model_t {
    mhs::model::ModelBuilder builder;
    std::vector<PendingBoundary> pending_boundaries;
    mhs::model::ThermalBoundary default_bc = mhs::model::NeumannBoundary {};
    std::vector<mhs::model::FluidBoundarySpec> pending_fluid;
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
    int32_t node_count = 0;
    int32_t cell_count = 0;
};

struct mhs_solution_t {
    mhs::core::Solution solution;
    std::vector<double> node_temperatures;
    int32_t node_count = 0;
    int32_t cell_count = 0;
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

/* ------------------------------------------------------------------ */
/*  Enum conversions                                                   */
/* ------------------------------------------------------------------ */
static MhsModelAxis _to_axis(mhs_axis_t a)
{
    switch (a) {
    case MHS_AXIS_X:
        return MhsModelAxis::X;
    case MHS_AXIS_Y:
        return MhsModelAxis::Y;
    case MHS_AXIS_Z:
        return MhsModelAxis::Z;
    default:
        return MhsModelAxis::Z;
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
        return mhs::model::StudyType::Steady;
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
        return mhs::model::LengthUnit::Meter;
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
    return mhs::sim::SolverType::Pardiso;
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
    return mhs::model::FluidBoundaryKind::None;
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
    try {
        auto def = mhs::io::read_xml(path);

        /* Reset model handle state. */
        m->builder = mhs::model::ModelBuilder {};
        m->pending_boundaries.clear();
        m->default_bc = mhs::model::NeumannBoundary {};
        m->pending_fluid.clear();

        /* Populate builder. */
        m->builder.set_settings(def.settings);
        m->builder.set_mesh(def.mesh);

        for (auto& v : def.variables)
            m->builder.add_variable(std::move(v));
        for (auto& fn : def.functions)
            m->builder.add_function(std::move(fn));
        for (auto& mat : def.materials)
            m->builder.add_material(std::move(mat));

        for (auto& layer : def.layers) {
            mhs::model::LayerParams lp {
                std::move(layer.thickness), std::move(layer.x_offset), std::move(layer.y_offset)};
            auto lid = m->builder.add_layer(std::move(lp));
            for (auto& block : layer.blocks) {
                mhs::model::BlockParams bp {std::move(block.material), std::move(block.volumetric_heat_source),
                    std::move(block.x_offset), std::move(block.y_offset), std::move(block.thickness)};
                auto bid = m->builder.add_block(lid, std::move(bp));
                for (auto& rect : block.geometry)
                    m->builder.add_rect(bid, std::move(rect));
            }
        }

        /* Transfer boundaries into pending slots. */
        for (auto& bp : def.boundaries) {
            auto& pb = m->pending_boundaries.emplace_back();
            pb.regions = std::move(bp.regions);
            if (auto* d = std::get_if<mhs::model::DirichletBoundary>(&bp.condition)) {
                pb.type = PendingBoundary::Dirichlet;
                pb.dirichlet = std::move(*d);
            }
            else if (auto* n = std::get_if<mhs::model::NeumannBoundary>(&bp.condition)) {
                pb.type = PendingBoundary::Neumann;
                pb.neumann = std::move(*n);
            }
            else if (auto* cv = std::get_if<mhs::model::ConvectionBoundary>(&bp.condition)) {
                pb.type = PendingBoundary::Convection;
                pb.convection = std::move(*cv);
            }
        }

        m->default_bc = std::move(def.default_boundary);

        for (auto& ob : def.observation_points)
            m->builder.add_observation_point(std::move(ob));
        for (auto& fb : def.fluid_boundaries)
            m->pending_fluid.push_back(std::move(fb));

        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("read_xml: " << e.what());
        return MHS_ERR_IO;
    }
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  settings, mesh, variables                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_set_settings(mhs_model_t* m, mhs_study_t study, mhs_length_unit_t length_unit,
    double initial_temperature_K, double duration, double output_interval)
{
    CHECK_NULL(m);
    try {
        mhs::model::ModelSettings s;
        s.study_type = _to_model_study(study);
        s.length_unit = _to_unit(length_unit);
        s.initial_temperature = initial_temperature_K;
        s.transient_duration = duration;
        s.transient_output_interval = output_interval;
        m->builder.set_settings(std::move(s));
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("set_settings: " << e.what());
        return MHS_ERR_RUNTIME;
    }
}

MHS_API mhs_status_t mhs_model_set_mesh(
    mhs_model_t* m, int32_t nx, const double* x, int32_t ny, const double* y, int32_t nz, const double* z)
{
    CHECK_NULL(m);
    try {
        auto spec = m->builder.peek().mesh;
        if (nx >= 2) {
            CHECK_NULL(x);
            spec.x_vertices.assign(x, x + nx);
        }
        if (ny >= 2) {
            CHECK_NULL(y);
            spec.y_vertices.assign(y, y + ny);
        }
        if (nz >= 2) {
            CHECK_NULL(z);
            spec.z_vertices.assign(z, z + nz);
        }
        m->builder.set_mesh(std::move(spec));
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("set_mesh: " << e.what());
        return MHS_ERR_INVALID_ARG;
    }
}

MHS_API mhs_status_t mhs_model_add_variable(mhs_model_t* m, const char* name, const char* expression)
{
    CHECK_NULL(m);
    CHECK_NULL(name);
    CHECK_NULL(expression);
    try {
        m->builder.add_variable({name, expression});
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("add_variable(" << name << "): " << e.what());
        return MHS_ERR_INVALID_ARG;
    }
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  materials, layers, blocks, rects           */
/* ------------------------------------------------------------------ */

MHS_API mhs_material_id_t mhs_model_add_material(mhs_model_t* m, const char* name, const char* kx, const char* ky,
    const char* kz, const char* rho, const char* c, const char* dynamic_viscosity)
{
    if (!m) {
        SET_ERR("NULL pointer: m");
        return MHS_MATERIAL_ID_INVALID;
    }
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_MATERIAL_ID_INVALID;
    }
    try {
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

        m->builder.add_material({name, std::move(spec)});
        const auto id = static_cast<mhs_material_id_t>(m->builder.peek().materials.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_material(" << (name ? name : "?") << "): " << e.what());
        return MHS_MATERIAL_ID_INVALID;
    }
}

MHS_API mhs_layer_id_t mhs_model_add_layer(
    mhs_model_t* m, const char* thickness, const char* x_offset, const char* y_offset)
{
    CHECK_NULL(m);
    if (!thickness) {
        SET_ERR("NULL pointer: thickness");
        return MHS_LAYER_ID_INVALID;
    }
    if (!x_offset) {
        SET_ERR("NULL pointer: x_offset");
        return MHS_LAYER_ID_INVALID;
    }
    if (!y_offset) {
        SET_ERR("NULL pointer: y_offset");
        return MHS_LAYER_ID_INVALID;
    }
    try {
        auto id = m->builder.add_layer({thickness, x_offset, y_offset});
        tls_err.clear();
        return static_cast<mhs_layer_id_t>(id);
    }
    catch (const std::exception& e) {
        SET_ERR("add_layer: " << e.what());
        return MHS_LAYER_ID_INVALID;
    }
}

MHS_API mhs_block_id_t mhs_model_add_block(mhs_model_t* m, mhs_layer_id_t layer, const char* material_name,
    const char* heat_source, const char* x_offset, const char* y_offset, const char* thickness)
{
    CHECK_NULL(m);
    if (layer < 0) {
        SET_ERR("invalid layer ID: " << layer);
        return MHS_BLOCK_ID_INVALID;
    }
    if (!material_name) {
        SET_ERR("NULL pointer: material_name");
        return MHS_BLOCK_ID_INVALID;
    }
    try {
        mhs::model::BlockParams bp;
        bp.material = material_name;
        bp.volumetric_heat_source = heat_source ? heat_source : "0.0";
        bp.x_offset = x_offset ? x_offset : "0.0";
        bp.y_offset = y_offset ? y_offset : "0.0";
        if (thickness)
            bp.thickness = std::string(thickness);

        auto id = m->builder.add_block(static_cast<mhs::model::LayerId>(layer), std::move(bp));
        tls_err.clear();
        return static_cast<mhs_block_id_t>(id);
    }
    catch (const std::exception& e) {
        SET_ERR("add_block: " << e.what());
        return MHS_BLOCK_ID_INVALID;
    }
}

MHS_API mhs_status_t mhs_model_add_rect(mhs_model_t* m, mhs_block_id_t block, mhs_geometry_op_t op, const char* x,
    const char* y, const char* width, const char* height)
{
    CHECK_NULL(m);
    if (block < 0) {
        SET_ERR("invalid block ID: " << block);
        return MHS_ERR_INVALID_ARG;
    }
    if (!x) {
        SET_ERR("NULL pointer: x");
        return MHS_ERR_NULL_PTR;
    }
    if (!y) {
        SET_ERR("NULL pointer: y");
        return MHS_ERR_NULL_PTR;
    }
    if (!width) {
        SET_ERR("NULL pointer: width");
        return MHS_ERR_NULL_PTR;
    }
    if (!height) {
        SET_ERR("NULL pointer: height");
        return MHS_ERR_NULL_PTR;
    }
    try {
        mhs::model::RectOperation rect_op;
        rect_op.operation
            = (op == MHS_GEOM_SUB) ? mhs::model::GeometryOperation::Subtract : mhs::model::GeometryOperation::Add;
        rect_op.rect = {x, y, width, height};
        m->builder.add_rect(static_cast<mhs::model::BlockId>(block), std::move(rect_op));
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("add_rect: " << e.what());
        return MHS_ERR_INVALID_ARG;
    }
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  boundary conditions (two-step build)       */
/* ------------------------------------------------------------------ */

static mhs_status_t _check_boundary_id(const mhs_model_t* m, mhs_boundary_id_t id)
{
    if (id >= 0 && static_cast<size_t>(id) < m->pending_boundaries.size())
        return MHS_OK;
    SET_ERR("invalid boundary id: " << id);
    return MHS_ERR_INVALID_ARG;
}

MHS_API mhs_boundary_id_t mhs_model_add_boundary(mhs_model_t* m)
{
    CHECK_NULL(m);
    try {
        const auto id = static_cast<mhs_boundary_id_t>(m->pending_boundaries.size());
        m->pending_boundaries.emplace_back();
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_boundary: " << e.what());
        return MHS_BOUNDARY_ID_INVALID;
    }
}

MHS_API mhs_status_t mhs_boundary_set_dirichlet(mhs_model_t* m, mhs_boundary_id_t id, const char* temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(temperature);
    auto st = _check_boundary_id(m, id);
    if (st != MHS_OK)
        return st;
    auto& pb = m->pending_boundaries[static_cast<size_t>(id)];
    pb.type = PendingBoundary::Dirichlet;
    pb.dirichlet = {temperature};
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_boundary_set_neumann(mhs_model_t* m, mhs_boundary_id_t id, const char* heat_flux)
{
    CHECK_NULL(m);
    CHECK_NULL(heat_flux);
    auto st = _check_boundary_id(m, id);
    if (st != MHS_OK)
        return st;
    auto& pb = m->pending_boundaries[static_cast<size_t>(id)];
    pb.type = PendingBoundary::Neumann;
    pb.neumann = {heat_flux};
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_boundary_set_convection(
    mhs_model_t* m, mhs_boundary_id_t id, const char* coefficient, const char* ambient_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(coefficient);
    CHECK_NULL(ambient_temperature);
    auto st = _check_boundary_id(m, id);
    if (st != MHS_OK)
        return st;
    auto& pb = m->pending_boundaries[static_cast<size_t>(id)];
    pb.type = PendingBoundary::Convection;
    pb.convection = {coefficient, ambient_temperature};
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_boundary_add_face_region(
    mhs_model_t* m, mhs_boundary_id_t id, mhs_axis_t axis, double coordinate, mhs_rect2d_t region)
{
    CHECK_NULL(m);
    auto st = _check_boundary_id(m, id);
    if (st != MHS_OK)
        return st;
    try {
        m->pending_boundaries[static_cast<size_t>(id)].regions.push_back(_make_face_region(axis, coordinate, region));
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("add_face_region: " << e.what());
        return MHS_ERR_INVALID_ARG;
    }
}

/* Default boundaries. */
MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(temperature);
    m->default_bc = mhs::model::DirichletBoundary {temperature};
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux)
{
    CHECK_NULL(m);
    CHECK_NULL(heat_flux);
    m->default_bc = mhs::model::NeumannBoundary {heat_flux};
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_model_set_default_convection(
    mhs_model_t* m, const char* coefficient, const char* ambient_temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(coefficient);
    CHECK_NULL(ambient_temperature);
    m->default_bc = mhs::model::ConvectionBoundary {coefficient, ambient_temperature};
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  function library                            */
/* ------------------------------------------------------------------ */

MHS_API mhs_function_id_t mhs_model_add_function_expr(mhs_model_t* m, const char* name, const char* expression)
{
    CHECK_NULL(m);
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_FUNCTION_ID_INVALID;
    }
    if (!expression) {
        SET_ERR("NULL pointer: expression");
        return MHS_FUNCTION_ID_INVALID;
    }
    try {
        m->builder.add_function({name, mhs::model::ExpressionFunctionSpec {expression}});
        const auto id = static_cast<mhs_function_id_t>(m->builder.peek().functions.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_function_expr(" << name << "): " << e.what());
        return MHS_FUNCTION_ID_INVALID;
    }
}

MHS_API mhs_function_id_t mhs_model_add_function_gauss(
    mhs_model_t* m, const char* name, double amplitude, double tau, double center)
{
    CHECK_NULL(m);
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_FUNCTION_ID_INVALID;
    }
    try {
        m->builder.add_function({name, mhs::model::GaussFunctionSpec {amplitude, tau, center}});
        const auto id = static_cast<mhs_function_id_t>(m->builder.peek().functions.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_function_gauss(" << name << "): " << e.what());
        return MHS_FUNCTION_ID_INVALID;
    }
}

MHS_API mhs_function_id_t mhs_model_add_function_sine(
    mhs_model_t* m, const char* name, double amplitude, double angular_frequency, double phase)
{
    CHECK_NULL(m);
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_FUNCTION_ID_INVALID;
    }
    try {
        m->builder.add_function({name, mhs::model::SineFunctionSpec {amplitude, angular_frequency, phase}});
        const auto id = static_cast<mhs_function_id_t>(m->builder.peek().functions.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_function_sine(" << name << "): " << e.what());
        return MHS_FUNCTION_ID_INVALID;
    }
}

MHS_API mhs_function_id_t mhs_model_add_function_double_exponential(
    mhs_model_t* m, const char* name, double amplitude, double alpha, double beta)
{
    CHECK_NULL(m);
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_FUNCTION_ID_INVALID;
    }
    try {
        m->builder.add_function({name, mhs::model::DoubleExponentialFunctionSpec {amplitude, alpha, beta}});
        const auto id = static_cast<mhs_function_id_t>(m->builder.peek().functions.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_function_double_exp(" << name << "): " << e.what());
        return MHS_FUNCTION_ID_INVALID;
    }
}

MHS_API mhs_function_id_t mhs_model_add_function_piecewise(
    mhs_model_t* m, const char* name, const mhs_point2d_t* points, int32_t count)
{
    CHECK_NULL(m);
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_FUNCTION_ID_INVALID;
    }
    if (!points) {
        SET_ERR("NULL pointer: points");
        return MHS_FUNCTION_ID_INVALID;
    }
    if (count < 2) {
        SET_ERR("piecewise requires count >= 2");
        return MHS_FUNCTION_ID_INVALID;
    }
    try {
        mhs::model::PiecewiseFunctionSpec spec;
        for (int32_t i = 0; i < count; ++i)
            spec.points.push_back({points[i].x, points[i].y});
        m->builder.add_function({name, std::move(spec)});
        const auto id = static_cast<mhs_function_id_t>(m->builder.peek().functions.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_function_piecewise(" << name << "): " << e.what());
        return MHS_FUNCTION_ID_INVALID;
    }
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  probes and fluid boundaries                */
/* ------------------------------------------------------------------ */

MHS_API mhs_probe_id_t mhs_model_add_probe(mhs_model_t* m, const char* name, double x, double y, double z)
{
    CHECK_NULL(m);
    if (!name) {
        SET_ERR("NULL pointer: name");
        return MHS_PROBE_ID_INVALID;
    }
    try {
        m->builder.add_observation_point({name, std::to_string(x), std::to_string(y), std::to_string(z)});
        const auto id = static_cast<mhs_probe_id_t>(m->builder.peek().observation_points.size()) - 1;
        tls_err.clear();
        return id;
    }
    catch (const std::exception& e) {
        SET_ERR("add_probe(" << (name ? name : "?") << "): " << e.what());
        return MHS_PROBE_ID_INVALID;
    }
}

MHS_API mhs_status_t mhs_model_add_fluid_boundary(mhs_model_t* m, mhs_axis_t axis, double coordinate,
    mhs_rect2d_t region, mhs_fluid_bc_t kind, double value, double inlet_temperature)
{
    CHECK_NULL(m);
    try {
        mhs::model::FluidBoundarySpec fb;
        fb.regions.push_back(_make_face_region(axis, coordinate, region));
        fb.kind = _to_fluid_kind(kind);
        fb.value = value;
        fb.inlet_temperature = inlet_temperature;
        m->pending_fluid.push_back(std::move(fb));
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("add_fluid_boundary: " << e.what());
        return MHS_ERR_INVALID_ARG;
    }
}

/* ------------------------------------------------------------------ */
/*  Model introspection                                                */
/* ------------------------------------------------------------------ */

MHS_API const char* mhs_model_material_name(const mhs_model_t* m, int32_t index)
{
    if (!m)
        return nullptr;
    const auto& materials = m->builder.peek().materials;
    if (index < 0 || static_cast<size_t>(index) >= materials.size())
        return nullptr;
    return materials[static_cast<size_t>(index)].name.c_str();
}

MHS_API int32_t mhs_model_material_count(const mhs_model_t* m)
{
    if (!m)
        return 0;
    return static_cast<int32_t>(m->builder.peek().materials.size());
}

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(mhs_model_t* m, mhs_compiled_t** out)
{
    CHECK_NULL(m);
    CHECK_NULL(out);
    try {
        mhs::model::ModelDefinition def = m->builder.peek();

        /* Validate and append pending thermal boundaries (copy). */
        def.boundaries.reserve(def.boundaries.size() + m->pending_boundaries.size());
        for (size_t i = 0; i < m->pending_boundaries.size(); ++i) {
            const auto& pb = m->pending_boundaries[i];
            if (pb.type == PendingBoundary::Unset) {
                SET_ERR("boundary slot " << i
                                         << " has no condition set (call set_dirichlet/"
                                            "set_neumann/set_convection)");
                *out = nullptr;
                return MHS_ERR_UNSET;
            }
            mhs::model::ThermalBoundary cond;
            switch (pb.type) {
            case PendingBoundary::Dirichlet:
                cond = pb.dirichlet;
                break;
            case PendingBoundary::Neumann:
                cond = pb.neumann;
                break;
            case PendingBoundary::Convection:
                cond = pb.convection;
                break;
            default:
                break;
            }
            def.boundaries.push_back({pb.regions, cond});
        }

        /* Append pending fluid boundaries (copy). */
        def.fluid_boundaries.reserve(def.fluid_boundaries.size() + m->pending_fluid.size());
        for (const auto& fb : m->pending_fluid)
            def.fluid_boundaries.push_back(fb);

        def.default_boundary = m->default_bc;

        /* Compile from the composed definition. */
        auto core_model = mhs::sim::build_model(def);

        /* Wrap in opaque handle. */
        auto* c = new (std::nothrow) mhs_compiled_t {std::move(core_model)};
        if (!c) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }
        c->node_count = static_cast<int32_t>((c->model.mesh.nx + 1) * (c->model.mesh.ny + 1) * (c->model.mesh.nz + 1));
        c->cell_count = static_cast<int32_t>(c->model.cells.cell_to_grid.size());

        *out = c;
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        *out = nullptr;
        SET_ERR("compile: " << e.what());
        return MHS_ERR_COMPILE;
    }
}

MHS_API mhs_status_t mhs_compiled_destroy(mhs_compiled_t* c)
{
    delete c;
    tls_err.clear();
    return MHS_OK;
}

MHS_API int32_t mhs_compiled_cell_count(const mhs_compiled_t* c)
{
    if (!c)
        return 0;
    return c->cell_count;
}

MHS_API int32_t mhs_compiled_state_count(const mhs_compiled_t* c)
{
    if (!c)
        return 0;
    return static_cast<int32_t>(c->model.dofs.total_count);
}

MHS_API int32_t mhs_compiled_node_count(const mhs_compiled_t* c)
{
    if (!c)
        return 0;
    return c->node_count;
}

MHS_API double mhs_compiled_initial_temperature(const mhs_compiled_t* c)
{
    if (!c)
        return 300.0;
    return c->model.initial_temperature;
}

MHS_API const uint32_t* mhs_compiled_layer_ids(const mhs_compiled_t* c)
{
    if (!c)
        return nullptr;
    return c->model.cells.layer_id.data();
}

MHS_API const uint32_t* mhs_compiled_block_ids(const mhs_compiled_t* c)
{
    if (!c)
        return nullptr;
    return c->model.cells.block_id.data();
}

MHS_API uint32_t mhs_compiled_layer_count(const mhs_compiled_t* c)
{
    if (!c)
        return 0;
    /* Count unique layer IDs by scanning for the max value. */
    if (c->model.cells.layer_id.empty())
        return 0;
    auto max_l = *std::max_element(c->model.cells.layer_id.begin(), c->model.cells.layer_id.end());
    return max_l + 1;
}

MHS_API uint32_t mhs_compiled_block_count(const mhs_compiled_t* c, uint32_t layer)
{
    if (!c)
        return 0;
    /* Count unique (layer, block) combos for that layer. */
    std::set<uint32_t> seen;
    for (size_t i = 0; i < c->model.cells.block_id.size(); i++) {
        if (c->model.cells.layer_id[i] == static_cast<mhs::core::TableIndex>(layer))
            seen.insert(c->model.cells.block_id[i]);
    }
    return static_cast<uint32_t>(seen.size());
}

MHS_API mhs_study_t mhs_compiled_study_type(const mhs_compiled_t* c)
{
    if (!c)
        return MHS_STUDY_STEADY;
    return _from_core_study(c->model.study_type);
}

/* ------------------------------------------------------------------ */
/*  Assembly (K, C, f in CSC format)                                   */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_assemble(
    const mhs_compiled_t* c, const double* state, double time, mhs_assembly_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    try {
        mhs::sim::Assembler assembler(c->model);

        const auto n = static_cast<std::size_t>(c->model.dofs.total_count);
        std::vector<double> current_state(n);
        if (state) {
            std::copy_n(state, n, current_state.begin());
        }
        else {
            std::fill(current_state.begin(), current_state.end(), c->model.initial_temperature);
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
        const int32_t dim = h->stiffness.n;
        h->rhs.assign(result.f.data(), result.f.data() + dim);

        *out = h.release();
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        *out = nullptr;
        SET_ERR("assemble: " << e.what());
        return MHS_ERR_ASSEMBLE;
    }
}

MHS_API mhs_status_t mhs_assembly_destroy(mhs_assembly_t* a)
{
    delete a;
    tls_err.clear();
    return MHS_OK;
}

MHS_API int32_t mhs_assembly_n(const mhs_assembly_t* a)
{
    if (!a)
        return 0;
    return a->stiffness.n;
}

MHS_API int32_t mhs_assembly_stiffness_nnz(const mhs_assembly_t* a)
{
    if (!a)
        return 0;
    return a->stiffness.nnz;
}

MHS_API const int32_t* mhs_assembly_stiffness_outer_indices(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->stiffness.outer_indices.data();
}

MHS_API const int32_t* mhs_assembly_stiffness_inner_indices(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->stiffness.inner_indices.data();
}

MHS_API const double* mhs_assembly_stiffness_values(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->stiffness.values.data();
}

MHS_API const double* mhs_assembly_rhs(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->rhs.data();
}

MHS_API int32_t mhs_assembly_capacity_nnz(const mhs_assembly_t* a)
{
    if (!a)
        return 0;
    return a->capacity.nnz;
}

MHS_API const int32_t* mhs_assembly_capacity_outer_indices(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->capacity.outer_indices.data();
}

MHS_API const int32_t* mhs_assembly_capacity_inner_indices(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->capacity.inner_indices.data();
}

MHS_API const double* mhs_assembly_capacity_values(const mhs_assembly_t* a)
{
    if (!a)
        return nullptr;
    return a->capacity.values.data();
}

/* ------------------------------------------------------------------ */
/*  Solve                                                              */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_compiled_solve(const mhs_compiled_t* c, const mhs_solver_opts_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(c);
    CHECK_NULL(out);
    try {
        /* Build SolveOptions. */
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

        /* Compute node temperatures from cell-centroid solution. */
        auto node_T = mhs::post::interpolate_cell_to_node(c->model, sol.cell_temperature, sol.time);

        auto* s = new (std::nothrow) mhs_solution_t {std::move(sol), std::move(node_T), c->node_count, c->cell_count};

        if (!s) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }

        *out = s;
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        *out = nullptr;
        SET_ERR("solve: " << e.what());
        return MHS_ERR_SOLVE;
    }
}

MHS_API mhs_status_t mhs_solve(mhs_model_t* m, const mhs_solver_opts_t* opts, mhs_solution_t** out)
{
    CHECK_NULL(m);
    CHECK_NULL(out);
    try {
        mhs_compiled_t* compiled = nullptr;
        auto st = mhs_model_compile(m, &compiled);
        if (st != MHS_OK) {
            *out = nullptr;
            return st;
        }

        /* Transfer ownership to unique_ptr so compile-and-solve doesn't
         * leak the compiled model on success or failure. */
        auto compiled_guard = std::unique_ptr<mhs_compiled_t>(compiled);

        mhs_solution_t* sol = nullptr;
        st = mhs_compiled_solve(compiled_guard.get(), opts, &sol);
        if (st != MHS_OK) {
            *out = nullptr;
            return st;
        }

        // compiled_guard destroyed here — the solution holds its own data copies.

        *out = sol;
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        *out = nullptr;
        SET_ERR("solve (single-step): " << e.what());
        return MHS_ERR_SOLVE;
    }
}

MHS_API mhs_status_t mhs_solution_destroy(mhs_solution_t* s)
{
    delete s;
    tls_err.clear();
    return MHS_OK;
}

/* ------------------------------------------------------------------ */
/*  Solution accessors                                                 */
/* ------------------------------------------------------------------ */

MHS_API int32_t mhs_solution_cell_count(const mhs_solution_t* s)
{
    if (!s)
        return 0;
    return s->cell_count;
}

MHS_API int32_t mhs_solution_state_count(const mhs_solution_t* s)
{
    if (!s)
        return 0;
    return static_cast<int32_t>(s->solution.state.size());
}

MHS_API int32_t mhs_solution_node_count(const mhs_solution_t* s)
{
    if (!s)
        return 0;
    return s->node_count;
}

MHS_API double mhs_solution_time(const mhs_solution_t* s)
{
    if (!s)
        return 0.0;
    return s->solution.time;
}

MHS_API const double* mhs_solution_cell_temperatures(const mhs_solution_t* s)
{
    if (!s)
        return nullptr;
    return s->solution.cell_temperature.data();
}

MHS_API const double* mhs_solution_states(const mhs_solution_t* s)
{
    if (!s)
        return nullptr;
    return s->solution.state.data();
}

MHS_API const double* mhs_solution_node_temperatures(const mhs_solution_t* s)
{
    if (!s)
        return nullptr;
    return s->node_temperatures.data();
}

/* ------------------------------------------------------------------ */
/*  Probe trace accessors                                              */
/* ------------------------------------------------------------------ */

MHS_API int32_t mhs_solution_probe_count(const mhs_solution_t* s)
{
    if (!s)
        return 0;
    return static_cast<int32_t>(s->solution.probe_traces.size());
}

MHS_API const char* mhs_solution_probe_name(const mhs_solution_t* s, int32_t index)
{
    if (!s)
        return nullptr;
    if (index < 0 || static_cast<size_t>(index) >= s->solution.probe_traces.size())
        return nullptr;
    return s->solution.probe_traces[static_cast<size_t>(index)].name.c_str();
}

MHS_API int32_t mhs_solution_probe_record_count(const mhs_solution_t* s, int32_t probe_index)
{
    if (!s)
        return 0;
    if (probe_index < 0 || static_cast<size_t>(probe_index) >= s->solution.probe_traces.size())
        return 0;
    return static_cast<int32_t>(s->solution.probe_traces[static_cast<size_t>(probe_index)].values.size());
}

MHS_API const double* mhs_solution_probe_times(const mhs_solution_t* s, int32_t probe_index)
{
    if (!s)
        return nullptr;
    if (probe_index < 0 || static_cast<size_t>(probe_index) >= s->solution.probe_traces.size())
        return nullptr;
    const auto& tr = s->solution.probe_traces[static_cast<size_t>(probe_index)];
    return tr.times.empty() ? nullptr : tr.times.data();
}

MHS_API const double* mhs_solution_probe_values(const mhs_solution_t* s, int32_t probe_index)
{
    if (!s)
        return nullptr;
    if (probe_index < 0 || static_cast<size_t>(probe_index) >= s->solution.probe_traces.size())
        return nullptr;
    return s->solution.probe_traces[static_cast<size_t>(probe_index)].values.data();
}
