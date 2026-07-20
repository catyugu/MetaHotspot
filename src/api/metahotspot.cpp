/* Implementation of the MetaHotspot C API. */
#include "api/metahotspot.h"

#include "compiler/model_compiler.hpp"
#include "io/model_io.hpp"
#include "model/model_builder.hpp"
#include "model/model_definition.hpp"
#include "solver/postprocessor.hpp"
#include "solver/scheduler.hpp"

#include <memory>
#include <optional>
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
    bool has_default_bc = false;
    mhs::model::ThermalBoundary default_bc = mhs::model::NeumannBoundary {};
    std::vector<mhs::model::FluidBoundarySpec> pending_fluid;

    /* Tracking counters for ID-based returns.
     * ModelBuilder's internals are private, so we shadow what we need. */
    int32_t material_count = 0;
    int32_t function_count = 0;
    int32_t probe_count = 0;
    std::vector<std::string> material_names;
};

struct mhs_compiled_t {
    mhs::core::Model model;
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
    }
    return mhs::model::StudyType::Steady;
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
    }
    return mhs::model::LengthUnit::Meter;
}

static mhs_study_t _from_core_study(mhs::core::StudyType s)
{
    switch (s) {
    case mhs::core::StudyType::Steady:
        return MHS_STUDY_STEADY;
    case mhs::core::StudyType::Transient:
        return MHS_STUDY_TRANSIENT;
    }
    return MHS_STUDY_STEADY;
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
/*  Internal helpers — flush pending boundaries into the builder       */
/* ------------------------------------------------------------------ */
static mhs_status_t _flush_boundaries(mhs_model_t* m)
{
    for (size_t i = 0; i < m->pending_boundaries.size(); ++i) {
        auto& pb = m->pending_boundaries[i];
        if (pb.type == PendingBoundary::Unset) {
            SET_ERR("boundary slot " << i
                                     << " has no condition set (call set_dirichlet/"
                                        "set_neumann/set_convection)");
            return MHS_ERR_UNSET;
        }
        mhs::model::ThermalBoundary cond;
        switch (pb.type) {
        case PendingBoundary::Dirichlet:
            cond = std::move(pb.dirichlet);
            break;
        case PendingBoundary::Neumann:
            cond = std::move(pb.neumann);
            break;
        case PendingBoundary::Convection:
            cond = std::move(pb.convection);
            break;
        default:
            break;
        }
        m->builder.add_boundary({std::move(pb.regions), std::move(cond)});
    }
    m->pending_boundaries.clear();

    if (m->has_default_bc)
        m->builder.set_default_boundary(m->default_bc);

    /* Flush pending fluid boundaries. */
    for (auto& fb : m->pending_fluid)
        m->builder.add_fluid_boundary(std::move(fb));
    m->pending_fluid.clear();

    return MHS_OK;
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
    case MHS_ERR_LAYER:
        return "invalid layer index or error";
    case MHS_ERR_BLOCK:
        return "invalid block index or error";
    case MHS_ERR_BOUNDARY:
        return "invalid boundary index or error";
    case MHS_ERR_MATERIAL:
        return "material error";
    case MHS_ERR_FUNCTION:
        return "function error";
    case MHS_ERR_COMPILE:
        return "compilation error";
    case MHS_ERR_SOLVE:
        return "solver did not converge";
    case MHS_ERR_IO:
        return "I/O error";
    case MHS_ERR_OOM:
        return "out of memory";
    case MHS_ERR_UNSET:
        return "unset required field";
    case MHS_ERR_FLUID:
        return "fluid boundary error";
    case MHS_ERR_MESH:
        return "mesh error";
    case MHS_ERR_VARIABLE:
        return "variable error";
    case MHS_ERR_PROBE:
        return "probe error";
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
        m->has_default_bc = false;
        m->default_bc = mhs::model::NeumannBoundary {};
        m->pending_fluid.clear();
        m->material_count = 0;
        m->function_count = 0;
        m->probe_count = 0;
        m->material_names.clear();

        /* Populate builder. */
        m->builder.set_settings(def.settings);
        m->builder.set_mesh(def.mesh);

        for (auto& v : def.variables)
            m->builder.add_variable(std::move(v));
        for (auto& fn : def.functions) {
            m->builder.add_function(std::move(fn));
            m->function_count++;
        }
        for (auto& mat : def.materials) {
            m->material_names.push_back(mat.name);
            m->builder.add_material(std::move(mat));
            m->material_count++;
        }

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
        m->has_default_bc = true;

        for (auto& ob : def.observation_points) {
            m->builder.add_observation_point(std::move(ob));
            m->probe_count++;
        }
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
        return MHS_ERR_COMPILE;
    }
}

MHS_API mhs_status_t mhs_model_set_mesh_x(mhs_model_t* m, int32_t count, const double* vertices)
{
    CHECK_NULL(m);
    CHECK_NULL(vertices);
    if (count < 2) {
        SET_ERR("set_mesh_x: count must be >= 2, got " << count);
        return MHS_ERR_MESH;
    }
    try {
        m->builder.set_mesh({std::vector<double>(vertices, vertices + count), {}, {}});
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("set_mesh_x: " << e.what());
        return MHS_ERR_MESH;
    }
}

MHS_API mhs_status_t mhs_model_set_mesh_y(mhs_model_t* m, int32_t count, const double* vertices)
{
    CHECK_NULL(m);
    CHECK_NULL(vertices);
    if (count < 2) {
        SET_ERR("set_mesh_y: count must be >= 2, got " << count);
        return MHS_ERR_MESH;
    }
    try {
        m->builder.set_mesh({{}, std::vector<double>(vertices, vertices + count), {}});
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("set_mesh_y: " << e.what());
        return MHS_ERR_MESH;
    }
}

MHS_API mhs_status_t mhs_model_set_mesh_z(mhs_model_t* m, int32_t count, const double* vertices)
{
    CHECK_NULL(m);
    CHECK_NULL(vertices);
    if (count < 2) {
        SET_ERR("set_mesh_z: count must be >= 2, got " << count);
        return MHS_ERR_MESH;
    }
    try {
        m->builder.set_mesh({{}, {}, std::vector<double>(vertices, vertices + count)});
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("set_mesh_z: " << e.what());
        return MHS_ERR_MESH;
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
        return MHS_ERR_VARIABLE;
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
        spec.conductivity_x = kx ? kx : "0.0";
        spec.conductivity_y = ky ? ky : "0.0";
        spec.conductivity_z = kz ? kz : "0.0";
        spec.density = rho ? rho : "0.0";
        spec.specific_heat = c ? c : "0.0";
        if (dynamic_viscosity)
            spec.dynamic_viscosity = std::string(dynamic_viscosity);

        m->builder.add_material({name, std::move(spec)});
        m->material_names.push_back(name ? name : "");
        const auto id = static_cast<mhs_material_id_t>(m->material_count);
        m->material_count++;
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
        return MHS_ERR_BLOCK;
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
        return MHS_ERR_BLOCK;
    }
}

/* ------------------------------------------------------------------ */
/*  Model construction  —  boundary conditions (two-step build)       */
/* ------------------------------------------------------------------ */

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
    if (id < 0 || static_cast<size_t>(id) >= m->pending_boundaries.size()) {
        SET_ERR("invalid boundary id: " << id);
        return MHS_ERR_BOUNDARY;
    }
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
    if (id < 0 || static_cast<size_t>(id) >= m->pending_boundaries.size()) {
        SET_ERR("invalid boundary id: " << id);
        return MHS_ERR_BOUNDARY;
    }
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
    if (id < 0 || static_cast<size_t>(id) >= m->pending_boundaries.size()) {
        SET_ERR("invalid boundary id: " << id);
        return MHS_ERR_BOUNDARY;
    }
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
    if (id < 0 || static_cast<size_t>(id) >= m->pending_boundaries.size()) {
        SET_ERR("invalid boundary id: " << id);
        return MHS_ERR_BOUNDARY;
    }
    try {
        m->pending_boundaries[static_cast<size_t>(id)].regions.push_back(_make_face_region(axis, coordinate, region));
        tls_err.clear();
        return MHS_OK;
    }
    catch (const std::exception& e) {
        SET_ERR("add_face_region: " << e.what());
        return MHS_ERR_BOUNDARY;
    }
}

/* Default boundaries. */
MHS_API mhs_status_t mhs_model_set_default_dirichlet(mhs_model_t* m, const char* temperature)
{
    CHECK_NULL(m);
    CHECK_NULL(temperature);
    m->default_bc = mhs::model::DirichletBoundary {temperature};
    m->has_default_bc = true;
    tls_err.clear();
    return MHS_OK;
}

MHS_API mhs_status_t mhs_model_set_default_neumann(mhs_model_t* m, const char* heat_flux)
{
    CHECK_NULL(m);
    CHECK_NULL(heat_flux);
    m->default_bc = mhs::model::NeumannBoundary {heat_flux};
    m->has_default_bc = true;
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
    m->has_default_bc = true;
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
        const auto id = static_cast<mhs_function_id_t>(m->function_count);
        m->function_count++;
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
        const auto id = static_cast<mhs_function_id_t>(m->function_count);
        m->function_count++;
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
        const auto id = static_cast<mhs_function_id_t>(m->function_count);
        m->function_count++;
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
        const auto id = static_cast<mhs_function_id_t>(m->function_count);
        m->function_count++;
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
        const auto id = static_cast<mhs_function_id_t>(m->function_count);
        m->function_count++;
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
        const auto id = static_cast<mhs_probe_id_t>(m->probe_count);
        m->probe_count++;
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
        return MHS_ERR_FLUID;
    }
}

/* ------------------------------------------------------------------ */
/*  Model introspection                                                */
/* ------------------------------------------------------------------ */

MHS_API const char* mhs_model_material_name(const mhs_model_t* m, int32_t index)
{
    if (!m)
        return nullptr;
    if (index < 0 || static_cast<size_t>(index) >= m->material_names.size())
        return nullptr;
    return m->material_names[static_cast<size_t>(index)].c_str();
}

MHS_API int32_t mhs_model_material_count(const mhs_model_t* m)
{
    if (!m)
        return 0;
    return m->material_count;
}

/* ------------------------------------------------------------------ */
/*  Compilation                                                        */
/* ------------------------------------------------------------------ */

MHS_API mhs_status_t mhs_model_compile(mhs_model_t* m, mhs_compiled_t** out)
{
    CHECK_NULL(m);
    CHECK_NULL(out);
    try {
        /* Flush pending boundaries into the builder. */
        auto st = _flush_boundaries(m);
        if (st != MHS_OK) {
            *out = nullptr;
            return st;
        }

        /* Compile from the builder's current state without consuming it.
         * build_model() takes a const ref, so peek() suffices. */
        auto core_model = mhs::sim::build_model(m->builder.peek());

        /* Wrap in opaque handle. */
        auto* c = new (std::nothrow) mhs_compiled_t {std::move(core_model)};
        if (!c) {
            *out = nullptr;
            SET_ERR("memory allocation failed");
            return MHS_ERR_OOM;
        }

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
    return static_cast<int32_t>(c->model.cells.cell_to_grid.size());
}

MHS_API int32_t mhs_compiled_node_count(const mhs_compiled_t* c)
{
    if (!c)
        return 0;
    return static_cast<int32_t>((c->model.mesh.nx + 1) * (c->model.mesh.ny + 1) * (c->model.mesh.nz + 1));
}

MHS_API double mhs_compiled_initial_temperature(const mhs_compiled_t* c)
{
    if (!c)
        return 300.0;
    return c->model.initial_temperature;
}

MHS_API mhs_study_t mhs_compiled_study_type(const mhs_compiled_t* c)
{
    if (!c)
        return MHS_STUDY_STEADY;
    return _from_core_study(c->model.study_type);
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
        auto node_T = mhs::post::interpolate_cell_to_node(c->model, sol.temperature, sol.time);

        const auto node_count
            = static_cast<int32_t>((c->model.mesh.nx + 1) * (c->model.mesh.ny + 1) * (c->model.mesh.nz + 1));

        auto* s = new (std::nothrow) mhs_solution_t {
            std::move(sol), std::move(node_T), node_count, static_cast<int32_t>(c->model.cells.cell_to_grid.size())};

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
    return s->solution.temperature.data();
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
