// Regression tests for fluid boundary-condition influence on the assembled RHS.
//
// Bug: After the refactor that introduced fluid BC dispatch
// (commit 646f364 "try supporing more kinds of fluid bc..."), the assembler
// gated the MassFlowRateType / VelocityType override behind
// `cell_is_fluid && netOutflux != 0.0`. For a Neumann inlet + Dirichlet
// outlet configuration, the pressure-driven netOutflux is zero everywhere
// (the pressure solver has no driving force at a Neumann inlet), so the
// override never ran — the BC value was ignored.
//
// These tests fail with the broken gate (ops.f is identical for two different
// MassFlowRate values) and pass after the gate is removed (ops.f differs).

#include "assembler/assembler.hpp"
#include "config.h"
#include "data/internal_model.hpp"
#include "data/io_model.hpp"
#include "io/io.hpp"
#include "preprocessor/fluid_preprocessor.hpp"
#include "preprocessor/preprocessor.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <string>

using namespace mhs::sim;
using namespace mhs::io;

namespace {

    // Write a transient XML text into a temp file and return the path.
    std::filesystem::path write_tmp(const std::string& content, const std::string& suffix)
    {
        auto path = std::filesystem::temp_directory_path()
                    / ("mhstest_fluidbc_" + std::to_string(std::rand()) + suffix);
        std::ofstream f(path);
        f << content;
        f.close();
        return path;
    }

    // Build the canonical steady_case1 IO data from disk.
    bool load_steady_case1(mhs::core::IOStructure& out)
    {
        std::string case_path
            = std::string(PROJECT_SOURCE_DIR) + "/cases/microfluid_cases/steady_case1.xml";
        if (!std::filesystem::exists(case_path))
            return false;
        out = mhs::io::read_xml(case_path);
        return true;
    }

    // Build a MassFlowRate overlay XML with the given inlet value.
    std::string make_mdot_overlay(double mdot)
    {
        return R"(<?xml version="1.0" encoding="UTF-8"?>
<FluidOverlay xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
    <FluidMaterial name="water">
        <DynamicViscosity>0.00089</DynamicViscosity>
    </FluidMaterial>
    <Boundary>
        <BoundaryCategory>Fluidic</BoundaryCategory>
        <Name>fluid_inlet_mdot</Name>
        <FaceKeys>
            <string>X|E|0|0.5|1.5|0.2|0.4</string>
            <string>X|E|0|2.5|3.5|0.2|0.4</string>
        </FaceKeys>
        <MassFlowRate>)"
            + std::to_string(mdot) + R"(</MassFlowRate>
        <InletTemperature>298.15</InletTemperature>
    </Boundary>
    <Boundary>
        <BoundaryCategory>Fluidic</BoundaryCategory>
        <Name>fluid_outlet_pres</Name>
        <FaceKeys>
            <string>X|E|8|0.5|1.5|0.2|0.4</string>
            <string>X|E|8|2.5|3.5|0.2|0.4</string>
        </FaceKeys>
        <Pressure>0</Pressure>
    </Boundary>
</FluidOverlay>)";
    }

    // Build a Velocity overlay XML with the given inlet value.
    std::string make_velocity_overlay(double v)
    {
        return R"(<?xml version="1.0" encoding="UTF-8"?>
<FluidOverlay xmlns="http://schemas.datacontract.org/2004/07/ThermalSim.Models">
    <FluidMaterial name="water">
        <DynamicViscosity>0.00089</DynamicViscosity>
    </FluidMaterial>
    <Boundary>
        <BoundaryCategory>Fluidic</BoundaryCategory>
        <Name>fluid_inlet_vel</Name>
        <FaceKeys>
            <string>X|E|0|0.5|1.5|0.2|0.4</string>
            <string>X|E|0|2.5|3.5|0.2|0.4</string>
        </FaceKeys>
        <Velocity>)"
            + std::to_string(v) + R"(</Velocity>
        <InletTemperature>298.15</InletTemperature>
    </Boundary>
    <Boundary>
        <BoundaryCategory>Fluidic</BoundaryCategory>
        <Name>fluid_outlet_pres</Name>
        <FaceKeys>
            <string>X|E|8|0.5|1.5|0.2|0.4</string>
            <string>X|E|8|2.5|3.5|0.2|0.4</string>
        </FaceKeys>
        <Pressure>0</Pressure>
    </Boundary>
</FluidOverlay>)";
    }

    // Run the full preprocessor + assembler pipeline on a case with the given
    // overlay XML content. Returns the assembled ops so callers can diff them.
    struct AssembledOps {
        Eigen::SparseMatrix<double> K;
        Eigen::VectorXd f;
        Eigen::VectorXd M_diag;
    };

    std::optional<AssembledOps> assemble_with_overlay(const std::string& overlay_xml)
    {
        mhs::core::IOStructure io_data;
        if (!load_steady_case1(io_data))
            return std::nullopt;

        auto overlay_path = write_tmp(overlay_xml, "_overlay.xml");
        auto overlay = mhs::io::read_fluid_overlay_xml(overlay_path.string());
        std::filesystem::remove(overlay_path);
        if (!overlay.has_value())
            return std::nullopt;

        Preprocessor preprocessor;
        auto model = preprocessor.load(io_data, overlay);
        if (!model)
            return std::nullopt;

        int N = static_cast<int>(model->cells.cell_bcs.size());
        std::vector<double> T(static_cast<std::size_t>(N), model->initial_temperature);
        Eigen::Map<const Eigen::VectorXd> T_map(T.data(), N);
        AssembleContext ctx {T_map, 0.0};

        Assembler assembler(*model);
        auto ops = assembler.assemble(ctx);
        return AssembledOps {std::move(ops.K), std::move(ops.f), std::move(ops.M_diag)};
    }

} // namespace

// Pre-fix failure mode: ops.f was identical for any MassFlowRate value
// because the override branch was gated on pressure-driven netOutflux == 0
// and never ran. Post-fix: ops.f scales linearly with the inlet value.
TEST(FluidBCTest, MassFlowRateAffectsAssembledRhs)
{
    auto ops_a = assemble_with_overlay(make_mdot_overlay(1e-4));
    auto ops_b = assemble_with_overlay(make_mdot_overlay(1e-2));
    ASSERT_TRUE(ops_a.has_value());
    ASSERT_TRUE(ops_b.has_value());

    // Find the maximum per-component absolute difference of f. The two
    // MassFlowRate values differ by 100x, so any cell whose RHS reflects
    // the inlet enthalpy source must contribute a non-zero delta.
    double max_delta = 0.0;
    int differing_cells = 0;
    for (Eigen::Index i = 0; i < ops_a->f.size(); ++i) {
        double d = std::fabs(ops_a->f[i] - ops_b->f[i]);
        if (d > 1e-12) {
            ++differing_cells;
            max_delta = std::max(max_delta, d);
        }
    }
    EXPECT_GT(differing_cells, 0) << "MassFlowRate=1e-4 vs 1e-2 must change ops.f at inlet cells";
    EXPECT_GT(max_delta, 0.0);
}

TEST(FluidBCTest, VelocityAffectsAssembledRhs)
{
    auto ops_a = assemble_with_overlay(make_velocity_overlay(1e-5));
    auto ops_b = assemble_with_overlay(make_velocity_overlay(1e-1));
    ASSERT_TRUE(ops_a.has_value());
    ASSERT_TRUE(ops_b.has_value());

    double max_delta = 0.0;
    int differing_cells = 0;
    for (Eigen::Index i = 0; i < ops_a->f.size(); ++i) {
        double d = std::fabs(ops_a->f[i] - ops_b->f[i]);
        if (d > 1e-12) {
            ++differing_cells;
            max_delta = std::max(max_delta, d);
        }
    }
    EXPECT_GT(differing_cells, 0) << "Velocity=1e-5 vs 1e-1 must change ops.f at inlet cells";
}

// Preprocessor-level invariant: MassFlowRate is interpreted as the flux
// per face-key (consistent with Pressure BC: 1000 Pa at one boundary with
// 2 face_keys means 1000 Pa at each face, not 500 Pa divided across them).
// Sum of (mass_flow_rate[param_idx] for each inlet cell) must therefore
// equal (user value) * (number of face keys matched by the boundary).
TEST(FluidBCTest, MassFlowRateAppliedPerFaceKey)
{
    mhs::core::IOStructure io_data;
    ASSERT_TRUE(load_steady_case1(io_data));

    const double mdot_value = 1e-3;
    // The canonical overlay has 2 face_keys for the inlet boundary. With the
    // "per face_key" semantics, each face_key gets mdot_value, so the total
    // across all inlet cells = 2 * mdot_value. We assert the sum equals that.
    auto overlay_path = write_tmp(make_mdot_overlay(mdot_value), "_overlay.xml");
    auto overlay = mhs::io::read_fluid_overlay_xml(overlay_path.string());
    std::filesystem::remove(overlay_path);
    ASSERT_TRUE(overlay.has_value());

    Preprocessor preprocessor;
    auto model = preprocessor.load(io_data, overlay);
    ASSERT_NE(model, nullptr);

    double total_mdot = 0.0;
    int inlet_cell_count = 0;
    for (size_t fi = 0; fi < model->fluid_bcs.size(); ++fi) {
        if (model->fluid_bcs[fi].kind == mhs::core::FluidBCType::MassFlowRateType) {
            total_mdot += model->fluid_bc_params.mass_flow_rate[model->fluid_bcs[fi].param_idx];
            ++inlet_cell_count;
        }
    }
    EXPECT_GT(inlet_cell_count, 0) << "Expected at least one MassFlowRate inlet cell";

    // Inlet overlay has 2 face_keys. Each face_key independently contributes
    // mdot_value to the cells it covers. Total = mdot_value * 2.
    constexpr int kFaceKeys = 2;
    EXPECT_NEAR(total_mdot, mdot_value * kFaceKeys, 1e-12)
        << "Total MassFlowRate across cells must equal (user value) * (face_key count)";

    // All inlet cells should have a strictly positive per-cell flux.
    for (size_t fi = 0; fi < model->fluid_bcs.size(); ++fi) {
        if (model->fluid_bcs[fi].kind == mhs::core::FluidBCType::MassFlowRateType) {
            double per_cell = model->fluid_bc_params.mass_flow_rate[model->fluid_bcs[fi].param_idx];
            EXPECT_GT(per_cell, 0.0) << "Inlet cell fi=" << fi << " has zero per-cell flux";
        }
    }
}

// =====================================================================
// Pressure-solver tests
// =====================================================================
//
// Pre-fix failure mode: solveFluidFlow() built the Poisson RHS only for
// PressureType cells. A MassFlowRate inlet + Pressure=0 outlet gave
// rhs(fi) = 0 everywhere, so the pressure field collapsed to ~0 and the
// outlet cell had no pressure-driven outflow. Energy entered at the inlet
// (via the assembler override) but nothing left, so the steady state
// ran away to millions of K.
// Post-fix: a MassFlowRate inlet contributes +m_dot/rho to the RHS at
// each inlet cell, producing a non-degenerate pressure gradient that
// drives flow out through the Dirichlet outlet.

namespace {

    // Load the canonical case + overlay, run the pressure solve, return the model.
    // The caller can then inspect model.pressure, model.flow_axes, etc.
    std::unique_ptr<mhs::core::InternalModel> load_case1_and_solve_fluid(const std::string& overlay_xml)
    {
        mhs::core::IOStructure io_data;
        if (!load_steady_case1(io_data))
            return nullptr;

        auto overlay_path = write_tmp(overlay_xml, "_solve_overlay.xml");
        auto overlay = mhs::io::read_fluid_overlay_xml(overlay_path.string());
        std::filesystem::remove(overlay_path);
        if (!overlay.has_value())
            return nullptr;

        Preprocessor preprocessor;
        auto model = preprocessor.load(io_data, overlay);
        if (!model)
            return nullptr;

        mhs::sim::solveFluidFlow(*model);
        return model;
    }

} // namespace

TEST(FluidBCTest, MassFlowRateDrivesNonDegeneratePressureField)
{
    // With a MassFlowRate inlet and Pressure=0 outlet, the Poisson system
    // must have a non-zero RHS at the inlet cells. This drives a pressure
    // gradient between inlet and outlet, so the pressure field must span
    // a positive range.
    auto model = load_case1_and_solve_fluid(make_mdot_overlay(1e-4));
    ASSERT_NE(model, nullptr);
    ASSERT_GT(model->n_fluid, 0);

    double p_min = *std::min_element(model->pressure.begin(), model->pressure.end());
    double p_max = *std::max_element(model->pressure.begin(), model->pressure.end());
    EXPECT_GT(p_max - p_min, 1.0)
        << "Pressure field must be non-degenerate when MassFlowRate drives the inlet";
}

TEST(FluidBCTest, PressureTypeOutletSitsAtSpecifiedPressure)
{
    // After the Poisson solve, any cell whose bc.kind is PressureType must
    // sit at the prescribed pressure (Dirichlet). For the canonical case
    // the outlet is Pressure=0, so we expect the outlet cells to read ~0.
    auto model = load_case1_and_solve_fluid(make_mdot_overlay(1e-4));
    ASSERT_NE(model, nullptr);

    int outlet_cells_checked = 0;
    for (size_t fi = 0; fi < model->fluid_bcs.size(); ++fi) {
        if (model->fluid_bcs[fi].kind == mhs::core::FluidBCType::PressureType) {
            EXPECT_NEAR(model->pressure[fi], 0.0, 1.0)
                << "PressureType cell fi=" << fi << " should sit at p=0";
            ++outlet_cells_checked;
        }
    }
    EXPECT_GT(outlet_cells_checked, 0) << "Test case must contain at least one PressureType cell";
}

TEST(FluidBCTest, MassFlowRateProducesDominantFlowAxes)
{
    // With a non-degenerate pressure field, the dominant-flow-axis pass
    // (Phase 3 of solveFluidFlow) should pick an axis for at least some
    // cells. Pre-fix: pressure was 0 everywhere, so all faces had zero
    // flux and flow_axes stayed at -1 for every cell.
    auto model = load_case1_and_solve_fluid(make_mdot_overlay(1e-4));
    ASSERT_NE(model, nullptr);

    int n_with_axis = 0;
    for (auto ax : model->flow_axes) {
        if (ax >= 0)
            ++n_with_axis;
    }
    EXPECT_GT(n_with_axis, 0)
        << "At least some cells must have a dominant flow axis when pressure field is non-degenerate";
}
