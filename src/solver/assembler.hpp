#pragma once

#include "core/csr_matrix.hpp"

#include "model/model.hpp"
#include <span>

namespace mhs::solver {

    /// Operators K, C, f of the linearised system: C * dx/dt + K * x = f.
    struct Operators {
        mhs::core::CSRMatrix<double> K;
        mhs::core::CSRMatrix<double> C;
        mhs::core::Vector<double> f;
    };

    // One workspace per solve, for a fixed model topology. Numerical buffers
    // are overwritten on every assembly, including explicit zero coefficients.
    struct ThermalAssembly {
        explicit ThermalAssembly(const mhs::model::Model& model);
        void assemble(const mhs::model::Model& model, std::span<const double> temperature, double time);

        Operators operators;
        mhs::core::Vector<mhs::core::SparseIndex> diagonal_positions;

    private:
        mhs::core::Vector<mhs::model::Vec3> conductivity;
        mhs::core::Vector<double> density, specific_heat;
    };

    // Owning export for the C API and standalone operator inspection.
    Operators assemble_thermal(const mhs::model::Model& model, std::span<const double> temperature, double time);

} // namespace mhs::solver
