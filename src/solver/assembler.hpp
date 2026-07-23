#pragma once

#include "runtime/model.hpp"
#include "solver/operator_assembly.hpp"
#include <vector>

namespace mhs::sim {

    class Assembler {
    public:
        explicit Assembler(const mhs::core::Model& model) : model_(model) { }
        ~Assembler() = default;

        /// Assemble C * dx/dt + K * x = f over the complete state layout.
        AssemblyResult assemble(const AssembleContext& ctx) const;

    private:
        const mhs::core::Model& model_;
    };

} // namespace mhs::sim
