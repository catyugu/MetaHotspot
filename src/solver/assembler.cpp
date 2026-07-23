#include "solver/assembler.hpp"

#include "solver/cell_assembler.hpp"
#include "solver/fluid_assembler.hpp"

#include <cassert>
#include <limits>
#include <utility>

namespace mhs::sim {

    AssemblyResult Assembler::assemble(const AssembleContext& context) const
    {
        const auto state_count = model_.dofs.total_count;
        assert(context.state.size() == state_count);
        assert(state_count <= static_cast<mhs::core::Index>(std::numeric_limits<Eigen::Index>::max()));
        const auto eigen_count = static_cast<Eigen::Index>(state_count);

        OperatorAccumulator accumulator(eigen_count);
        accumulator.add(assemble_cell_domain(model_, context));
        accumulator.add(mhs::sim::fluid::assemble_operator(model_, context));
        return std::move(accumulator).finish();
    }

} // namespace mhs::sim
