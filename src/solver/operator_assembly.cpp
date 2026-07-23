#include "solver/operator_assembly.hpp"

#include <iterator>
#include <utility>

namespace mhs::sim {

    OperatorAccumulator::OperatorAccumulator(Eigen::Index state_count)
        : state_count_(state_count), source_(Eigen::VectorXd::Zero(state_count))
    {
    }

    void OperatorAccumulator::add(OperatorContribution contribution)
    {
        stiffness_.insert(stiffness_.end(), std::make_move_iterator(contribution.stiffness.begin()),
            std::make_move_iterator(contribution.stiffness.end()));
        capacity_.insert(capacity_.end(), std::make_move_iterator(contribution.capacity.begin()),
            std::make_move_iterator(contribution.capacity.end()));
        for (const auto& entry : contribution.source)
            source_(entry.row) += entry.value;
    }

    AssemblyResult OperatorAccumulator::finish() &&
    {
        Eigen::SparseMatrix<double> stiffness(state_count_, state_count_);
        stiffness.setFromTriplets(stiffness_.begin(), stiffness_.end());

        Eigen::SparseMatrix<double> capacity(state_count_, state_count_);
        capacity.setFromTriplets(capacity_.begin(), capacity_.end());

        return {std::move(stiffness), std::move(capacity), std::move(source_)};
    }

} // namespace mhs::sim
