// The standard FVM of Sec. III: one operator over the whole stack, iterated with
// the same fixed-point approach the paper uses for the nonlinear dies. It is both
// the accuracy reference and the timing baseline of the HDI method.
#pragma once

#include "model.hpp"

#include <vector>

namespace mhs::hdim {

    /// One step of eq. (3) over the whole stack from `previous` (an empty vector
    /// starts from the ambient). `dt` <= 0 is the steady form of eq. (2), i.e.
    /// eq. (3) without the capacity term. `tolerance` is the largest temperature
    /// change between two sweeps, in K.
    std::vector<double> solve_reference(const Model& model, const std::vector<double>& previous, double dt,
        double tolerance, int max_sweeps, RunInfo& info);

} // namespace mhs::hdim
