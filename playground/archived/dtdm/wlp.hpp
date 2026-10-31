// Weighted-Laguerre-polynomial (WLP) time transformation, paper eqs. (2)-(4).
//
//   phi_p(x) = exp(-x/2) L_p(x),   x = s t   (x is the paper's t-bar)
//
// orthonormal on [0, inf), with the derivative identity
//
//   phi_q'(x) = -(1/2) phi_q(x) - sum_{k<q} phi_k(x)
//
// whose Galerkin projection gives the paper's order-independent operator
//
//   (K + (s/2) C) T_q = int g phi_q dx - s C sum_{k<q} T_k
#pragma once

#include <cmath>
#include <vector>

namespace mhs::dtdm {

    /// L_0..L_{order-1} at x, by the forward three-term recurrence.
    inline void laguerre_values(int order, double x, std::vector<double>& out)
    {
        out.assign(static_cast<std::size_t>(order), 0.0);
        if (order == 0)
            return;
        out[0] = 1.0;
        if (order == 1)
            return;
        out[1] = 1.0 - x;
        for (int p = 1; p + 1 < order; ++p)
            out[static_cast<std::size_t>(p + 1)]
                = ((2.0 * p + 1.0 - x) * out[static_cast<std::size_t>(p)] - p * out[static_cast<std::size_t>(p - 1)])
                / (p + 1.0);
    }

    /// phi_0..phi_{order-1} at x.
    inline void phi(int order, double x, std::vector<double>& out)
    {
        laguerre_values(order, x, out);
        const double decay = std::exp(-0.5 * x);
        for (auto& value : out)
            value *= decay;
    }

} // namespace mhs::dtdm
