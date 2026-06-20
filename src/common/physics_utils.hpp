#pragma once

#include <algorithm>

namespace mhs::utils {

    /**
     * @brief Nusselt number for fully-developed laminar flow in a rectangular duct.
     *
     * Shah & London correlation for constant wall temperature (Tu),
     * valid for 0 < AR <= 1 where AR = min(w,h) / max(w,h).
     *
     *   Nu = 8.235 * (1 - 2.0421*AR + 3.0853*AR^2 - 2.4765*AR^3
     *                 + 1.0578*AR^4 - 0.1861*AR^5)
     *
     * @param w Duct width (any orientation, SI meters)
     * @param h Duct height (any orientation, SI meters)
     * @return Nusselt number (dimensionless)
     */
    inline double nusselt_rectangular(double w, double h)
    {
        double ar = std::min(w, h) / std::max(w, h);
        double ar2 = ar * ar;
        double ar3 = ar2 * ar;
        double ar4 = ar2 * ar2;
        double ar5 = ar4 * ar;

        return 8.235 * (1.0 - 2.0421 * ar + 3.0853 * ar2 - 2.4765 * ar3 + 1.0578 * ar4 - 0.1861 * ar5);
    }

    inline double f_re_rectangular(double w, double h)
    {
        double ar = std::min(w, h) / std::max(w, h);
        double ar2 = ar * ar;
        double ar3 = ar2 * ar;
        double ar4 = ar2 * ar2;
        double ar5 = ar4 * ar;
        return 24.0 * (1.0 - 1.355 * ar + 1.946 * ar2 - 1.701 * ar3 + 0.956 * ar4 - 0.253 * ar5);
    }
} // namespace mhs::utils
