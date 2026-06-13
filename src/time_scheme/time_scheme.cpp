#include "time_scheme.hpp"

#include "bdf1_scheme.hpp"
#include "bdf2_scheme.hpp"
#include "adaptive_bdf_scheme.hpp"

#include <utility>

namespace mhs::sim::time_scheme {

    std::unique_ptr<TimeScheme> create_scheme(const TimeSchemeConfig& cfg)
    {
        switch (cfg.kind) {
        case TimeSchemeKind::Bdf1:         return std::make_unique<Bdf1Scheme>(cfg);
        case TimeSchemeKind::Bdf2:         return std::make_unique<Bdf2Scheme>(cfg);
        case TimeSchemeKind::AdaptiveBdf:  return std::make_unique<AdaptiveBdfScheme>(cfg);
        }
        // unreachable
        return std::make_unique<Bdf1Scheme>(cfg);
    }

} // namespace mhs::sim::time_scheme
