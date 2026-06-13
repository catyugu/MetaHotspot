#include "time_scheme.hpp"

#include "adaptive_bdf_scheme.hpp"
#include "bdf1_scheme.hpp"
#include "bdf2_scheme.hpp"
#include "common/logger.hpp"

namespace mhs::sim::time_scheme {

    std::unique_ptr<TimeScheme> create_scheme(const TimeSchemeConfig& cfg)
    {
        switch (cfg.kind) {
        case TimeSchemeKind::Bdf1:
            return std::make_unique<Bdf1Scheme>(cfg);
        case TimeSchemeKind::Bdf2:
            return std::make_unique<Bdf2Scheme>(cfg);
        case TimeSchemeKind::AdaptiveBdf:
            return std::make_unique<AdaptiveBdfScheme>(cfg);
        }
        MHS_LOG_ERROR("create_scheme: unknown TimeSchemeKind {}", static_cast<int>(cfg.kind));
    }

} // namespace mhs::sim::time_scheme
