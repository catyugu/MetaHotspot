#include "postprocessor.hpp"

namespace mhs {

    void Postprocessor::writeVTU(const std::string& path, const model::InternalModel& model, const std::vector<double>& solution)
    {
        (void)path;
        (void)model;
        (void)solution;
    }

    void Postprocessor::writeXML(const std::string& path, const model::InternalModel& model, const std::vector<double>& solution)
    {
        (void)path;
        (void)model;
        (void)solution;
    }

} // namespace mhs
