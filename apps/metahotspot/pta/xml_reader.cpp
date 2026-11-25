#include "io.hpp"
#include "pta/model_input.hpp"
#include "pta/xml_helpers.hpp"

#include <stdexcept>
#include <utility>

namespace mhs::app::pta {
    using namespace tinyxml2;
    using detail::get_text;
    using detail::parse_double;

    static mhs::solver::StudyType parse_study_type(const std::string& val)
    {
        if (val == "Steady")
            return mhs::solver::StudyType::Steady;
        if (val == "Transient")
            return mhs::solver::StudyType::Transient;
        throw std::runtime_error("unknown StudyType: '" + val + "' (expected Steady or Transient)");
    }

    Input read_xml(const std::string& xml_path)
    {
        XMLDocument doc;
        if (doc.LoadFile(xml_path.c_str()) != XML_SUCCESS)
            throw std::runtime_error("failed to load XML file: " + xml_path);
        const XMLElement* root = doc.FirstChildElement("Structure");
        if (!root)
            throw std::runtime_error("no <Structure> element found in " + xml_path);

        mhs::solver::Study study;
        // StudyType (child element only)
        if (const XMLElement* study_elem = root->FirstChildElement("StudyType")) {
            study.type = parse_study_type(get_text(study_elem));
        }

        // Transient settings
        if (const XMLElement* trans = root->FirstChildElement("TransientStudyDuration")) {
            study.duration = parse_double(get_text(trans));
        }
        else if (study.type == mhs::solver::StudyType::Transient) {
            throw std::runtime_error("transient study requires <TransientStudyDuration>");
        }
        if (const XMLElement* step = root->FirstChildElement("TransientStudyTimeStep")) {
            study.output_interval = parse_double(get_text(step));
        }
        else if (study.type == mhs::solver::StudyType::Transient) {
            throw std::runtime_error("transient study requires <TransientStudyTimeStep>");
        }

        auto input = read_model(*root);
        study.initial_temperature = input.model.reference_temperature;
        return {study, std::move(input.model), std::move(input.mesh)};
    }

} // namespace mhs::app::pta
