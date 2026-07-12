#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinyxml2.h>

#include "smart_block_reader.hpp"

namespace mhs::core {

    SmartMacroModelData read_smart_macro_model(const std::string& xml_path)
    {
        tinyxml2::XMLDocument doc;
        if (doc.LoadFile(xml_path.c_str()) != tinyxml2::XML_SUCCESS) {
            throw std::runtime_error("Failed to load SmartMacro model XML: " + xml_path);
        }

        const auto* root = doc.FirstChildElement("SmartMacroModel");
        if (!root) {
            throw std::runtime_error("No <SmartMacroModel> element found in " + xml_path);
        }

        SmartMacroModelData result;

        // Name
        if (const auto* name_elem = root->FirstChildElement("Name")) {
            if (const char* text = name_elem->GetText())
                result.name = std::string(text);
        }

        // NPorts
        int n_ports = 0;
        if (const auto* np_elem = root->FirstChildElement("NPorts")) {
            if (const char* text = np_elem->GetText())
                n_ports = std::stoi(text);
        }
        else {
            throw std::runtime_error("Missing <NPorts> in SmartMacro model");
        }

        if (n_ports <= 0) {
            throw std::runtime_error("Invalid NPorts=" + std::to_string(n_ports) + " in SmartMacro model");
        }

        // PortOrder
        result.port_ix.reserve(n_ports);
        result.port_iy.reserve(n_ports);
        result.port_iz.reserve(n_ports);

        const auto* po_elem = root->FirstChildElement("PortOrder");
        if (!po_elem) {
            throw std::runtime_error("Missing <PortOrder> in SmartMacro model");
        }

        for (const auto* port_elem = po_elem->FirstChildElement("Port"); port_elem;
            port_elem = port_elem->NextSiblingElement("Port")) {

            int ix = 0, iy = 0, iz = 0;
            if (const auto* e = port_elem->FirstChildElement("IX"))
                ix = std::stoi(e->GetText() ? e->GetText() : "0");
            if (const auto* e = port_elem->FirstChildElement("IY"))
                iy = std::stoi(e->GetText() ? e->GetText() : "0");
            if (const auto* e = port_elem->FirstChildElement("IZ"))
                iz = std::stoi(e->GetText() ? e->GetText() : "0");

            result.port_ix.push_back(ix);
            result.port_iy.push_back(iy);
            result.port_iz.push_back(iz);
        }

        if (static_cast<int>(result.port_ix.size()) != n_ports) {
            throw std::runtime_error("PortOrder has " + std::to_string(result.port_ix.size())
                + " entries but NPorts=" + std::to_string(n_ports));
        }

        // DataFile: resolve relative to XML directory
        std::string data_file;
        if (const auto* df_elem = root->FirstChildElement("DataFile")) {
            if (const char* text = df_elem->GetText())
                data_file = std::string(text);
        }
        if (data_file.empty()) {
            throw std::runtime_error("Missing <DataFile> in SmartMacro model");
        }

        auto data_path = std::filesystem::path(xml_path).parent_path() / data_file;

        // Read binary data: [f_port: N doubles][K_port: N*N doubles, row-major]
        std::ifstream bin(data_path, std::ios::binary);
        if (!bin) {
            throw std::runtime_error("Failed to open binary data file: " + data_path.string());
        }

        // Read f_port
        result.f_port.resize(n_ports);
        bin.read(reinterpret_cast<char*>(result.f_port.data()), n_ports * sizeof(double));
        if (!bin) {
            throw std::runtime_error("Failed to read f_port from binary data");
        }

        // Read K_port
        result.K_port.resize(n_ports, n_ports);
        std::vector<double> buf(n_ports);
        for (int r = 0; r < n_ports; ++r) {
            bin.read(reinterpret_cast<char*>(buf.data()), n_ports * sizeof(double));
            if (!bin) {
                throw std::runtime_error("Failed to read K_port row " + std::to_string(r));
            }
            for (int c = 0; c < n_ports; ++c) {
                result.K_port(r, c) = buf[c];
            }
        }

        return result;
    }

} // namespace mhs::core
