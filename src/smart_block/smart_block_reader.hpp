#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>

namespace mhs::core {

    /// Parse a trained SmartMacro model.
    ///
    /// Format: tiny XML <SmartMacroModel> with a <DataFile> sibling .data binary:
    ///   <SmartMacroModel>
    ///     <Name>copper_block</Name>
    ///     <NPorts>400</NPorts>
    ///     <DataFile>trained_copper.data</DataFile>
    ///     <PortOrder>
    ///       <Port><IX>0</IX><IY>0</IY><IZ>3</IZ></Port>
    ///       ...
    ///     </PortOrder>
    ///   </SmartMacroModel>
    ///
    /// The .data file is little-endian doubles: [f_port (N doubles)] [K_port (N*N doubles, row-major)].
    ///
    /// Returns a struct with:
    ///   - name        : the model name
    ///   - K_port      : dense DtN matrix
    ///   - port_ix     : original-grid X indices (size N_ports)
    ///   - port_iy     : original-grid Y indices (size N_ports)
    ///   - port_iz     : original-grid Z indices (size N_ports)
    struct SmartMacroModelData {
        std::string name;
        Eigen::MatrixXd K_port;
        Eigen::VectorXd f_port; // RHS vector from BCs (size N_ports)
        std::vector<int> port_ix;
        std::vector<int> port_iy;
        std::vector<int> port_iz;
    };

    SmartMacroModelData read_smart_macro_model(const std::string& xml_path);

} // namespace mhs::core
