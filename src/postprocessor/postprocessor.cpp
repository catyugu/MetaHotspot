#include "postprocessor.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace mhs {

void Postprocessor::writeVTU(const std::string& path,
                             const model::InternalModel& model,
                             const std::vector<double>& solution)
{
    const auto& mesh = model.mesh;
    int nx = mesh.nx;
    int ny = mesh.ny;
    int nz = mesh.nz;

    std::ofstream file(path);
    if (!file.is_open()) {
        return;
    }

    // VTK XML header
    file << "<?xml version=\"1.0\"?>\n";
    file << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    file << "  <UnstructuredGrid>\n";

    // Number of points = (nx+1)*(ny+1)*(nz+1)
    int npoints = (nx + 1) * (ny + 1) * (nz + 1);
    int ncells = nx * ny * nz;

    file << "    <Piece NumberOfPoints=\"" << npoints << "\" NumberOfCells=\"" << ncells << "\">\n";

    // Points
    file << "      <Points>\n";
    file << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";

    // Write vertex coordinates
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                file << "          " << mesh.vertex_x[i] << " "
                     << mesh.vertex_y[j] << " "
                     << mesh.vertex_z[k] << "\n";
            }
        }
    }
    file << "        </DataArray>\n";
    file << "      </Points>\n";

    // Cells/connectivity
    file << "      <Cells>\n";
    file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";

    // Each cell has 8 corners (hexahedron)
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int idx[8];
                idx[0] = (k) * (nx + 1) * (ny + 1) + (j) * (nx + 1) + i;
                idx[1] = idx[0] + 1;
                idx[2] = idx[0] + (nx + 1);
                idx[3] = idx[2] + 1;
                idx[4] = idx[0] + (nx + 1) * (ny + 1);
                idx[5] = idx[4] + 1;
                idx[6] = idx[4] + (nx + 1);
                idx[7] = idx[6] + 1;

                file << "          " << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3] << " "
                     << idx[4] << " " << idx[5] << " " << idx[6] << " " << idx[7] << "\n";
            }
        }
    }
    file << "        </DataArray>\n";

    // Offsets
    file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (int c = 0; c < ncells; ++c) {
        file << "          " << (c + 1) * 8 << "\n";
    }
    file << "        </DataArray>\n";

    // Cell types (hexahedron = 12)
    file << "        <DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\n";
    for (int c = 0; c < ncells; ++c) {
        file << "          12\n";
    }
    file << "        </DataArray>\n";
    file << "      </Cells>\n";

    // Point data (temperature)
    file << "      <PointData Scalars=\"Temperature\">\n";
    file << "        <DataArray type=\"Float64\" Name=\"Temperature\" format=\"ascii\">\n";

    // Interpolate solution to vertices
    for (int k = 0; k <= nz; ++k) {
        for (int j = 0; j <= ny; ++j) {
            for (int i = 0; i <= nx; ++i) {
                // Find cell containing this vertex
                int ci = std::min(i, nx - 1);
                int cj = std::min(j, ny - 1);
                int ck = std::min(k, nz - 1);

                // Simple averaging from nearby cells
                double T = 0.0;
                int count = 0;

                for (int d3 = 0; d3 <= 1 && ck + d3 < nz; ++d3) {
                    for (int d2 = 0; d2 <= 1 && cj + d2 < ny; ++d2) {
                        for (int d1 = 0; d1 <= 1 && ci + d1 < nx; ++d1) {
                            int cell_idx = (ck + d3) * nx * ny + (cj + d2) * nx + (ci + d1);
                            if (cell_idx < static_cast<int>(solution.size())) {
                                T += solution[cell_idx];
                                ++count;
                            }
                        }
                    }
                }

                if (count > 0) {
                    T /= count;
                }

                file << "          " << std::fixed << std::setprecision(6) << T << "\n";
            }
        }
    }
    file << "        </DataArray>\n";
    file << "      </PointData>\n";

    file << "    </Piece>\n";
    file << "  </UnstructuredGrid>\n";
    file << "</VTKFile>\n";

    file.close();
}

void Postprocessor::writeXML(const std::string& path,
                             const model::InternalModel& model,
                             const std::vector<double>& solution)
{
    const auto& mesh = model.mesh;

    std::ofstream file(path);
    if (!file.is_open()) {
        return;
    }

    file << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    file << "<Structure xmlns=\"http://schemas.datacontract.org/2004/07/ThermalSim.Models\"\n";
    file << "    xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\">\n";

    // Write mesh and temperature results
    file << "    <Results xmlns:a=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\">\n";
    file << "        <a:anyType i:type=\"Result3D\">\n";

    // Mesh
    file << "            <Mesh xmlns:b=\"http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh\">\n";
    file << "                <b:XArray>\n";
    for (const auto& x : mesh.vertex_x) {
        file << "                    <a:double>" << x << "</a:double>\n";
    }
    file << "                </b:XArray>\n";
    file << "                <b:YArray>\n";
    for (const auto& y : mesh.vertex_y) {
        file << "                    <a:double>" << y << "</a:double>\n";
    }
    file << "                </b:YArray>\n";
    file << "                <b:ZArray>\n";
    for (const auto& z : mesh.vertex_z) {
        file << "                    <a:double>" << z << "</a:double>\n";
    }
    file << "                </b:ZArray>\n";
    file << "            </Mesh>\n";

    // Temperature values
    file << "            <Name>Temperature</Name>\n";
    file << "            <Values>\n";
    file << "                <Data>\n";

    for (const auto& T : solution) {
        file << "                    <a:double>" << std::fixed << std::setprecision(6) << T << "</a:double>\n";
    }

    file << "                </Data>\n";
    file << "                <SizeX>" << mesh.nx << "</SizeX>\n";
    file << "                <SizeY>" << mesh.ny << "</SizeY>\n";
    file << "                <SizeZ>" << mesh.nz << "</SizeZ>\n";
    file << "            </Values>\n";
    file << "        </a:anyType>\n";
    file << "    </Results>\n";
    file << "</Structure>\n";

    file.close();
}

double Postprocessor::max_temperature(const std::vector<double>& T) const
{
    if (T.empty()) {
        return 0.0;
    }
    double max_val = T[0];
    for (const auto& v : T) {
        if (v > max_val) {
            max_val = v;
        }
    }
    return max_val;
}

double Postprocessor::min_temperature(const std::vector<double>& T) const
{
    if (T.empty()) {
        return 0.0;
    }
    double min_val = T[0];
    for (const auto& v : T) {
        if (v < min_val) {
            min_val = v;
        }
    }
    return min_val;
}

} // namespace mhs