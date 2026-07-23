#include "io/result_io.hpp"

#include <tinyxml2.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace mhs::io {

    void write_vtu(const std::string& path, const mhs::core::Model& model, const std::vector<double>& cell_temperature)
    {
        using namespace tinyxml2;
        const auto& mesh = model.mesh;
        const auto& cells = model.cells;
        const mhs::core::Index node_nx = mesh.nx + 1;
        const mhs::core::Index node_ny = mesh.ny + 1;
        const mhs::core::Index node_nz = mesh.nz + 1;

        const mhs::core::Index total_nodes = node_nx * node_ny * node_nz;
        std::vector<mhs::core::Index> node_remap(total_nodes, mhs::core::invalidIndex);

        auto node_idx = [](mhs::core::Index vx, mhs::core::Index vy, mhs::core::Index vz, mhs::core::Index nny,
                            mhs::core::Index nnz) { return vx * nny * nnz + vy * nnz + vz; };

        // Pass 1: mark all nodes referenced by at least one active cell
        for (mhs::core::Index ix = 0; ix < mesh.nx; ix++) {
            for (mhs::core::Index iy = 0; iy < mesh.ny; iy++) {
                for (mhs::core::Index iz = 0; iz < mesh.nz; iz++) {
                    const mhs::core::Index old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    if (cells.grid_to_cell[old_idx] == mhs::core::invalidIndex)
                        continue;

                    const mhs::core::Index n[8]
                        = {node_idx(ix, iy, iz, node_ny, node_nz), node_idx(ix + 1, iy, iz, node_ny, node_nz),
                            node_idx(ix + 1, iy + 1, iz, node_ny, node_nz), node_idx(ix, iy + 1, iz, node_ny, node_nz),
                            node_idx(ix, iy, iz + 1, node_ny, node_nz), node_idx(ix + 1, iy, iz + 1, node_ny, node_nz),
                            node_idx(ix + 1, iy + 1, iz + 1, node_ny, node_nz),
                            node_idx(ix, iy + 1, iz + 1, node_ny, node_nz)};
                    for (int k = 0; k < 8; k++) {
                        if (node_remap[n[k]] == mhs::core::invalidIndex)
                            node_remap[n[k]] = 0; // mark as referenced
                    }
                }
            }
        }

        // Pass 2: compact node remap — assign sequential output indices
        mhs::core::Index compact_count = 0;
        for (mhs::core::Index i = 0; i < total_nodes; i++) {
            if (node_remap[i] != mhs::core::invalidIndex)
                node_remap[i] = compact_count++;
        }

        const int num_points = static_cast<int>(compact_count);

        char buf[64];
        std::string coords_str;
        for (mhs::core::Index vx = 0; vx < node_nx; vx++) {
            for (mhs::core::Index vy = 0; vy < node_ny; vy++) {
                for (mhs::core::Index vz = 0; vz < node_nz; vz++) {
                    const mhs::core::Index i = node_idx(vx, vy, vz, node_ny, node_nz);
                    if (node_remap[i] == mhs::core::invalidIndex)
                        continue;
                    const double node_x
                        = (vx == 0) ? mesh.cx[0] - mesh.dx[0] * 0.5 : mesh.cx[vx - 1] + mesh.dx[vx - 1] * 0.5;
                    const double node_y
                        = (vy == 0) ? mesh.cy[0] - mesh.dy[0] * 0.5 : mesh.cy[vy - 1] + mesh.dy[vy - 1] * 0.5;
                    const double node_z
                        = (vz == 0) ? mesh.cz[0] - mesh.dz[0] * 0.5 : mesh.cz[vz - 1] + mesh.dz[vz - 1] * 0.5;
                    std::snprintf(buf, sizeof(buf), "%.8g %.8g %.8g\n", node_x, node_y, node_z);
                    coords_str += buf;
                }
            }
        }

        std::string conn_str;
        std::string off_str;
        std::string type_str;
        std::string cell_temp_str;
        int cell_num = 0;

        for (mhs::core::Index ix = 0; ix < mesh.nx; ix++) {
            for (mhs::core::Index iy = 0; iy < mesh.ny; iy++) {
                for (mhs::core::Index iz = 0; iz < mesh.nz; iz++) {
                    const mhs::core::Index old_idx = ix * mesh.ny * mesh.nz + iy * mesh.nz + iz;
                    const mhs::core::Index compact_cell = cells.grid_to_cell[old_idx];
                    if (compact_cell == mhs::core::invalidIndex)
                        continue;

                    const int n[8] = {static_cast<int>(node_idx(ix, iy, iz, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix + 1, iy, iz, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix + 1, iy + 1, iz, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix, iy + 1, iz, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix, iy, iz + 1, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix + 1, iy, iz + 1, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix + 1, iy + 1, iz + 1, node_ny, node_nz)),
                        static_cast<int>(node_idx(ix, iy + 1, iz + 1, node_ny, node_nz))};

                    std::snprintf(buf, sizeof(buf), "%d %d %d %d %d %d %d %d\n", static_cast<int>(node_remap[n[0]]),
                        static_cast<int>(node_remap[n[1]]), static_cast<int>(node_remap[n[2]]),
                        static_cast<int>(node_remap[n[3]]), static_cast<int>(node_remap[n[4]]),
                        static_cast<int>(node_remap[n[5]]), static_cast<int>(node_remap[n[6]]),
                        static_cast<int>(node_remap[n[7]]));
                    conn_str += buf;

                    cell_num++;
                    std::snprintf(buf, sizeof(buf), "%d\n", cell_num * 8);
                    off_str += buf;
                    type_str += "12\n";

                    std::snprintf(buf, sizeof(buf), "%.8g\n", cell_temperature[compact_cell]);
                    cell_temp_str += buf;
                }
            }
        }

        XMLDocument doc;
        XMLElement* vtk_elem = doc.NewElement("VTKFile");
        vtk_elem->SetAttribute("type", "UnstructuredGrid");
        vtk_elem->SetAttribute("version", "0.1");
        vtk_elem->SetAttribute("byte_order", "LittleEndian");
        doc.InsertFirstChild(vtk_elem);

        XMLElement* grid_elem = doc.NewElement("UnstructuredGrid");
        vtk_elem->InsertEndChild(grid_elem);

        XMLElement* piece_elem = doc.NewElement("Piece");
        piece_elem->SetAttribute("NumberOfPoints", num_points);
        piece_elem->SetAttribute("NumberOfCells", cell_num);
        grid_elem->InsertEndChild(piece_elem);

        // Points: node coordinates (active nodes only)
        XMLElement* points_elem = doc.NewElement("Points");
        piece_elem->InsertEndChild(points_elem);
        XMLElement* coords_arr = doc.NewElement("DataArray");
        coords_arr->SetAttribute("type", "Float64");
        coords_arr->SetAttribute("NumberOfComponents", "3");
        coords_arr->SetAttribute("format", "ascii");
        coords_arr->SetText(coords_str.c_str());
        points_elem->InsertEndChild(coords_arr);

        // CellData: per-active-cell temperature (body-centered, no interpolation)
        XMLElement* cell_data = doc.NewElement("CellData");
        piece_elem->InsertEndChild(cell_data);
        XMLElement* temp_arr = doc.NewElement("DataArray");
        temp_arr->SetAttribute("type", "Float64");
        temp_arr->SetAttribute("Name", "Temperature");
        temp_arr->SetAttribute("NumberOfComponents", "1");
        temp_arr->SetAttribute("format", "ascii");
        temp_arr->SetText(cell_temp_str.c_str());
        cell_data->InsertEndChild(temp_arr);

        // Cells: hexahedral topology
        XMLElement* cells_elem = doc.NewElement("Cells");
        piece_elem->InsertEndChild(cells_elem);

        XMLElement* conn_arr_el = doc.NewElement("DataArray");
        conn_arr_el->SetAttribute("type", "Int32");
        conn_arr_el->SetAttribute("Name", "connectivity");
        conn_arr_el->SetAttribute("format", "ascii");
        conn_arr_el->SetText(conn_str.c_str());
        cells_elem->InsertEndChild(conn_arr_el);

        XMLElement* offsets_arr = doc.NewElement("DataArray");
        offsets_arr->SetAttribute("type", "Int32");
        offsets_arr->SetAttribute("Name", "offsets");
        offsets_arr->SetAttribute("format", "ascii");
        offsets_arr->SetText(off_str.c_str());
        cells_elem->InsertEndChild(offsets_arr);

        XMLElement* types_arr = doc.NewElement("DataArray");
        types_arr->SetAttribute("type", "UInt8");
        types_arr->SetAttribute("Name", "types");
        types_arr->SetAttribute("format", "ascii");
        types_arr->SetText(type_str.c_str());
        cells_elem->InsertEndChild(types_arr);

        const std::filesystem::path dir_path(path);
        if (!dir_path.parent_path().empty() && !std::filesystem::exists(dir_path.parent_path())) {
            std::filesystem::create_directories(dir_path.parent_path());
        }

        doc.SaveFile(path.c_str());
    }

} // namespace mhs::io
