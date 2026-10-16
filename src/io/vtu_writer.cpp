#include "io/result_io.hpp"

#include <tinyxml2.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace mhs::io {

    void write_vtu(const std::string &path, const mhs::core::Model &model, std::span<const double> cell_temperature) {
        using namespace tinyxml2;
        if (cell_temperature.size() != model.discrete.cells.size())
            throw std::invalid_argument("VTU state size mismatch");
        const auto &output = model.output;
        std::vector<mhs::core::Index> node_remap(output.points.size(), mhs::core::invalidIndex);
        for (const auto &cell : output.cell_vertices)
            for (auto v : cell)
                node_remap[v] = 0;
        mhs::core::Index compact = 0;
        for (auto &v : node_remap)
            if (v != mhs::core::invalidIndex)
                v = compact++;
        const auto num_points = static_cast<int>(compact);
        char buf[256];
        std::string coords_str, conn_str, off_str, type_str, cell_temp_str;
        for (size_t v = 0; v < output.points.size(); ++v) {
            if (node_remap[v] == mhs::core::invalidIndex)
                continue;
            const auto p = output.points[v];
            std::snprintf(buf, sizeof(buf), "%.17g %.17g %.17g\n", p[0], p[1], p[2]);
            coords_str += buf;
        }
        int cell_num = 0;
        for (size_t c = 0; c < output.cell_vertices.size(); ++c) {
            for (auto v : output.cell_vertices[c]) {
                conn_str += std::to_string(node_remap[v]);
                conn_str += ' ';
            }
            conn_str += '\n';
            ++cell_num;
            off_str += std::to_string(cell_num * 8) + "\n";
            type_str += "12\n";
            std::snprintf(buf, sizeof(buf), "%.17g\n",
                          cell_temperature[c]);
            cell_temp_str += buf;
        }

        XMLDocument doc;
        XMLElement *vtk_elem = doc.NewElement("VTKFile");
        vtk_elem->SetAttribute("type", "UnstructuredGrid");
        vtk_elem->SetAttribute("version", "0.1");
        vtk_elem->SetAttribute("byte_order", "LittleEndian");
        doc.InsertFirstChild(vtk_elem);

        XMLElement *grid_elem = doc.NewElement("UnstructuredGrid");
        vtk_elem->InsertEndChild(grid_elem);

        XMLElement *piece_elem = doc.NewElement("Piece");
        piece_elem->SetAttribute("NumberOfPoints", num_points);
        piece_elem->SetAttribute("NumberOfCells", cell_num);
        grid_elem->InsertEndChild(piece_elem);

        // Points: node coordinates (active nodes only)
        XMLElement *points_elem = doc.NewElement("Points");
        piece_elem->InsertEndChild(points_elem);
        XMLElement *coords_arr = doc.NewElement("DataArray");
        coords_arr->SetAttribute("type", "Float64");
        coords_arr->SetAttribute("NumberOfComponents", "3");
        coords_arr->SetAttribute("format", "ascii");
        coords_arr->SetText(coords_str.c_str());
        points_elem->InsertEndChild(coords_arr);

        // CellData: per-active-cell temperature (body-centered, no interpolation)
        XMLElement *cell_data = doc.NewElement("CellData");
        piece_elem->InsertEndChild(cell_data);
        XMLElement *temp_arr = doc.NewElement("DataArray");
        temp_arr->SetAttribute("type", "Float64");
        temp_arr->SetAttribute("Name", "Temperature");
        temp_arr->SetAttribute("NumberOfComponents", "1");
        temp_arr->SetAttribute("format", "ascii");
        temp_arr->SetText(cell_temp_str.c_str());
        cell_data->InsertEndChild(temp_arr);

        // Cells: hexahedral topology
        XMLElement *cells_elem = doc.NewElement("Cells");
        piece_elem->InsertEndChild(cells_elem);

        XMLElement *conn_arr_el = doc.NewElement("DataArray");
        conn_arr_el->SetAttribute("type", "Int32");
        conn_arr_el->SetAttribute("Name", "connectivity");
        conn_arr_el->SetAttribute("format", "ascii");
        conn_arr_el->SetText(conn_str.c_str());
        cells_elem->InsertEndChild(conn_arr_el);

        XMLElement *offsets_arr = doc.NewElement("DataArray");
        offsets_arr->SetAttribute("type", "Int32");
        offsets_arr->SetAttribute("Name", "offsets");
        offsets_arr->SetAttribute("format", "ascii");
        offsets_arr->SetText(off_str.c_str());
        cells_elem->InsertEndChild(offsets_arr);

        XMLElement *types_arr = doc.NewElement("DataArray");
        types_arr->SetAttribute("type", "UInt8");
        types_arr->SetAttribute("Name", "types");
        types_arr->SetAttribute("format", "ascii");
        types_arr->SetText(type_str.c_str());
        cells_elem->InsertEndChild(types_arr);

        const std::filesystem::path dir_path(path);
        if (!dir_path.parent_path().empty()) {
            std::filesystem::create_directories(dir_path.parent_path());
        }

        doc.SaveFile(path.c_str());
    }

} // namespace mhs::io
