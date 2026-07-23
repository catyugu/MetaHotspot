#include "io/face_region_parser.hpp"

#include <string>
#include <vector>

namespace mhs::io::detail {
    namespace {
        std::vector<std::string> split(const std::string& value, char delimiter)
        {
            std::vector<std::string> parts;
            std::string current;
            for (char character : value) {
                if (character == delimiter) {
                    parts.push_back(current);
                    current.clear();
                }
                else {
                    current += character;
                }
            }
            if (!current.empty())
                parts.push_back(current);
            return parts;
        }

        mhs::model::Axis parse_axis(const std::string& value)
        {
            if (!value.empty() && value[0] == 'X')
                return mhs::model::Axis::X;
            if (!value.empty() && value[0] == 'Y')
                return mhs::model::Axis::Y;
            return mhs::model::Axis::Z;
        }
    } // namespace

    mhs::model::FaceRegion parse_face_region(const std::string& encoded)
    {
        const auto parts = split(encoded, '|');
        mhs::model::FaceRegion region;
        if (parts.size() < 3)
            return region;

        region.axis = parse_axis(parts[0]);
        region.coordinate = std::stod(parts[2]);

        if (region.axis == mhs::model::Axis::Z && parts.size() == 4) {
            for (const auto& encoded_rectangle : split(parts[3], ';')) {
                const auto values = split(encoded_rectangle, ',');
                if (values.size() == 4) {
                    region.rectangles.push_back(
                        {std::stod(values[0]), std::stod(values[1]), std::stod(values[2]), std::stod(values[3])});
                }
            }
        }
        else if ((region.axis == mhs::model::Axis::X || region.axis == mhs::model::Axis::Y) && parts.size() >= 7) {
            region.rectangles.push_back(
                {std::stod(parts[3]), std::stod(parts[4]), std::stod(parts[5]), std::stod(parts[6])});
        }

        return region;
    }

} // namespace mhs::io::detail
