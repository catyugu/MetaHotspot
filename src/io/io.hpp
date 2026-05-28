#pragma once

#include "model/io_model.hpp"
#include <memory>
#include <string>

namespace mhs::io {

class Reader {
public:
    explicit Reader(const std::string& xml_path);
    ~Reader() = default;

    model::IOStructure read_structure();

private:
    std::string xml_path_;
};

} // namespace mhs::io