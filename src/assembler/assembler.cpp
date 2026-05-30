#include "assembler.hpp"

namespace mhs::assembler {

    LinearSystem Assembler::assemble(const model::GlobalState& state)
    {
        (void)model_;
        (void)state;
        return {};
    }

} // namespace mhs::assembler