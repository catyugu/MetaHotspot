#include "core/vector.hpp"

namespace mhs::core::detail {
    std::pmr::memory_resource* vector_resource()
    {
        // Small/medium buffers are reused; large buffers go upstream.
        static std::pmr::synchronized_pool_resource pool({16, 65536});
        return &pool;
    }
} // namespace mhs::core::detail
