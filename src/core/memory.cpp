#include "core/vector.hpp"
#include <memory_resource>

namespace mhs::core::detail {
    namespace {

        std::pmr::memory_resource* vector_pool()
        {
            // Small/medium buffers are reused; large buffers go upstream.
            static std::pmr::synchronized_pool_resource pool({16, 65536});
            return &pool;
        }
    }

    void* vector_allocate(std::size_t bytes, std::size_t alignment)
    { return vector_pool()->allocate(bytes, alignment); }

    void vector_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept
    { vector_pool()->deallocate(ptr, bytes, alignment); }
} // namespace mhs::core::detail
