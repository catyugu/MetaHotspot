#pragma once

#include <limits>
#include <memory_resource>
#include <type_traits>
#include <vector>

namespace mhs::core {
    namespace detail {
        std::pmr::memory_resource* vector_resource();

        template <class T> class PoolAllocator {
        public:
            using value_type = T;
            using propagate_on_container_move_assignment = std::true_type;
            using propagate_on_container_swap = std::true_type;

            PoolAllocator() : resource_(vector_resource()) { }
            template <class U> PoolAllocator(const PoolAllocator<U>& other) noexcept : resource_(other.resource()) { }

            T* allocate(std::size_t n)
            {
                if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
                    throw std::bad_array_new_length();
                return static_cast<T*>(resource_->allocate(n * sizeof(T), alignof(T)));
            }
            void deallocate(T* p, std::size_t n) noexcept { resource_->deallocate(p, n * sizeof(T), alignof(T)); }
            std::pmr::memory_resource* resource() const noexcept { return resource_; }
            template <class U> bool operator==(const PoolAllocator<U>& other) const noexcept
            { return resource_ == other.resource(); }

        private:
            // Preserve the allocating resource across copies, moves and module
            // boundaries so deallocation always returns to the original pool.
            std::pmr::memory_resource* resource_;
        };
    }

    template <class T> using Vector = std::vector<T, detail::PoolAllocator<T>>;
} // namespace mhs::core
