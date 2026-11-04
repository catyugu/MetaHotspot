#pragma once

#include <limits>
#include <type_traits>
#include <vector>

namespace mhs::core {
    namespace detail {

        void* vector_allocate(std::size_t bytes, std::size_t alignment);
        void vector_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

        template <class T> class PoolAllocator {
        public:
            using value_type = T;
            using is_always_equal = std::true_type;

            PoolAllocator() noexcept = default;

            template <class U> PoolAllocator(const PoolAllocator<U>&) noexcept { }

            T* allocate(std::size_t n)
            {
                if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
                    throw std::bad_array_new_length();

                return static_cast<T*>(vector_allocate(n * sizeof(T), alignof(T)));
            }

            void deallocate(T* p, std::size_t n) noexcept { vector_deallocate(p, n * sizeof(T), alignof(T)); }
        };

        template <class T, class U> bool operator==(const PoolAllocator<T>&, const PoolAllocator<U>&) noexcept
        { return true; }

    }
    template <class T> using Vector = std::vector<T, detail::PoolAllocator<T>>;
} // namespace mhs::core
