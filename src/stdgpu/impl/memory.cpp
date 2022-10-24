/*
 *  Copyright 2019 Patrick Stotko
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdgpu/memory.h>

#include <cstdio>
#include <map>
#include <mutex>

#include <stdgpu/contract.h>
#include <stdgpu/platform.h>

#include STDGPU_DETAIL_BACKEND_HEADER(memory.h)

namespace stdgpu
{
namespace detail
{

/**
 * \brief A class to manage memory for size and leak detection
 */
class memory_manager
{
public:
    /**
     * \brief Registers the allocated memory block
     * \param[in] pointer A pointer to the start of the memory block
     * \param[in] size The size of the memory blocks in bytes
     * \pre !contains_memory(pointer)
     * \post contains_memory(pointer)
     * \invariant valid()
     */
    void
    register_memory(void* pointer, index64_t size);

    /**
     * \brief De-registers the allocated memory block
     * \param[in] pointer A pointer to the start of the memory block
     * \param[in] size The size of the memory blocks in bytes
     * \pre contains_memory(pointer)
     * \post !contains_memory(pointer)
     * \invariant valid()
     */
    void
    deregister_memory(void* pointer, index64_t size);

    /**
     * \brief Checks whether the memory block is registered
     * \param[in] pointer A pointer to the start of the memory block
     * \return True if the memory block is registered, false otherwise
     */
    bool
    contains_memory(void* pointer) const;

    /**
     * \brief Checks whether a (sub)memory block is registered
     * \param[in] pointer A pointer to the start of the (sub)memory block
     * \param[in] size The size of the (sub)memory blocks in bytes
     * \return True if the (sub)memory block is registered, false otherwise
     */
    bool
    contains_submemory(void* pointer, const index64_t size) const;

    /**
     * \brief Finds the size of the memory block
     * \param[in] pointer A pointer to the start of the memory block
     * \return The size of the memory block if it is registered, 0 otherwise
     */
    index64_t
    find_size(void* pointer) const;

    /**
     * \brief Returns the number of currently registered memory blocks that may leak if not de-registered
     * \return The number of registered memory blocks
     */
    index64_t
    size() const;

    /**
     * \brief Returns the total number of registered memory blocks during lifetime
     * \return The number of registered memory blocks during lifetime
     */
    index64_t
    total_registrations() const;

    /**
     * \brief Returns the total number of de-registered memory blocks during lifetime
     * \return The number of de-registered memory blocks during lifetime
     */
    index64_t
    total_deregistrations() const;

    /**
     * \brief Checks whether the internal state is valid
     * \return True if the object is valid, false otherwise
     */
    bool
    valid() const;

private:
    mutable std::recursive_mutex _mutex = {};

    std::map<void*, index64_t> _pointers = {};
    index64_t _number_insertions = 0;
    index64_t _number_erasures = 0;
};

void
memory_manager::register_memory(void* pointer, index64_t size)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    STDGPU_EXPECTS(!contains_memory(pointer));
    STDGPU_EXPECTS(valid());

    _pointers[pointer] = size;
    _number_insertions++;

    STDGPU_ENSURES(contains_memory(pointer));
    STDGPU_ENSURES(valid());
}

void
memory_manager::deregister_memory(void* pointer, [[maybe_unused]] index64_t size)
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    STDGPU_EXPECTS(contains_memory(pointer));
    STDGPU_EXPECTS(valid());

    _pointers.erase(pointer);
    _number_erasures++;

    STDGPU_ENSURES(!contains_memory(pointer));
    STDGPU_ENSURES(valid());
}

bool
memory_manager::contains_memory(void* pointer) const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _pointers.find(pointer) != std::cend(_pointers);
}

bool
memory_manager::contains_submemory(void* pointer, const index64_t size) const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    std::uint8_t* pointer_query = static_cast<std::uint8_t*>(pointer);

    for (auto it = std::cbegin(_pointers), end = _pointers.lower_bound(static_cast<void*>(pointer_query + size));
         it != end;
         ++it)
    {
        std::uint8_t* pointer_it = static_cast<std::uint8_t*>(it->first);
        index64_t size_it = it->second;

        if (pointer_it <= pointer_query && pointer_query + size <= pointer_it + size_it)
        {
            return true;
        }
    }

    return false;
}

index64_t
memory_manager::find_size(void* pointer) const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    auto it = _pointers.find(pointer);

    return (it != std::cend(_pointers)) ? it->second : 0;
}

index64_t
memory_manager::size() const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return static_cast<index64_t>(_pointers.size());
}

index64_t
memory_manager::total_registrations() const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _number_insertions;
}

index64_t
memory_manager::total_deregistrations() const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return _number_erasures;
}

bool
memory_manager::valid() const
{
    std::lock_guard<std::recursive_mutex> lock(_mutex);

    return total_registrations() - total_deregistrations() == size();
}

memory_manager&
dispatch_allocation_manager(const dynamic_memory_type type)
{
    switch (type)
    {
        case dynamic_memory_type::device:
        {
            static memory_manager manager_device;
            return manager_device;
        }

        case dynamic_memory_type::host:
        {
            static memory_manager manager_host;
            return manager_host;
        }

        case dynamic_memory_type::managed:
        {
            static memory_manager manager_managed;
            return manager_managed;
        }

        case dynamic_memory_type::invalid:
        default:
        {
            printf("stdgpu::detail::dispatch_allocation_manager : Unsupported dynamic memory type\n");
            static memory_manager pointer_null;
            return pointer_null;
        }
    }
}

void
dispatch_malloc(const dynamic_memory_type type, void** array, index64_t bytes)
{
    stdgpu::STDGPU_BACKEND_NAMESPACE::dispatch_malloc(type, array, bytes);
}

void
dispatch_free(const dynamic_memory_type type, void* array)
{
    stdgpu::STDGPU_BACKEND_NAMESPACE::dispatch_free(type, array);
}

void
dispatch_memcpy(void* destination,
                const void* source,
                index64_t bytes,
                dynamic_memory_type destination_type,
                dynamic_memory_type source_type)
{
    stdgpu::STDGPU_BACKEND_NAMESPACE::dispatch_memcpy(destination, source, bytes, destination_type, source_type);
}

void
workaround_synchronize_managed_memory()
{
    stdgpu::STDGPU_BACKEND_NAMESPACE::workaround_synchronize_managed_memory();
}

[[nodiscard]] void*
allocate(index64_t bytes, dynamic_memory_type type)
{
    if (bytes <= 0)
    {
        printf("stdgpu::detail::allocate : Number of bytes are <= 0\n");
        return nullptr;
    }

    void* array = nullptr;

    dispatch_malloc(type, &array, bytes);

    // Update pointer management after allocation
    dispatch_allocation_manager(type).register_memory(array, bytes);

    return array;
}

void
deallocate(void* p, index64_t bytes, dynamic_memory_type type)
{
    if (p == nullptr)
    {
        printf("stdgpu::detail::deallocate : Deallocating null pointer not possible\n");
        return;
    }
    if (!dispatch_allocation_manager(type).contains_memory(p))
    {
        printf("stdgpu::detail::deallocate : Deallocating unknown pointer or double freeing not possible\n");
        return;
    }

    // Update pointer management before freeing
    dispatch_allocation_manager(type).deregister_memory(p, bytes);

    dispatch_free(type, p);
}

void
memcpy(void* destination,
       const void* source,
       index64_t bytes,
       dynamic_memory_type destination_type,
       dynamic_memory_type source_type,
       const bool external_memory)
{
    if (!external_memory)
    {
        if (!dispatch_allocation_manager(destination_type).contains_submemory(destination, bytes) &&
            !dispatch_allocation_manager(dynamic_memory_type::managed).contains_submemory(destination, bytes))
        {
            printf("stdgpu::detail::memcpy : Copying to unknown destination pointer not possible\n");
            return;
        }
        if (!dispatch_allocation_manager(source_type).contains_submemory(const_cast<void*>(source), bytes) &&
            !dispatch_allocation_manager(dynamic_memory_type::managed)
                     .contains_submemory(const_cast<void*>(source), bytes))
        {
            printf("stdgpu::detail::memcpy : Copying from unknown source pointer not possible\n");
            return;
        }
    }

    dispatch_memcpy(destination, source, bytes, destination_type, source_type);
}

memory_manager&
dispatch_size_manager(const dynamic_memory_type type)
{
    switch (type)
    {
        case dynamic_memory_type::device:
        {
            static memory_manager manager_device;
            return manager_device;
        }

        case dynamic_memory_type::host:
        {
            static memory_manager manager_host;
            return manager_host;
        }

        case dynamic_memory_type::managed:
        {
            static memory_manager manager_managed;
            return manager_managed;
        }

        case dynamic_memory_type::invalid:
        default:
        {
            printf("stdgpu::detail::dispatch_size_manager : Unsupported dynamic memory type\n");
            static memory_manager pointer_null;
            return pointer_null;
        }
    }
}

} // namespace detail

template <>
dynamic_memory_type
get_dynamic_memory_type(void* array)
{
    if (detail::dispatch_size_manager(dynamic_memory_type::device).contains_memory(array))
    {
        return dynamic_memory_type::device;
    }
    if (detail::dispatch_size_manager(dynamic_memory_type::host).contains_memory(array))
    {
        return dynamic_memory_type::host;
    }
    if (detail::dispatch_size_manager(dynamic_memory_type::managed).contains_memory(array))
    {
        return dynamic_memory_type::managed;
    }

    return dynamic_memory_type::invalid;
}

template <>
void
register_memory(void* p, index64_t n, dynamic_memory_type memory_type)
{
    if (p == nullptr)
    {
        printf("stdgpu::register_memory : Registering a null pointer not possible\n");
        return;
    }
    if (n <= 0)
    {
        printf("stdgpu::register_memory : Registering pointer with number of bytes <= 0 not possible\n");
        return;
    }
    if (detail::dispatch_size_manager(memory_type).contains_memory(p))
    {
        printf("stdgpu::register_memory : Registering already known pointer not possible\n");
        return;
    }
    detail::dispatch_size_manager(memory_type).register_memory(p, n);
}

template <>
void
deregister_memory(void* p, index64_t n, dynamic_memory_type memory_type)
{
    if (!detail::dispatch_size_manager(memory_type).contains_memory(p))
    {
        printf("stdgpu::deregister_memory : Deregistering unknown pointer not possible\n");
        return;
    }
    detail::dispatch_size_manager(memory_type).deregister_memory(p, n);
}

index64_t
get_allocation_count(dynamic_memory_type memory_type)
{
    return detail::dispatch_allocation_manager(memory_type).total_registrations();
}

index64_t
get_deallocation_count(dynamic_memory_type memory_type)
{
    return detail::dispatch_allocation_manager(memory_type).total_deregistrations();
}

template <>
index64_t
size_bytes(void* array)
{
    dynamic_memory_type type = get_dynamic_memory_type(array);

    index64_t array_size_bytes = detail::dispatch_size_manager(type).find_size(array);
    if (array_size_bytes == 0)
    {
        printf("stdgpu::size_bytes : Array not registered or not pointing to the first element. Returning 0 ...\n");
        return 0;
    }

    return array_size_bytes;
}

} // namespace stdgpu
