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

#ifndef STDGPU_MUTEX_H
#define STDGPU_MUTEX_H

#include <stdgpu/impl/platform_check.h>

/**
 * \addtogroup mutex mutex
 * \ingroup utilities
 * @{
 */

/**
 * \file stdgpu/mutex.cuh
 */

#include <stdgpu/bitset.cuh>
#include <stdgpu/cstddef.h>
#include <stdgpu/platform.h>

///////////////////////////////////////////////////////////

#include <stdgpu/mutex_fwd>

///////////////////////////////////////////////////////////

namespace stdgpu
{

/**
 * \ingroup mutex
 * \brief A class to model a mutex array on the GPU
 * \tparam Block The internal bit block type
 * \tparam Allocator The allocator type
 *
 * Differences to std::mutex:
 *  - Mutexes must be modeled as containers since threads have to call the exact same object
 *  - Manual allocation and destruction of container required
 *  - No guaranteed valid state
 *  - Blocking lock is not supported
 */
template <typename Block, typename Allocator>
class mutex_array
{
public:
    /**
     * \brief A proxy class to model a mutex reference on the GPU
     *
     * Differences to std::mutex:
     *  - No equivalent analogue
     *  - Additional locked function to check the lock state of the mutex
     */
    class reference
    {
    public:
        /**
         * \brief Deleted constructor
         */
        STDGPU_HOST_DEVICE
        reference() = delete;

        /**
         * \brief Tries to lock the mutex
         * \return True if the mutex has been locked, false otherwise
         */
        STDGPU_DEVICE_ONLY bool
        try_lock();

        /**
         * \brief Unlocks the mutex
         */
        STDGPU_DEVICE_ONLY void
        unlock();

        /**
         * \brief Checks whether the mutex is locked
         * \return True if the mutex is locked, false otherwise
         */
        STDGPU_DEVICE_ONLY bool
        locked() const;

    private:
        friend mutex_array;

        STDGPU_HOST_DEVICE
        explicit reference(const typename bitset<Block, Allocator>::reference& bit_ref);

        typename bitset<Block, Allocator>::reference _bit_ref;
    };

    using allocator_type = Allocator; /**< Allocator */

    /**
     * \brief Creates an object of this class on the GPU (device)
     * \param[in] size The size of this object
     * \param[in] allocator The allocator instance to use
     * \return A newly created object of this class allocated on the GPU (device)
     */
    static mutex_array
    createDeviceObject(const index_t& size, const Allocator& allocator = Allocator());

    /**
     * \brief Destroys the given object of this class on the GPU (device)
     * \param[in] device_object The object allocated on the GPU (device)
     */
    static void
    destroyDeviceObject(mutex_array& device_object);

    /**
     * \brief Empty constructor
     */
    mutex_array() noexcept = default;

    /**
     * \brief Returns the container allocator
     * \return The container allocator
     */
    STDGPU_HOST_DEVICE allocator_type
    get_allocator() const noexcept;

    /**
     * \brief Returns a reference to the n-th mutex
     * \param[in] n The position of the requested mutex
     * \return The n-th mutex
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY reference
    operator[](const index_t n);

    /**
     * \brief Returns a reference to the n-th mutex
     * \param[in] n The position of the requested mutex
     * \return The n-th mutex
     * \pre 0 <= n < size()
     */
    STDGPU_DEVICE_ONLY const reference
    operator[](const index_t n) const;

    /**
     * \brief Checks if this object is empty
     * \return True if this object is empty, false otherwise
     */
    [[nodiscard]] STDGPU_HOST_DEVICE bool
    empty() const noexcept;

    /**
     * \brief The size
     * \return The size of the object
     */
    STDGPU_HOST_DEVICE index_t
    size() const noexcept;

    /**
     * \brief Checks if the object is in valid state
     * \return True if the state is valid, false otherwise
     */
    bool
    valid() const;

private:
    explicit mutex_array(const bitset<Block, Allocator>& lock_bits);

    bitset<Block, Allocator> _lock_bits = {};
};

/**
 * \ingroup mutex
 * \brief Tryies to lock all the locks at the given positions {lock1, lock2, ..., lockn} for some n >= 1
 * \param[in] lock1 The first lock
 * \param[in] lock2 The second lock
 * \param[in] lockn The remaining n - 2 locks
 * \return -1 if locking was successful, the first position at which locking failed otherwise
 */
template <typename Lockable1, typename Lockable2, typename... LockableN>
STDGPU_DEVICE_ONLY int
try_lock(Lockable1 lock1, Lockable2 lock2, LockableN... lockn);

} // namespace stdgpu

/**
 * @}
 */

#include <stdgpu/impl/mutex_detail.cuh>

#endif // STDGPU_MUTEX_H
