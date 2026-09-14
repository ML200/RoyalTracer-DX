#pragma once

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/cuda_check.h"

namespace pb {

//====================================
//Minimal RAII wrapper around a typed device allocation. Move-only. Zero is
//treated as an empty buffer (ptr == nullptr, count == 0). Resize is
//destructive: it frees the old buffer and allocates a new one without
//preserving contents.
//====================================

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) {
        resize(count);
    }

    ~DeviceBuffer() {
        free();
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_   = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_         = other.ptr_;
            count_       = other.count_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void resize(std::size_t count) {
        if (count == count_) return;
        free();
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
            count_ = count;
        }
    }

    void zero() {
        if (ptr_ && count_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

    void upload(const T* host, std::size_t count) {
        if (count == 0) return;
        CUDA_CHECK(cudaMemcpy(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void upload(const std::vector<T>& host) {
        upload(host.data(), host.size());
    }

    void download(T* host, std::size_t count) const {
        if (count == 0) return;
        CUDA_CHECK(cudaMemcpy(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void download(std::vector<T>& host) const {
        host.resize(count_);
        download(host.data(), count_);
    }

    T*          data()        { return ptr_; }
    const T*    data()  const { return ptr_; }
    std::size_t count() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
    bool        empty() const { return count_ == 0; }

private:
    void free() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    T*          ptr_   = nullptr;
    std::size_t count_ = 0;
};

}
