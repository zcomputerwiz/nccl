#pragma once
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

namespace c10d {

inline void checkForNan(const at::Tensor& tensor, c10::cuda::CUDAStream stream) {
    (void)tensor;
    (void)stream;
}

inline void checkForNan(const at::Tensor& tensor) {
    (void)tensor;
}

} // namespace c10d
