/*
Copyright 2026 Álvaro Corrochano López

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <torch/extension.h>
#include "vectorialKernels.h"

torch::Tensor rvv_bind_tensor(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu(), "Inputs must be CPU tensors");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "Inputs must have matching dtypes");
    TORCH_CHECK(x.sizes() == y.sizes(), "Inputs must have identical shapes");

    auto x_c = x.contiguous();
    auto y_c = y.contiguous();
    auto z = torch::empty_like(x_c);
    size_t words = x_c.numel();

    AT_DISPATCH_INTEGRAL_TYPES(x_c.scalar_type(), "hdc_bind", ([&] {
        if constexpr (sizeof(scalar_t) == 8) {
            hdc_bind(reinterpret_cast<const hdc_word_t*>(x_c.data_ptr<scalar_t>()),
                     reinterpret_cast<const hdc_word_t*>(y_c.data_ptr<scalar_t>()),
                     reinterpret_cast<hdc_word_t*>(z.data_ptr<scalar_t>()), words);
        } else if constexpr (sizeof(scalar_t) == 4) {
            hdc_bind(reinterpret_cast<const hdc_word_t_32*>(x_c.data_ptr<scalar_t>()),
                     reinterpret_cast<const hdc_word_t_32*>(y_c.data_ptr<scalar_t>()),
                     reinterpret_cast<hdc_word_t_32*>(z.data_ptr<scalar_t>()), words);
        } else if constexpr (sizeof(scalar_t) == 2) {
            hdc_bind(reinterpret_cast<const hdc_word_t_16*>(x_c.data_ptr<scalar_t>()),
                     reinterpret_cast<const hdc_word_t_16*>(y_c.data_ptr<scalar_t>()),
                     reinterpret_cast<hdc_word_t_16*>(z.data_ptr<scalar_t>()), words);
        } else if constexpr (sizeof(scalar_t) == 1) {
            hdc_bind(reinterpret_cast<const hdc_word_t_8*>(x_c.data_ptr<scalar_t>()),
                     reinterpret_cast<const hdc_word_t_8*>(y_c.data_ptr<scalar_t>()),
                     reinterpret_cast<hdc_word_t_8*>(z.data_ptr<scalar_t>()), words);
        }
    }));

    return z;
}

torch::Tensor rvv_hamming_tensor(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu(), "Inputs must be CPU tensors");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "Inputs must have matching dtypes");
    TORCH_CHECK(x.sizes() == y.sizes(), "Inputs must have identical shapes");

    auto x_c = x.contiguous();
    auto y_c = y.contiguous();
    size_t words = x_c.numel();
    size_t alignment = 64;

    int64_t score = 0;

    AT_DISPATCH_INTEGRAL_TYPES(x_c.scalar_type(), "hdc_hamming", ([&] {
        size_t alloc_size = ((words * sizeof(scalar_t) + alignment - 1) / alignment) * alignment;

        if constexpr (sizeof(scalar_t) == 8) {
            hdc_hamming(reinterpret_cast<const hdc_word_t*>(x_c.data_ptr<scalar_t>()),
                        reinterpret_cast<const hdc_word_t*>(y_c.data_ptr<scalar_t>()),
                        reinterpret_cast<hdc_score_t*>(&score),
                        words, alignment, alloc_size);
        } else if constexpr (sizeof(scalar_t) == 4) {
            hdc_hamming(reinterpret_cast<const hdc_word_t_32*>(x_c.data_ptr<scalar_t>()),
                        reinterpret_cast<const hdc_word_t_32*>(y_c.data_ptr<scalar_t>()),
                        reinterpret_cast<hdc_score_t_32*>(&score),
                        words, alignment, alloc_size);
        } else if constexpr (sizeof(scalar_t) == 2) {
            hdc_hamming(reinterpret_cast<const hdc_word_t_16*>(x_c.data_ptr<scalar_t>()),
                        reinterpret_cast<const hdc_word_t_16*>(y_c.data_ptr<scalar_t>()),
                        reinterpret_cast<hdc_score_t_16*>(&score),
                        words, alignment, alloc_size);
        } else if constexpr (sizeof(scalar_t) == 1) {
            hdc_hamming(reinterpret_cast<const hdc_word_t_8*>(x_c.data_ptr<scalar_t>()),
                        reinterpret_cast<const hdc_word_t_8*>(y_c.data_ptr<scalar_t>()),
                        reinterpret_cast<hdc_score_t_8*>(&score),
                        words, alignment, alloc_size);
        }
    }));

    return torch::tensor(score, torch::kInt64);
}

torch::Tensor rvv_query_tensor(torch::Tensor M, torch::Tensor q) {
    TORCH_CHECK(M.device().is_cpu() && q.device().is_cpu(), "Inputs must be CPU tensors");
    TORCH_CHECK(M.scalar_type() == q.scalar_type(), "Inputs must have matching dtypes");
    TORCH_CHECK(M.dim() == 2, "Memory Matrix M must be 2D [nvec, words]");
    TORCH_CHECK(q.dim() == 1, "Query vector q must be 1D [words]");
    TORCH_CHECK(M.size(1) == q.size(0), "M dimension 1 must match query size");

    auto M_c = M.contiguous();
    auto q_c = q.contiguous();
    size_t nvec = M_c.size(0);
    size_t words = M_c.size(1);

    size_t alignment = 64;
    
    auto scores = torch::zeros({static_cast<int64_t>(nvec)}, torch::kInt64);

    AT_DISPATCH_INTEGRAL_TYPES(M_c.scalar_type(), "hdc_query", ([&] {
        size_t alloc_size = ((words * sizeof(scalar_t) + alignment - 1) / alignment) * alignment;
        
        if constexpr (sizeof(scalar_t) == 8) {
            hdc_query(reinterpret_cast<const hdc_word_t*>(M_c.data_ptr<scalar_t>()),
                      reinterpret_cast<const hdc_word_t*>(q_c.data_ptr<scalar_t>()),
                      reinterpret_cast<hdc_score_t*>(scores.data_ptr<int64_t>()),
                      nvec, words, alignment, alloc_size);
        } else if constexpr (sizeof(scalar_t) == 4) {
            hdc_query(reinterpret_cast<const hdc_word_t_32*>(M_c.data_ptr<scalar_t>()),
                      reinterpret_cast<const hdc_word_t_32*>(q_c.data_ptr<scalar_t>()),
                      reinterpret_cast<hdc_score_t_32*>(scores.data_ptr<int64_t>()),
                      nvec, words, alignment, alloc_size);
        } else if constexpr (sizeof(scalar_t) == 2) {
            hdc_query(reinterpret_cast<const hdc_word_t_16*>(M_c.data_ptr<scalar_t>()),
                      reinterpret_cast<const hdc_word_t_16*>(q_c.data_ptr<scalar_t>()),
                      reinterpret_cast<hdc_score_t_16*>(scores.data_ptr<int64_t>()),
                      nvec, words, alignment, alloc_size);
        } else if constexpr (sizeof(scalar_t) == 1) {
            hdc_query(reinterpret_cast<const hdc_word_t_8*>(M_c.data_ptr<scalar_t>()),
                      reinterpret_cast<const hdc_word_t_8*>(q_c.data_ptr<scalar_t>()),
                      reinterpret_cast<hdc_score_t_8*>(scores.data_ptr<int64_t>()),
                      nvec, words, alignment, alloc_size);
        }
    }));

    return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bind", &rvv_bind_tensor, "RVV Accelerated HDC Bind (XOR)");
    m.def("hamming", &rvv_hamming_tensor, "RVV Accelerated HDC Hamming Distance");
    m.def("query", &rvv_query_tensor, "RVV Accelerated HDC Query (Hamming Distance)");
}
