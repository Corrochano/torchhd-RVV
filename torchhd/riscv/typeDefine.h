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
#pragma once

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdint>

#include <riscv_vector.h>

const size_t ALIGNMENT = 64; 

struct benchmarkResult{
    double seconds;
    double gbs;
    double sseconds;
    double sgbs;
    double speedup;
};

typedef uint64_t hdc_word_t;
typedef uint64_t hdc_score_t;

typedef uint32_t hdc_word_t_32;
typedef uint32_t hdc_score_t_32;

typedef uint16_t hdc_word_t_16;
typedef uint16_t hdc_score_t_16;

typedef uint8_t hdc_word_t_8;
typedef uint8_t hdc_score_t_8;

inline size_t get_rvv_vl(size_t avl) {
      return __riscv_vsetvl_e64m1(avl);
}                                   
