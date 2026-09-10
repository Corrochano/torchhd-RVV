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
#include "vectorialKernels.h"

using namespace std;

//////////////////
// Bind Section //
//////////////////

void hdc_bind(
    const hdc_word_t *x,
    const hdc_word_t *y,
    hdc_word_t *z,
    size_t words)
{   
    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        
        // Load as vectors
        vuint64m1_t vx = __riscv_vle64_v_u64m1(&x[i], vl);
        vuint64m1_t vy = __riscv_vle64_v_u64m1(&y[i], vl);

        // Execute xor
        vuint64m1_t vz = __riscv_vxor_vv_u64m1(vx,vy,vl);

        // Save data into z
        __riscv_vse64_v_u64m1(&z[i], vz, vl);

        // Advance words
        i += vl;
    }
}

void hdc_bind(
    const hdc_word_t_32 *x,
    const hdc_word_t_32 *y,
    hdc_word_t_32 *z,
    size_t words)
{   
    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        
        // Load as vectors
        vuint32m1_t vx = __riscv_vle32_v_u32m1(&x[i], vl);
        vuint32m1_t vy = __riscv_vle32_v_u32m1(&y[i], vl);

        // Execute xor
        vuint32m1_t vz = __riscv_vxor_vv_u32m1(vx,vy,vl);

        // Save data into z
        __riscv_vse32_v_u32m1(&z[i], vz, vl);

        // Advance words
        i += vl;
    }
}

void hdc_bind(
    const hdc_word_t_16 *x,
    const hdc_word_t_16 *y,
    hdc_word_t_16 *z,
    size_t words)
{   
    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        
        // Load as vectors
        vuint16m1_t vx = __riscv_vle16_v_u16m1(&x[i], vl);
        vuint16m1_t vy = __riscv_vle16_v_u16m1(&y[i], vl);

        // Execute xor
        vuint16m1_t vz = __riscv_vxor_vv_u16m1(vx,vy,vl);

        // Save data into z
        __riscv_vse16_v_u16m1(&z[i], vz, vl);

        // Advance words
        i += vl;
    }
}

void hdc_bind(
    const hdc_word_t_8 *x,
    const hdc_word_t_8 *y,
    hdc_word_t_8 *z,
    size_t words)
{   
    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        
        // Load as vectors
        vuint8m1_t vx = __riscv_vle8_v_u8m1(&x[i], vl);
        vuint8m1_t vy = __riscv_vle8_v_u8m1(&y[i], vl);

        // Execute xor
        vuint8m1_t vz = __riscv_vxor_vv_u8m1(vx,vy,vl);

        // Save data into z
        __riscv_vse8_v_u8m1(&z[i], vz, vl);

        // Advance words
        i += vl;
    }
}

/////////////////////
// Hamming Section //
/////////////////////

void hdc_hamming(
    const hdc_word_t *x,
    const hdc_word_t *y,
    hdc_score_t *acc,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    hdc_word_t *z = (hdc_word_t*)aligned_alloc(alignment, alloc_size);
    //vuint64m1_t vz = __riscv_vle64_v_u64m1(z, vl); // Need it as a vector

    hdc_bind(x,y,z,words); // Perform XOR

    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        vuint64m1_t vz = __riscv_vle64_v_u64m1(z, vl); // Need it as a vector    
        vbool64_t bz = __riscv_vmsne_vx_u64m1_b64(vz, 0, vl); // Need bool argument

        *acc += __riscv_vcpop_m_b64(bz, vl); // pop count

        i += vl;
    }

    free(z);
}

void hdc_hamming(
    const hdc_word_t_32 *x,
    const hdc_word_t_32 *y,
    hdc_score_t_32 *acc,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    hdc_word_t_32 *z = (hdc_word_t_32*)aligned_alloc(alignment, alloc_size);

    hdc_bind(x,y,z,words); // Perform XOR

    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        vuint32m1_t vz = __riscv_vle32_v_u32m1(z, vl); // Need it as a vector    
        vbool32_t bz = __riscv_vmsne_vx_u32m1_b32(vz, 0, vl); // Need bool argument

        *acc += __riscv_vcpop_m_b32(bz, vl); // pop count

        i += vl;
    }

    free(z);
}

void hdc_hamming(
    const hdc_word_t_16 *x,
    const hdc_word_t_16 *y,
    hdc_score_t_16 *acc,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    hdc_word_t_16 *z = (hdc_word_t_16*)aligned_alloc(alignment, alloc_size);

    hdc_bind(x,y,z,words); // Perform XOR

    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        vuint16m1_t vz = __riscv_vle16_v_u16m1(z, vl); // Need it as a vector    
        vbool16_t bz = __riscv_vmsne_vx_u16m1_b16(vz, 0, vl); // Need bool argument

        *acc += __riscv_vcpop_m_b16(bz, vl); // pop count

        i += vl;
    }

    free(z);
}

void hdc_hamming(
    const hdc_word_t_8 *x,
    const hdc_word_t_8 *y,
    hdc_score_t_8 *acc,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    hdc_word_t_8 *z = (hdc_word_t_8*)aligned_alloc(alignment, alloc_size);

    hdc_bind(x,y,z,words); // Perform XOR

    size_t i = 0;
    while (i < words){
        // Get VL
        size_t vl = get_rvv_vl(words - i);
        vuint8m1_t vz = __riscv_vle8_v_u8m1(z, vl); // Need it as a vector    
        vbool8_t bz = __riscv_vmsne_vx_u8m1_b8(vz, 0, vl); // Need bool argument

        *acc += __riscv_vcpop_m_b8(bz, vl); // pop count

        i += vl;
    }

    free(z);
}

///////////////////
// Query Section //
///////////////////

void hdc_query(
    const hdc_word_t *M,
    const hdc_word_t *q,
    hdc_score_t *scores,
    size_t nvec,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    size_t i = 0;
    while (i < nvec){
        hdc_hamming(&M[i * words], q, &scores[i], words, alignment, alloc_size);
        
        i++;
    }
}

void hdc_query(
    const hdc_word_t_32 *M,
    const hdc_word_t_32 *q,
    hdc_score_t_32 *scores,
    size_t nvec,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    size_t i = 0;
    while (i < nvec){
        hdc_hamming(&M[i * words], q, &scores[i], words, alignment, alloc_size);
        
        i++;
    }
}

void hdc_query(
    const hdc_word_t_16 *M,
    const hdc_word_t_16 *q,
    hdc_score_t_16 *scores,
    size_t nvec,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    size_t i = 0;
    while (i < nvec){
        hdc_hamming(&M[i * words], q, &scores[i], words, alignment, alloc_size);
        
        i++;
    }
}

void hdc_query(
    const hdc_word_t_8 *M,
    const hdc_word_t_8 *q,
    hdc_score_t_8 *scores,
    size_t nvec,
    size_t words,
    size_t alignment, 
    size_t alloc_size)
{
    size_t i = 0;
    while (i < nvec){
        hdc_hamming(&M[i * words], q, &scores[i], words, alignment, alloc_size);
        
        i++;
    }
}
