R"(
/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

// No reason to include these beyond version 1.2, where double is not an extension.
#if defined(DOUBLE) && __OPENCL_VERSION__ < CL_VERSION_1_2
  #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
  #else
    #error "Double precision floating point not supported by OpenCL implementation."
  #endif
#endif

#ifndef VALUE_TYPE
#error "VALUE_TYPE undefined!"
#endif

#ifndef SIZE_TYPE
#error "SIZE_TYPE undefined!"
#endif

#ifndef WG_SIZE
#error "WG_SIZE undefined!"
#endif

#ifndef REDUCE_BLOCK_SIZE
#error "REDUCE_BLOCK_SIZE undefined!"
#endif

#ifndef N_THREADS
#error "N_THREADS undefined!"
#endif
)"

R"(
VALUE_TYPE operation(VALUE_TYPE A, VALUE_TYPE B)
{
#ifdef OP_RO_PLUS
    return A + B;
#elif OP_RO_SQR
    return A + (B*B);
#elif OP_RO_FABS
    return A + fabs(B);
#else
    return A;
#endif
}
)"

R"(
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
__kernel
void reduce(const SIZE_TYPE size,
          __global const VALUE_TYPE* pX,
          __global VALUE_TYPE* pSum)
{
    __local VALUE_TYPE buf_tmp[REDUCE_BLOCK_SIZE];

    SIZE_TYPE idx = get_global_id(0);

    SIZE_TYPE block_idx = idx / REDUCE_BLOCK_SIZE;
    SIZE_TYPE thread_in_block_idx = idx % REDUCE_BLOCK_SIZE;

    SIZE_TYPE eidx = idx;

    VALUE_TYPE sum = 0;
    while(eidx < size)
    {
        sum = operation(sum, pX[eidx]);
        eidx += N_THREADS;
    }

    buf_tmp[thread_in_block_idx] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Seqential part
    if (get_local_id(0) == 0)
    {
        sum = 0.0;
        for (uint i = 0; i < REDUCE_BLOCK_SIZE; i++)
        {
            sum += buf_tmp[i];
        }

        pSum[ block_idx ] = sum;
    }
}
)"
