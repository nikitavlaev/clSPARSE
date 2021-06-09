/* ************************************************************************
* The MIT License (MIT)
* Copyright 2014-2015 University of Copenhagen
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:

*  The above copyright notice and this permission notice shall be included in
*  all copies or substantial portions of the Software.

*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*  THE SOFTWARE.
* ************************************************************************ */

/* ************************************************************************
*  < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
*
*  < See papers:
*  1. Weifeng Liu and Brian Vinter, "A Framework for General Sparse
*      Matrix-Matrix Multiplication on GPUs and Heterogeneous
*      Processors," Journal of Parallel and Distributed Computing, 2015.
*  2. Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
*      Matrix-Matrix Multiplication for Irregular Data," Parallel and
*      Distributed Processing Symposium, 2014 IEEE 28th International
*      (IPDPS '14), pp.370-381, 19-23 May 2014.
*  for details. >
* ************************************************************************ */

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include <cmath>

#define GROUPSIZE_256 256
#define TUPLE_QUEUE 6
#define NUM_SEGMENTS 128
//#define WARPSIZE_NV_2HEAP 64
#define value_type float
#define index_type int
#define MERGEPATH_LOCAL 0
#define MERGEPATH_LOCAL_L2 1
#define MERGEPATH_GLOBAL 2
#define MERGELIST_INITSIZE 256
#define BHSPARSE_SUCCESS 0

#define copyCT2C_kernels "bool_copyCt2C_kernels"
#define computeNnzCt_kernels "SpGEMM_computeNnzCt_kernels"
#define ESC_0_1_kernels "bool_ESC_0_1_kernels"
#define ESC_2heap_kernels "bool_ESC_2heap_kernels"
#define ESC_bitonic_kernels "bool_ESC_bitonic_kernels"
#define ESC_2heap_kernels "bool_ESC_2heap_kernels"
#define EM_kernels "bool_EM_kernels"

using namespace std;

// use statictics from the dense implementation
int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m);

clsparseStatus bool_compute_nnzCt(int _m, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowPtrCt, clsparseControl control)
{

    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, computeNnzCt_kernels, "compute_nnzCt_kernel", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    size_t num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrRowPtrCt << _m;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position, cl_mem queue_one, cl_mem csrRowPtrC, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, ESC_0_1_kernels, "ESC_0", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrC << counter << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position, cl_mem queue_one,
                                      cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                                      cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, ESC_0_1_kernels, "ESC_1", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrC << csrRowPtrCt << csrColIndCt << counter << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_compute_nnzC_Ct_2heap_noncoalesced_local(int num_threads, int num_blocks, int j, int counter, int position,
                                                             cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA,
                                                             cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowPtrC,
                                                             cl_mem csrRowPtrCt, cl_mem csrColIndCt, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    cl::Kernel kernel = KernelCache::get(control->queue, ESC_2heap_kernels, "ESC_2heap_noncoalesced_local", params);

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrC
             << csrRowPtrCt << csrColIndCt << cl::Local(j * num_threads * sizeof(int)) << counter << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_compute_nnzC_Ct_bitonic_scan(int num_threads, int num_blocks, int j, int position, cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB,
                                                 cl_mem csrColIndB, cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, int _n, clsparseControl control)
{

    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, ESC_bitonic_kernels, "ESC_bitonic_scan", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    int buffer_size = 2 * num_threads;

    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrC << csrRowPtrCt
             << csrColIndCt << cl::Local(buffer_size * sizeof(int)) << cl::Local((buffer_size + 1) * sizeof(short)) << position << _n;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j, int mergebuffer_size, int position, int *count_next, int mergepath_location,
                                              cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB,
                                              cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem *csrColIndCt, int *_nnzCt, int m, int *_h_queue_one, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel1 = KernelCache::get(control->queue, EM_kernels, "EM_mergepath", params);
    cl::Kernel kernel2 = KernelCache::get(control->queue, EM_kernels, "EM_mergepath_global", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status;

    if (mergepath_location == MERGEPATH_LOCAL)
    {
        KernelWrap kWrapper1(kernel1);
        kWrapper1 << queue_one << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrC
                  << csrRowPtrCt << *csrColIndCt << cl::Local((mergebuffer_size) * sizeof(int)) << cl::Local((num_threads + 1) * sizeof(short)) << position << mergebuffer_size << cl::Local(sizeof(cl_int) * (num_threads + 1)) << cl::Local(sizeof(cl_int) * (num_threads + 1));

        status = kWrapper1.run(control, global, local);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }
    else if (mergepath_location == MERGEPATH_GLOBAL)
    {
        int mergebuffer_size_local = 2304;

        KernelWrap kWrapper2(kernel2);
        kWrapper2 << queue_one << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrC
                  << csrRowPtrCt << *csrColIndCt << cl::Local((mergebuffer_size_local) * sizeof(int)) << cl::Local((num_threads + 1) * sizeof(short)) << position << mergebuffer_size_local << cl::Local(sizeof(cl_int) * (num_threads + 1)) << cl::Local(sizeof(cl_int) * (num_threads + 1));

        status = kWrapper2.run(control, global, local);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    int temp_queue[6] = {0, 0, 0, 0, 0, 0};
    int counter = 0;
    int temp_num = 0;

    status = clEnqueueReadBuffer(control->queue(),
                                 queue_one,
                                 1,
                                 0,
                                 TUPLE_QUEUE * m * sizeof(int),
                                 _h_queue_one,
                                 0,
                                 0,
                                 0);

    for (int i = position; i < position + num_blocks; i++)
    {
        if (_h_queue_one[TUPLE_QUEUE * i + 2] != -1)
        {
            temp_queue[0] = _h_queue_one[TUPLE_QUEUE * i]; // row id
            if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
            {
                int accum = 0;
                switch (mergebuffer_size)
                {
                case 256:
                    accum = 512;
                    break;
                case 512:
                    accum = 1024;
                    break;
                case 1024:
                    accum = 2048;
                    break;
                case 2048:
                    accum = 2304;
                    break;
                case 2304:
                    accum = 2 * (2304 * 2);
                    break;
                }

                temp_queue[1] = *_nnzCt + counter * accum; // new start address
            }
            else if (mergepath_location == MERGEPATH_GLOBAL)
                temp_queue[1] = *_nnzCt + counter * (2 * (mergebuffer_size + 2304));
            temp_queue[2] = _h_queue_one[TUPLE_QUEUE * i + 2]; // merged size
            temp_queue[3] = _h_queue_one[TUPLE_QUEUE * i + 3]; // i
            temp_queue[4] = _h_queue_one[TUPLE_QUEUE * i + 4]; // k
            temp_queue[5] = _h_queue_one[TUPLE_QUEUE * i + 1]; // old start address

            _h_queue_one[TUPLE_QUEUE * i] = _h_queue_one[TUPLE_QUEUE * (position + counter)];         // row id
            _h_queue_one[TUPLE_QUEUE * i + 1] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 1]; // new start address
            _h_queue_one[TUPLE_QUEUE * i + 2] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 2]; // merged size
            _h_queue_one[TUPLE_QUEUE * i + 3] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 3]; // i
            _h_queue_one[TUPLE_QUEUE * i + 4] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 4]; // k
            _h_queue_one[TUPLE_QUEUE * i + 5] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 5]; // old start address

            _h_queue_one[TUPLE_QUEUE * (position + counter)] = temp_queue[0];     // row id
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 1] = temp_queue[1]; // new start address
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 2] = temp_queue[2]; // merged size
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 3] = temp_queue[3]; // i
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 4] = temp_queue[4]; // k
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 5] = temp_queue[5]; // old start address

            counter++;
            temp_num += _h_queue_one[TUPLE_QUEUE * i + 2];
        }
    }

    status = clEnqueueWriteBuffer(control->queue(),
                                  queue_one,
                                  1,
                                  0,
                                  TUPLE_QUEUE * m * sizeof(int),
                                  _h_queue_one,
                                  0,
                                  0,
                                  0);

    //*
    if (counter > 0)
    {
        int nnzCt_new;
        if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
        {
            int accum = 0;
            switch (mergebuffer_size)
            {
            case 256:
                accum = 512;
                break;
            case 512:
                accum = 1024;
                break;
            case 1024:
                accum = 2048;
                break;
            case 2048:
                accum = 2304;
                break;
            case 2304:
                accum = 2 * (2304 * 2);
                break;
            }

            nnzCt_new = *_nnzCt + counter * accum;
        }
        else if (mergepath_location == MERGEPATH_GLOBAL)
            nnzCt_new = *_nnzCt + counter * (2 * (mergebuffer_size + 2304));

        cl_mem csrColIndCt_new = ::clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE, nnzCt_new * sizeof(cl_int), NULL, NULL);

        clEnqueueCopyBuffer(control->queue(),
                            *csrColIndCt,
                            csrColIndCt_new,
                            0,
                            0,
                            sizeof(cl_int) * (*_nnzCt),
                            0,
                            NULL,
                            NULL);

        clReleaseMemObject(*csrColIndCt);

        *csrColIndCt = csrColIndCt_new;

        *_nnzCt = nnzCt_new;
    }

    *count_next = counter;

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_compute_nnzC_Ct_opencl(int *_h_counter_one, cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem *csrColIndCt, int _n, int _nnzCt, int m, int *queue_one_h, clsparseControl control)
{
    //int err = 0;
    int counter = 0;

    clsparseStatus run_status;

    for (int j = 0; j < NUM_SEGMENTS; j++)
    {
        counter = _h_counter_one[j + 1] - _h_counter_one[j];
        if (counter != 0)
        {

            if (j == 0)
            {
                int num_threads = GROUPSIZE_256;
                size_t num_blocks = ceil((double)counter / (double)num_threads);

                run_status = bool_compute_nnzC_Ct_0(num_threads, num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrC, control);
            }
            else if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                size_t num_blocks = ceil((double)counter / (double)num_threads);

                run_status = bool_compute_nnzC_Ct_1(num_threads, num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, control);
            }
            else if (j > 1 && j <= 32)
            {
                int num_threads = 64; //WARPSIZE_NV_2HEAP;
                size_t num_blocks = ceil((double)counter / (double)num_threads);
                run_status = bool_compute_nnzC_Ct_2heap_noncoalesced_local(num_threads, num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, control);
            }
            else if (j > 32 && j <= 64)
            {
                int num_threads = 32;
                int num_blocks = counter;

                run_status = bool_compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, _n, control);
            }
            else if (j > 64 && j <= 122)
            {
                int num_threads = 64;
                int num_blocks = counter;

                run_status = bool_compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, _n, control);
            }
            else if (j == 123)
            {
                int num_threads = 128;
                int num_blocks = counter;

                run_status = bool_compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, _n, control);
            }
            else if (j == 124)
            {
                int num_threads = 256;
                int num_blocks = counter;

                run_status = bool_compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, _n, control);
            }
            else if (j == 127)
            {
                int count_next = counter;
                int num_threads, num_blocks, mergebuffer_size;

                int num_threads_queue[5] = {64, 128, 256, 256, 256};
                int mergebuffer_size_queue[5] = {256, 512, 1024, 2048, 2304}; //{256, 464, 924, 1888, 3840};

                int queue_counter = 0;

                while (count_next > 0)
                {
                    num_blocks = count_next;

                    if (queue_counter < 5)
                    {
                        num_threads = num_threads_queue[queue_counter];
                        mergebuffer_size = mergebuffer_size_queue[queue_counter];

                        run_status = bool_compute_nnzC_Ct_mergepath(num_threads, num_blocks, j, mergebuffer_size, _h_counter_one[j], &count_next, MERGEPATH_LOCAL, queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, csrColIndCt, &_nnzCt, m, queue_one_h, control);

                        queue_counter++;
                    }
                    else
                    {
                        num_threads = num_threads_queue[4];
                        mergebuffer_size += mergebuffer_size_queue[4];

                        run_status = bool_compute_nnzC_Ct_mergepath(num_threads, num_blocks, j, mergebuffer_size, _h_counter_one[j], &count_next, MERGEPATH_GLOBAL, queue_one, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt, csrColIndCt, &_nnzCt, m, queue_one_h, control);
                    }
                }
            }

            if (run_status != clsparseSuccess)
            {
                return clsparseInvalidKernelExecution;
            }
        }
    }

    return clsparseSuccess;
}

clsparseStatus bool_copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position,
                                        cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{

    int j = 1;

    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, copyCT2C_kernels, "copyCt2C_Single", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrC << csrColIndC << csrRowPtrCt << csrColIndCt << queue_one << local_size << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position,
                                          cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{
    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, copyCT2C_kernels, "copyCt2C_Loopless", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrC << csrColIndC << csrRowPtrCt << csrColIndCt << queue_one << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus bool_copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position,
                                      cl_mem csrRowPtrC, cl_mem csrColIndC,
                                      cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{

    const std::string params = std::string() +
                               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type + " -DVALUE_TYPE=" + OclTypeTraits<cl_int>::type;

    cl::Kernel kernel = KernelCache::get(control->queue, copyCT2C_kernels, "copyCt2C_Loop", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0] = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrC << csrColIndC << csrRowPtrCt << csrColIndCt << queue_one << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

int bool_copy_Ct_to_C_opencl(int *counter_one, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{
    int counter = 0;

    clsparseStatus run_status;

    for (int j = 1; j < NUM_SEGMENTS; j++)
    {
        counter = counter_one[j + 1] - counter_one[j];
        if (counter != 0)
        {
            if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                size_t num_blocks = ceil((double)counter / (double)num_threads);
                run_status = bool_copy_Ct_to_C_Single(num_threads, num_blocks, counter, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            }
            else if (j > 1 && j <= 32)
                run_status = bool_copy_Ct_to_C_Loopless(32, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j > 32 && j <= 64)
                run_status = bool_copy_Ct_to_C_Loopless(64, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j > 63 && j <= 96)
                run_status = bool_copy_Ct_to_C_Loopless(96, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j > 96 && j <= 122)
                run_status = bool_copy_Ct_to_C_Loopless(128, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 123)
                run_status = bool_copy_Ct_to_C_Loopless(256, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 124)
                run_status = bool_copy_Ct_to_C_Loop(256, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 127)
                run_status = bool_copy_Ct_to_C_Loop(256, counter, j, counter_one[j], csrRowPtrC, csrColIndC, csrRowPtrCt, csrColIndCt, queue_one, control);

            if (run_status != CL_SUCCESS)
            {
                return clsparseInvalidKernelExecution;
            }
        }
    }

    return clsparseSuccess;
}

CLSPARSE_EXPORT clsparseStatus
clsparseBoolScsrSpGemm(
    const clsparseBoolCsrMatrix *sparseMatA,
    const clsparseBoolCsrMatrix *sparseMatB,
    clsparseBoolCsrMatrix *sparseMatC,
    const clsparseControl control)
{
    cl_int run_status;

    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    const clsparseBoolCsrMatrixPrivate *matA = static_cast<const clsparseBoolCsrMatrixPrivate *>(sparseMatA);
    const clsparseBoolCsrMatrixPrivate *matB = static_cast<const clsparseBoolCsrMatrixPrivate *>(sparseMatB);
    clsparseBoolCsrMatrixPrivate *matC = static_cast<clsparseBoolCsrMatrixPrivate *>(sparseMatC);

    size_t m = matA->num_rows;
    size_t k1 = matA->num_cols;
    size_t k2 = matB->num_rows;
    size_t n = matB->num_cols;
    size_t nnzA = matA->num_nonzeros;
    size_t nnzB = matB->num_nonzeros;

    if (k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl;
        return clsparseInvalidKernelExecution;
    }

    cl_mem csrRowPtrA = matA->row_pointer;
    cl_mem csrColIndA = matA->col_indices;
    cl_mem csrRowPtrB = matB->row_pointer;
    cl_mem csrColIndB = matB->col_indices;

    matC->row_pointer = ::clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE, (m + 1) * sizeof(cl_int), NULL, &run_status);

    int pattern = 0;
    clEnqueueFillBuffer(control->queue(), matC->row_pointer, &pattern, sizeof(cl_int), 0, (m + 1) * sizeof(cl_int), 0, NULL, NULL);

    cl_mem csrRowPtrC = matC->row_pointer;

    std::vector<int> csrRowPtrC_h(m + 1, 0);

    cl_mem csrRowPtrCt_d = ::clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE, (m + 1) * sizeof(cl_int), NULL, &run_status);
    clEnqueueFillBuffer(control->queue(), csrRowPtrCt_d, &pattern, sizeof(cl_int), 0, (m + 1) * sizeof(cl_int), 0, NULL, NULL);

    std::vector<int> csrRowPtrCt_h(m + 1, 0);

    // STAGE 1
    bool_compute_nnzCt(m, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrCt_d, control);

    // statistics
    std::vector<int> counter(NUM_SEGMENTS, 0);

    std::vector<int> counter_one(NUM_SEGMENTS + 1, 0);

    std::vector<int> counter_sum(NUM_SEGMENTS + 1, 0);

    std::vector<int> queue_one(m * TUPLE_QUEUE, 0);

    cl_mem queue_one_d = ::clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE, TUPLE_QUEUE * m * sizeof(int), NULL, &run_status);

    run_status = clEnqueueReadBuffer(control->queue(),
                                     csrRowPtrCt_d,
                                     1,
                                     0,
                                     (m + 1) * sizeof(cl_int),
                                     csrRowPtrCt_h.data(),
                                     0,
                                     0,
                                     0);

    // STAGE 2 - STEP 1 : statistics
    int nnzCt = statistics(csrRowPtrCt_h.data(), counter.data(), counter_one.data(), counter_sum.data(), queue_one.data(), m);
    // STAGE 2 - STEP 2 : create Ct

    cl_mem csrColIndCt = ::clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE, nnzCt * sizeof(cl_int), NULL, &run_status);

    //copy queue_one
    run_status = clEnqueueWriteBuffer(control->queue(),
                                      queue_one_d,
                                      1,
                                      0,
                                      TUPLE_QUEUE * m * sizeof(int),
                                      queue_one.data(),
                                      0,
                                      0,
                                      0);

    // STAGE 3 - STEP 1 : compute nnzC and Ct
    bool_compute_nnzC_Ct_opencl(counter_one.data(), queue_one_d, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrC, csrRowPtrCt_d, &csrColIndCt, n, nnzCt, m, queue_one.data(), control);
    // STAGE 3 - STEP 2 : malloc C on devices
    run_status = clEnqueueReadBuffer(control->queue(),
                                     csrRowPtrC,
                                     1,
                                     0,
                                     (m + 1) * sizeof(cl_int),
                                     csrRowPtrC_h.data(),
                                     0,
                                     0,
                                     0);

    int old_val, new_val;
    old_val = csrRowPtrC_h[0];
    csrRowPtrC_h[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrC_h[i];
        csrRowPtrC_h[i] = old_val + csrRowPtrC_h[i - 1];
        old_val = new_val;
    }

    int nnzC = csrRowPtrC_h[m];

    matC->col_indices = ::clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE, nnzC * sizeof(cl_int), NULL, &run_status);

    cl_mem csrColIndC = matC->col_indices;

    run_status = clEnqueueWriteBuffer(control->queue(),
                                      csrRowPtrC,
                                      1,
                                      0,
                                      (m + 1) * sizeof(cl_int),
                                      csrRowPtrC_h.data(),
                                      0,
                                      0,
                                      0);

    bool_copy_Ct_to_C_opencl(counter_one.data(), csrRowPtrC, csrColIndC, csrRowPtrCt_d, csrColIndCt, queue_one_d, control);

    matC->num_rows = m;
    matC->num_cols = n;
    matC->num_nonzeros = nnzC;

    ::clReleaseMemObject(csrRowPtrCt_d);
    ::clReleaseMemObject(queue_one_d);
    ::clReleaseMemObject(csrColIndCt);
    return clsparseSuccess;
}
