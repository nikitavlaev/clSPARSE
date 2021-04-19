#include <iostream>
#include <inttypes.h>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION
#include <CL/cl2.hpp>

#include "clSPARSE.h"
#include "clSPARSE-error.h"

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#define GROUPSIZE_256 256
#define WG_SIZE 64
#define TUPLE_QUEUE 6
#define NUM_SEGMENTS 128
//#define WARPSIZE_NV_2HEAP 64
#define value_type float
#define index_type int
#define MERGEPATH_LOCAL     0
#define MERGEPATH_LOCAL_L2  1
#define MERGEPATH_GLOBAL    2
#define MERGELIST_INITSIZE 256
#define BHSPARSE_SUCCESS 0

int run_bool_scan(
    cl_mem csrRowPtrA,
    cl_mem csrColIndA,
    cl_mem csrRowPtrB,
    cl_mem csrColIndB,
    cl_mem csrRowPtrCt_d,
    uint32_t &total_sum,
    int m,
    cl::Context context,
    const clsparseControl control)
{
    cl_int cl_status;

    uint array_size = m + 1;
    uint work_group_size = uint32_t(256);  
    uint block_size = work_group_size;
    uint32_t a_size = (array_size + block_size - 1) / block_size; // max to save first roots
    uint32_t b_size = (a_size + block_size - 1) / block_size; // max to save second roots
    // std::cout << "sizes " << array_size << ' ' << a_size << ' ' << b_size << std::endl;

    cl_mem a_gpu = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, sizeof(uint32_t) * a_size, NULL, &cl_status);
    cl_mem b_gpu = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, sizeof(uint32_t) * b_size, NULL, &cl_status);
    cl_mem total_sum_gpu = ::clCreateBuffer(context(), CL_MEM_READ_WRITE, sizeof(uint32_t), NULL, &cl_status);

    cl::LocalSpaceArg local_array = cl::Local(sizeof(uint32_t) * block_size);
    
    cl::Kernel kernel_scan = KernelCache::get(control->queue, "bool_csradd_scan", "scan_blelloch");
    cl::Kernel kernel_update = KernelCache::get(control->queue, "bool_csradd_scan", "update_pref_sum");

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = work_group_size;
    szGlobalWorkSize[0] = (array_size + work_group_size - 1) / work_group_size * work_group_size;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    KernelWrap kWrapper_scan(kernel_scan);
    KernelWrap kWrapper_update(kernel_update);

    uint leaf_size = 1;

    kWrapper_scan << a_gpu << csrRowPtrCt_d << local_array << total_sum_gpu << array_size;

    cl_status = kWrapper_scan.run(control, global, local);

    if (cl_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }


    uint32_t outer = (array_size + block_size - 1) / block_size;

    cl_mem *a_gpu_ptr = &a_gpu;
    cl_mem *b_gpu_ptr = &b_gpu;

    unsigned int *a_size_ptr = &a_size;
    unsigned int *b_size_ptr = &b_size;

    clEnqueueReadBuffer(control->queue(), total_sum_gpu, CL_TRUE, 0, sizeof(uint32_t), &total_sum, 0, NULL, NULL);

    // std::cout << "INNER TOTAL SUM: " << total_sum << " OUTER: " << outer << std::endl;

    while (outer > 1) {
        leaf_size *= block_size;

        // std::cout << "META: " << (outer + work_group_size - 1) / work_group_size * work_group_size << std::endl;
        size_t rec_szLocalWorkSize[1];
        size_t rec_szGlobalWorkSize[1];

        rec_szLocalWorkSize[0]  = work_group_size;
        rec_szGlobalWorkSize[0] = (outer + work_group_size - 1) / work_group_size * work_group_size;

        cl::NDRange rec_local(rec_szLocalWorkSize[0]);
        cl::NDRange rec_global(rec_szGlobalWorkSize[0]);

        // std::cout << "scan " << std::endl;

        kWrapper_scan.reset();
        kWrapper_scan << *b_gpu_ptr << *a_gpu_ptr << local_array << total_sum_gpu << outer;

        cl_status = kWrapper_scan.run(control, rec_global, rec_local);

        if (cl_status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        // std::cout << "update " << std::endl;

        kWrapper_update.reset();
        kWrapper_update << csrRowPtrCt_d << *a_gpu_ptr << array_size << leaf_size;

        cl_status = kWrapper_update.run(control, global, local);

        if (cl_status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        outer = (outer + block_size - 1) / block_size;
        std::swap(a_gpu_ptr, b_gpu_ptr);
        std::swap(a_size_ptr, b_size_ptr);
    }
    clEnqueueReadBuffer(control->queue(), total_sum_gpu, CL_TRUE, 0, sizeof(uint32_t), &total_sum, 0, NULL, NULL);

    return clsparseSuccess;
}

int run_bool_merge_count(
    cl_mem csrRowPtrA,
    cl_mem csrColIndA,
    cl_mem csrRowPtrB,
    cl_mem csrColIndB,
    cl_mem csrRowPtrCt_d,
    int m,
    const clsparseControl control
)
{
    cl::Kernel kernel = KernelCache::get(control->queue, "bool_csradd_merge", "merge_count");

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = WG_SIZE;
    size_t num_blocks = m;

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrCt_d;

    cl_int cl_status_2 = kWrapper.run(control, global, local);

    if (cl_status_2 != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

int run_bool_merge_fill(
    cl_mem csrRowPtrA,
    cl_mem csrColIndA,
    cl_mem csrRowPtrB,
    cl_mem csrColIndB,
    cl_mem csrRowPtrCt_d,
    cl_mem csrColIndC,
    int m,
    int total_sum,
    const clsparseControl control
)
{
    cl::Kernel kernel = KernelCache::get(control->queue, "bool_csradd_merge", "merge_fill");

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = WG_SIZE;
    szGlobalWorkSize[0] = m * WG_SIZE;

    printf("local %d global %d \n", szLocalWorkSize[0], szGlobalWorkSize[0]);

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrColIndB << csrRowPtrCt_d << csrColIndC;

    cl_int cl_status_2 = kWrapper.run(control, global, local);

    if (cl_status_2 != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

 CLSPARSE_EXPORT clsparseStatus
        clsparseBoolScsrElemAdd(
        const clsparseBoolCsrMatrix* sparseMatA,
        const clsparseBoolCsrMatrix* sparseMatB,
              clsparseBoolCsrMatrix* sparseMatC,
        const clsparseControl control )
{
    cl_int cl_status;

    if (!clsparseInitialized)
    {
       return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
       return clsparseInvalidControlObject;
    }

    const clsparseBoolCsrMatrixPrivate* A = static_cast<const clsparseBoolCsrMatrixPrivate*>(sparseMatA);
    const clsparseBoolCsrMatrixPrivate* B = static_cast<const clsparseBoolCsrMatrixPrivate*>(sparseMatB);
    clsparseBoolCsrMatrixPrivate* C = static_cast<clsparseBoolCsrMatrixPrivate*>(sparseMatC);

    // outer init
    cl_mem csrRowPtrA = A->row_pointer;
    cl_mem csrColIndA = A->col_indices;

    //int is important here, since kernel receives only bootstrapint, not size_t
    int m = A->num_rows;
    int k1 = A->num_cols;
    int k2 = B->num_rows;
    int n  = B->num_cols;
    int nnzA = A->num_nonzeros;
    int nnzB = B->num_nonzeros;

    if(k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl;
        return clsparseInvalidKernelExecution;
    }

    cl_mem csrRowPtrB = B->row_pointer;
    cl_mem csrColIndB = B->col_indices;

    int pattern = 0;

    cl::Context cxt = control->getContext();

    cl_mem csrRowPtrCt_d = ::clCreateBuffer(cxt(), CL_MEM_READ_WRITE, (m + 1) * sizeof( cl_int ), NULL, &cl_status );
    std::vector<int> csrRowPtrCt_h(m + 1, 0);

    clEnqueueFillBuffer(control->queue(), csrRowPtrCt_d, &pattern, sizeof(cl_int), 0, (m + 1)*sizeof(cl_int), 0, NULL, NULL);

    //std::cout << "mergecount " << std::endl;
    run_bool_merge_count(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrCt_d, m, control);

    int run_status = clEnqueueReadBuffer(control->queue(),
                                     csrRowPtrCt_d,
                                     1,
                                     0,
                                     (m + 1)*sizeof(cl_int),
                                     csrRowPtrCt_h.data(),
                                     0,
                                     0,
                                     0);
    // for (auto i = csrRowPtrCt_h.begin(); i != csrRowPtrCt_h.end(); ++i)
    // {
    //     std::cout << *i << ' '; 
    // }
    // std::cout << std::endl;

    uint32_t total_sum = 0;
    // std::cout << "scan " << std::endl;
    run_bool_scan(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrCt_d, total_sum, m, cxt, control);
    // std::cout << "TOTAL " << total_sum << std::endl;
    run_status = clEnqueueReadBuffer(control->queue(),
                                     csrRowPtrCt_d,
                                     1,
                                     0,
                                     (m + 1)*sizeof(cl_int),
                                     csrRowPtrCt_h.data(),
                                     0,
                                     0,
                                     0);
    // for (auto i = csrRowPtrCt_h.begin(); i != csrRowPtrCt_h.end(); ++i)
    // {
    //     std::cout << *i << ' '; 
    // }
    // std::cout << std::endl;

    // std::cout << "mergefill " << std::endl;
    std::vector<int> csrColIndC_h(total_sum, 0);
    cl_mem csrColIndC = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, total_sum * sizeof( cl_int ), NULL, &cl_status );

    run_bool_merge_fill(csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrCt_d, csrColIndC, m, total_sum, control);

    run_status = clEnqueueReadBuffer(control->queue(),
                                     csrColIndC,
                                     1,
                                     0,
                                     total_sum*sizeof(cl_int),
                                     csrColIndC_h.data(),
                                     0,
                                     0,
                                     0);

    C->num_rows = m;
    C->num_cols = n;
    C->num_nonzeros = total_sum;
    C->row_pointer = csrRowPtrCt_d;
    C->col_indices = csrColIndC;
    // for (auto i = csrColIndC_h.begin(); i != csrColIndC_h.end(); ++i)
    // {
    //     std::cout << *i << ' '; 
    // }
    // std::cout << std::endl;

    return clsparseSuccess;
}