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

 /*! \file
 * \brief Simple demonstration code for how to calculate a SpM-dV (Sparse matrix
 * times dense Vector) multiply
 */

#include <iostream>
#include <vector>
#include <assert.h>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION
#include <CL/cl2.hpp>

#include "clSPARSE.h"
#include "clSPARSE-error.h"

/**
 * \brief Sample Sparse Matrix dense Vector multiplication (SPMV C++)
 *  \details [y = alpha * A*x + beta * y]
 *
 * A - [m x n] matrix in CSR format
 * x - dense vector of n elements
 * y - dense vector of m elements
 * alpha, beta - scalars
 *
 *
 * Program presents usage of clSPARSE library in csrmv (y = A*x) operation
 * where A is sparse matrix in CSR format, x, y are dense vectors.
 *
 * clSPARSE offers two spmv algorithms for matrix stored in CSR format.
 * First one is called vector algorithm, the second one is called adaptve.
 * Adaptive version is usually faster but for given matrix additional
 * structure (rowBlocks) needs to be calculated first.
 *
 * To calculate rowBlock structure you have to execute clsparseCsrMetaSize
 * for given matrix stored in CSR format. It is enough to calculate the
 * structure once, it is related to its nnz pattern.
 *
 * After the matrix is read from disk with the function
 * clsparse<S,D>CsrMatrixfromFile
 * the rowBlock structure can be calculated using clsparseCsrMetaCompute
 *
 * If rowBlocks are calculated the clsparseCsrMatrix.rowBlocks field is not null.
 *
 * Program is executing by completing following steps:
 * 1. Setup OpenCL environment
 * 2. Setup GPU buffers
 * 3. Init clSPARSE library
 * 4. Execute algorithm cldenseSaxpy
 * 5. Shutdown clSPARSE library & OpenCL
 *
 * usage:
 *
 * sample-spmv path/to/matrix/in/mtx/format.mtx
 *
 */

int readAndCreateBoolCsrMatrix(clsparseBoolCsrMatrix* A, std::string matrix_path, clsparseCreateResult createResult, cl::Context context)
{
    cl_int cl_status;
    clsparseInitBoolCsrMatrix(A);
    // Read matrix from file. Calculates the rowBlocks structures as well.
    clsparseIdx_t nnz, row, col;
    // read MM header to get the size of the matrix;
    clsparseStatus fileError
            = clsparseHeaderfromFile( &nnz, &row, &col, matrix_path.c_str( ) );

    if( fileError != clsparseSuccess )
    {
        std::cout << "Could not read matrix market header from disk" << std::endl;
        return -5;
    }

    A->num_nonzeros = nnz;
    A->num_rows = row;
    A->num_cols = col;

    // Allocate memory for CSR matrix

    A->col_indices = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                     A->num_nonzeros * sizeof( clsparseIdx_t ), NULL, &cl_status );

    A->row_pointer = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                     ( A->num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, &cl_status );


    // Read matrix market file with explicit zero values included.
    fileError = clsparseSBoolCsrMatrixfromFile( A, matrix_path.c_str( ), createResult.control, true );

    // This function allocates memory for rowBlocks structure. If not called
    // the structure will not be calculated and clSPARSE will run the vectorized
    // version of SpMV instead of adaptive;
    clsparseBoolCsrMetaCreate( A, createResult.control );

    if (fileError != clsparseSuccess)
    {
        std::cout << "Problem with reading matrix from " << matrix_path
                  << std::endl;
        return -6;
    }
}

int main (int argc, char* argv[])
{
    //parse command line
    std::string matrix_path;

    if (argc < 2)
    {
        std::cout << "Not enough parameters. "
                  << "Please specify path to matrix in mtx format as parameter"
                  << std::endl;
        return -1;
    }
    else
    {
        matrix_path = std::string(argv[1]);
    }

    std::cout << "Executing sample clSPARSE SpMM (y = A*A) C++" << std::endl;

    std::cout << "Matrix will be read from: " << matrix_path << std::endl;

    /**  Step 1. Setup OpenCL environment; **/

    // Init OpenCL environment;
    cl_int cl_status;

    // Get OpenCL platforms
    std::vector<cl::Platform> platforms;

    cl_status = cl::Platform::get(&platforms);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting OpenCL platforms"
                  << " [" << cl_status << "]" << std::endl;
        return -2;
    }

    int platform_id = 0;
    for (const auto& p : platforms)
    {
        std::cout << "Platform ID " << platform_id++ << " : "
                  << p.getInfo<CL_PLATFORM_NAME>() << std::endl;

    }

    // Using first platform
    platform_id = 0;
    cl::Platform platform = platforms[platform_id];

    // Get device from platform
    std::vector<cl::Device> devices;
    cl_status = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting devices from platform"
                  << " [" << platform_id << "] " << platform.getInfo<CL_PLATFORM_NAME>()
                  << " error: [" << cl_status << "]" << std::endl;
    }

    std::cout << std::endl
              << "Getting devices from platform " << platform_id << std::endl;
    cl_int device_id = 0;
    for (const auto& device : devices)
    {
        std::cout << "Device ID " << device_id++ << " : "
                  << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    }

    // Using first device;
    device_id = 0;
    cl::Device device = devices[device_id];

    // Create OpenCL context;
    cl::Context context (device);

    // Create OpenCL queue;
    cl::CommandQueue queue(context, device);

    /** Step 2. Setup GPU buffers **/

    /** Step 3. Init clSPARSE library **/

    clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with executing clsparseSetup()" << std::endl;
        return -3;
    }


    // Create clsparseControl object
    clsparseCreateResult createResult = clsparseCreateControl( queue( ) );
    CLSPARSE_V( createResult.status, "Failed to create clsparse control" );

    clsparseBoolCsrMatrix A;
    int error = readAndCreateBoolCsrMatrix(&A, matrix_path, createResult, context);
    clsparseBoolCsrMatrix B;
    int error1 = readAndCreateBoolCsrMatrix(&B, matrix_path, createResult, context);


    clsparseBoolCsrMatrix C;
    /**Step 4. Call the spmv algorithm */
    // status = clsparseBoolScsrSpGemm(&A, &B, &C, createResult.control );
    status = clsparseBoolScsrElemAdd(&A, &B, &C, createResult.control );

    std::vector<int> csrRowPtrC_h((C.num_rows + 1), 0);
    int run_status = clEnqueueReadBuffer(queue(),
                                     C.row_pointer,
                                     1,
                                     0,
                                     (C.num_rows + 1)*sizeof(cl_int),
                                     csrRowPtrC_h.data(),
                                     0,
                                     0,
                                     0);
    for (auto i = csrRowPtrC_h.begin(); i != csrRowPtrC_h.end(); ++i)
    {
        std::cout << *i << ' '; 
    }
    std::cout << std::endl;

    std::vector<int> csrColIndC_h(C.num_nonzeros, 0);
    run_status = clEnqueueReadBuffer(queue(),
                                     C.col_indices,
                                     1,
                                     0,
                                     C.num_nonzeros*sizeof(cl_int),
                                     csrColIndC_h.data(),
                                     0,
                                     0,
                                     0);
    for (auto i = csrColIndC_h.begin(); i != csrColIndC_h.end(); ++i)
    {
        std::cout << *i << ' '; 
    }
    std::cout << std::endl;


    // CPU ADDITION

    assert(A.num_rows == B.num_rows);

    clsparseIdx_t* row_ptr_A = (clsparseIdx_t*)malloc((A.num_rows + 1) * sizeof(clsparseIdx_t));
    clsparseIdx_t* cols_A = (clsparseIdx_t*)malloc(A.num_nonzeros * sizeof(clsparseIdx_t));
    clsparseIdx_t* row_ptr_B = (clsparseIdx_t*)malloc((B.num_rows + 1) * sizeof(clsparseIdx_t));
    clsparseIdx_t* cols_B = (clsparseIdx_t*)malloc(B.num_nonzeros * sizeof(clsparseIdx_t));

    clEnqueueReadBuffer(queue(), A.row_pointer, CL_TRUE, 0, (A.num_rows + 1) * sizeof(clsparseIdx_t),
                        row_ptr_A, 0, NULL, NULL);
    clEnqueueReadBuffer(queue(), A.col_indices, CL_TRUE, 0, A.num_nonzeros * sizeof(clsparseIdx_t),
                        cols_A, 0, NULL, NULL);

    clEnqueueReadBuffer(queue(), B.row_pointer, CL_TRUE, 0, (B.num_rows + 1) * sizeof(clsparseIdx_t),
                        row_ptr_B, 0, NULL, NULL);
    clEnqueueReadBuffer(queue(), B.col_indices, CL_TRUE, 0, B.num_nonzeros * sizeof(clsparseIdx_t),
                        cols_B, 0, NULL, NULL);

    std::vector<int> row_ptr_C;
    std::vector<int> cols_C;

    row_ptr_C.push_back(0);
    for (int i = 1; i <= A.num_rows; i++)
    {
        int start_A = row_ptr_A[i - 1];
        int end_A = row_ptr_A[i];
        int start_B = row_ptr_B[i - 1];
        int end_B = row_ptr_B[i];

        std::vector<int> dst;
        std::merge(cols_A + start_A, cols_A + end_A, cols_B + start_B, cols_B + end_B, std::back_inserter(dst));
        dst.erase(std::unique(dst.begin(), dst.end()), dst.end());

        row_ptr_C.push_back(row_ptr_C[i - 1] + dst.size());
        cols_C.insert(cols_C.end(), dst.begin(), dst.end());
        dst.clear();
    }

    for (auto i = row_ptr_C.begin(); i != row_ptr_C.end(); ++i)
    {
        std::cout << *i << ' '; 
    }
    std::cout << std::endl;

    for (auto i = cols_C.begin(); i != cols_C.end(); ++i)
    {
        std::cout << *i << ' '; 
    }
    std::cout << std::endl;

    // VERIFY RESULTS

    assert(csrRowPtrC_h == row_ptr_C);
    assert(csrColIndC_h == cols_C);


    if (status != clsparseSuccess)
    {
        std::cout << "Problem with execution SpMV algorithm."
                  << " Error: " << status << std::endl;
    }


    /** Step 5. Close & release resources */
    status = clsparseReleaseControl( createResult.control );
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with releasing control object."
                  << " Error: " << status << std::endl;
    }

    status = clsparseTeardown();

    if (status != clsparseSuccess)
    {
        std::cout << "Problem with closing clSPARSE library."
                  << " Error: " << status << std::endl;
    }


    //release mem;
    clsparseBoolCsrMetaDelete( &A );
    clReleaseMemObject ( A.col_indices );
    clReleaseMemObject ( A.row_pointer );

    clsparseBoolCsrMetaDelete( &B );
    clReleaseMemObject ( B.col_indices );
    clReleaseMemObject ( B.row_pointer );

    std::cout << C.num_nonzeros << std::endl;
    clReleaseMemObject ( C.col_indices );
    clReleaseMemObject ( C.row_pointer );
    std::cout << "Program completed successfully." << std::endl;

    return 0;
}
