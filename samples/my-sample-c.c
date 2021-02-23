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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h> 

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION
#include <CL/cl.h>

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
 * If rowBlocks are calculated the clsparseBoolCsrMatrix.rowBlocks field is not null.
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

int readAndCreateCsrMatrix(clsparseBoolCsrMatrix* A, char* matrix_path, clsparseCreateResult createResult, cl_context context)
{
    cl_int cl_status;
    clsparseInitBoolCsrMatrix(A);
    // Read matrix from file. Calculates the rowBlocks structures as well.
    clsparseIdx_t nnz, row, col;
    // read MM header to get the size of the matrix;
    clsparseStatus fileError
            = clsparseHeaderfromFile( &nnz, &row, &col, matrix_path );

    if( fileError != clsparseSuccess )
    {
        printf("Could not read matrix market header from disk\n");
        return -5;
    }

    A->num_nonzeros = nnz;
    A->num_rows = row;
    A->num_cols = col;

    // Allocate memory for CSR matrix
    // A->values = clCreateBuffer( context, CL_MEM_READ_ONLY,
    //                              A->num_nonzeros * sizeof( float ), NULL, &cl_status );

    A->col_indices = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                     A->num_nonzeros * sizeof( clsparseIdx_t ), NULL, &cl_status );

    A->row_pointer = clCreateBuffer( context, CL_MEM_READ_ONLY,
                                     ( A->num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, &cl_status );


    // Read matrix market file with explicit zero values included.
    fileError = clsparseSBoolCsrMatrixfromFile( A, matrix_path, createResult.control, 1 );

    // This function allocates memory for rowBlocks structure. If not called
    // the structure will not be calculated and clSPARSE will run the vectorized
    // version of SpMV instead of adaptive;
    clsparseBoolCsrMetaCreate( A, createResult.control );

    if (fileError != clsparseSuccess)
    {
        printf("Problem with reading matrix from %s\n", matrix_path);
    }
}

int returnCsrToHost(clsparseBoolCsrMatrix A, clsparseIdx_t* row_ptr, clsparseIdx_t* cols, float* vals, cl_command_queue queue)
{
    clEnqueueReadBuffer(queue, A.row_pointer, CL_TRUE, 0, (A.num_rows + 1) * sizeof(clsparseIdx_t),
                        row_ptr, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, A.col_indices, CL_TRUE, 0, A.num_nonzeros * sizeof(clsparseIdx_t),
                        cols, 0, NULL, NULL);
    // clEnqueueReadBuffer(queue, A.values, CL_TRUE, 0, A.num_nonzeros * sizeof(float),
    //                     vals, 0, NULL, NULL);
}

void printMatrixNonZeros(int num_rows, clsparseIdx_t* row_ptr_A, clsparseIdx_t* cols_A)
{
    for (int i = 0; i < num_rows; i++)
    {
        printf("%d %d   ", row_ptr_A[i], row_ptr_A[i + 1]);
        printf("row: %d col: ", i);
        for (clsparseIdx_t j = row_ptr_A[i]; j < row_ptr_A[i + 1]; j++)
        {
            printf("(%u %u) ", j, cols_A[j]);
        }
        printf("\n");
    }
}

void printMatrixDense(clsparseBoolCsrMatrix A, clsparseIdx_t* row_ptr_A, clsparseIdx_t* cols_A)
{
    char* buffer = malloc(A.num_cols * sizeof(char));

    for (int i = 0; i < A.num_rows; i++)
    {
        memset(buffer, 0, A.num_cols);
        for (uint j = row_ptr_A[i]; j < row_ptr_A[i + 1]; j++)
        {
            buffer[cols_A[j]] = 1;
        }
        for (int j = 0; j < A.num_cols; j++)
        {
            printf("%d ", buffer[j]);
        }
        printf("\n");
    }

    free(buffer);
}

int main(int argc, char* argv[])
{
    //parse command line
    char* matrix_path;

    if (argc < 2)
    {
        printf("Not enough parameters. Please specify path to matrix in mtx format as parameter\n");
        return -1;
    }
    else
    {
        matrix_path = argv[1];
    }

    printf( "Executing sample clSPARSE SpMM (y = A*A) C\n");

    printf( "Matrix will be read from: %s", matrix_path);

    /**  Step 1. Setup OpenCL environment; **/

    cl_int cl_status = CL_SUCCESS;

    cl_platform_id* platforms = NULL;
    cl_device_id* devices = NULL;
    cl_uint num_platforms = 0;
    cl_uint num_devices = 0;


    // Get number of compatible OpenCL platforms
    cl_status = clGetPlatformIDs(0, NULL, &num_platforms);

    if (num_platforms == 0)
    {
        printf ("No OpenCL platforms found. Exiting.\n");
        return 0;
    }

    // Allocate memory for platforms
    platforms = (cl_platform_id*) malloc (num_platforms * sizeof(cl_platform_id));

    // Get platforms
    cl_status = clGetPlatformIDs(num_platforms, platforms, NULL);

    if (cl_status != CL_SUCCESS)
    {
        printf("Problem with getting platform IDs. Err: %d\n", cl_status);
        free(platforms);
        return -1;
    }


    // Get devices count from first available platform;
    cl_status = clGetDeviceIDs(platforms[ 0 ], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);

    if (num_devices == 0)
    {
        printf("No OpenCL GPU devices found on platform 0. Exiting\n");
        free(platforms);
        return -2;
    }

    // Allocate space for devices
    devices = (cl_device_id*) malloc( num_devices * sizeof(cl_device_id));

    // Get devices from platform 0;
    cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);

    if (cl_status != CL_SUCCESS)
    {
        printf("Problem with getting device id from platform. Exiting\n");
        free(devices);
        free(platforms);
        return -3;
    }

    // Get context and queue
    cl_context context = clCreateContext( NULL, 1, devices, NULL, NULL, NULL );
    cl_command_queue queue = clCreateCommandQueue( context, devices[ 0 ], 0, NULL );

    /** Step 2. Setup GPU buffers **/

    /** Step 3. Init clSPARSE library **/

    clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        printf( "Problem with executing clsparseSetup()\n");
        return -3;
    }

    // Create clsparseControl object
    clsparseCreateResult createResult = clsparseCreateControl( queue );
    CLSPARSE_V( createResult.status, "Failed to create clsparse control" );

    clsparseBoolCsrMatrix A;
    int error = readAndCreateCsrMatrix(&A, matrix_path, createResult, context);
    clsparseBoolCsrMatrix B;
    int error1 = readAndCreateCsrMatrix(&B, matrix_path, createResult, context);

    clsparseBoolCsrMatrix C;
    /**Step 4. Call the spgemm algorithm */
    status = clsparseBoolScsrSpGemm(&A, &B, &C, createResult.control );

    if (status != clsparseSuccess)
    {
        printf( "Problem with execution SpMV algorithm.\n Error: \n", status);
    }

    /** Step 5. Close & release resources */
    status = clsparseReleaseControl( createResult.control );
    if (status != clsparseSuccess)
    {
        printf( "Problem with releasing control object.\n Error: \n", status);
    }

    status = clsparseTeardown();

    if (status != clsparseSuccess)
    {
        printf( "Problem with closing clSPARSE library.\n Error: \n", status);
    }

    printf("A \n");

    clsparseIdx_t* row_ptr_A = malloc((A.num_rows + 1) * sizeof(clsparseIdx_t));
    clsparseIdx_t* cols_A = malloc(A.num_nonzeros * sizeof(clsparseIdx_t));
    float* vals_A = malloc(A.num_nonzeros * sizeof(float));

    returnCsrToHost(A, row_ptr_A, cols_A, vals_A, queue);

    printMatrixNonZeros(A.num_rows, row_ptr_A, cols_A);
    // printMatrixDense(A, row_ptr_A, cols_A);

    printf("C \n");

    clsparseIdx_t* row_ptr_C = malloc((C.num_rows + 1) * sizeof(clsparseIdx_t));
    clsparseIdx_t* cols_C = malloc(C.num_nonzeros * sizeof(clsparseIdx_t));
    float* vals_C = malloc(C.num_nonzeros * sizeof(clsparseIdx_t));

    returnCsrToHost(C, row_ptr_C, cols_C, vals_C, queue);

    printMatrixNonZeros(C.num_rows, row_ptr_C, cols_C);
    // printMatrixDense(C, row_ptr_C, cols_C);

    //release mem;
    clsparseBoolCsrMetaDelete( &A );
    // clReleaseMemObject ( A.values );
    clReleaseMemObject ( A.col_indices );
    clReleaseMemObject ( A.row_pointer );

    clsparseBoolCsrMetaDelete( &B );
    // clReleaseMemObject ( B.values );
    clReleaseMemObject ( B.col_indices );
    clReleaseMemObject ( B.row_pointer );

    printf("%d %d %d\n", C.num_rows, C.num_cols, C.num_nonzeros);
    // clReleaseMemObject ( C.values );
    clReleaseMemObject ( C.col_indices );
    clReleaseMemObject ( C.row_pointer );
    printf( "Program completed successfully.\n");

    return 0;
}
