/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
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
#ifndef _SPARSE_BOOL_MATRIX_ENVIRONMENT_H_
#define _SPARSE_BOOL_MATRIX_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>

#include "clsparse_environment.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include <cfloat>

using CLSE = ClSparseEnvironment;

namespace uBLAS = boost::numeric::ublas;

/**
* @brief The CSRSparseBoolEnvironment class will have the input parameters for SpMSpM tests
* They are list of csr matrices in csr format in mtx files.
*/
// Currently only single precision is considered
class CSRSparseBoolEnvironment : public ::testing::Environment {
public:
    using sMatrixType = uBLAS::compressed_matrix<float, uBLAS::row_major, 0, uBLAS::unbounded_array<clsparseIdx_t> >;
    //using dMatrixType = uBLAS::compressed_matrix<double, uBLAS::row_major, 0, uBLAS::unbounded_array<size_t> >;

    explicit CSRSparseBoolEnvironment(const std::string& path, cl_command_queue queue, cl_context context, cl_bool explicit_zeroes = true)
        : queue(queue), context(context)
    {
        file_name = path;
        clsparseStatus read_status = clsparseHeaderfromFile(&n_vals, &n_rows, &n_cols, file_name.c_str());
        if (read_status)
        {
            exit(-3);
        }

        clsparseInitBoolCsrMatrix(&csrSMatrix);
        csrSMatrix.num_nonzeros = n_vals;
        csrSMatrix.num_rows = n_rows;
        csrSMatrix.num_cols = n_cols;

        //  Load single precision data from file; this API loads straight into GPU memory
        cl_int status;

        csrSMatrix.col_indices = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrix.num_nonzeros * sizeof(cl_int), NULL, &status);

        csrSMatrix.row_pointer = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int), NULL, &status);

        clsparseStatus fileError = clsparseSBoolCsrMatrixfromFile(&csrSMatrix, file_name.c_str(), CLSE::control, explicit_zeroes);
        if (fileError != clsparseSuccess)
            throw std::runtime_error("Could not read matrix market data from disk");

        clsparseBoolCsrMetaCreate( &csrSMatrix, CLSE::control );

        //  Download sparse matrix data to host
        //  First, create space on host to hold the data
        ublasSCsr = sMatrixType(n_rows, n_cols, n_vals);

        // This is nasty. Without that call ublasSCsr is not working correctly.
        ublasSCsr.complete_index1_data();

        // copy host matrix arrays to device;
        cl_int copy_status;

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.row_pointer, CL_TRUE, 0,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int),
            ublasSCsr.index1_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.col_indices, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_int),
            ublasSCsr.index2_data().begin(),
            0, NULL, NULL);

        for (int r = 0; r < n_rows; r++) {
            for (int j = ublasSCsr.index1_data()[r]; j < ublasSCsr.index1_data()[r + 1]; j++) {
                ublasSCsr.value_data()[r, ublasSCsr.index2_data()[j]] = 1;
            }
        }

        if (copy_status)
        {
            TearDown();
            exit(-5);
        }
    }// end C'tor

    void SetUp()
    {
        // Prepare data to it's default state
    }

    //cleanup
    void TearDown()
    {
    }

    std::string getFileName()
    {
        return file_name;
    }

    ~CSRSparseBoolEnvironment()
    {
        //release buffers;
        ::clReleaseMemObject(csrSMatrix.col_indices);
        ::clReleaseMemObject(csrSMatrix.row_pointer);

        //bring csrSMatrix  to its initial state
        clsparseInitBoolCsrMatrix(&csrSMatrix);
    }
        

    static sMatrixType ublasSCsr;
    //static sMatrixType ublasCsrB;
    //static sMatrixType ublasCsrC;    

    static clsparseIdx_t n_rows;
    static clsparseIdx_t n_cols;
    static clsparseIdx_t n_vals;

    //cl buffers ;
    static clsparseBoolCsrMatrix csrSMatrix; // input 1

    //static clsparseCsrMatrix csrMatrixC; // output

    static std::string file_name;

private:
    cl_command_queue queue;
    cl_context context;
};


#endif // _SPARSE_MATRIX_ENVIRONMENT_H_
