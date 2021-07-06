#include <iostream>
#include <fstream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION
#include <CL/cl2.hpp>

#include "clSPARSE.h"

void printDense(std::ostream &out, clsparseBoolCsrMatrix A, std::vector<int> csrRowPtrA_h, std::vector<int> csrColIndA_h)
{

    for (int i = 0; i < A.num_rows; i++)
    {
        std::vector<int> buffer(A.num_cols, 0);
        for (uint j = csrRowPtrA_h[i]; j < csrRowPtrA_h[i + 1]; j++)
        {
            buffer[csrColIndA_h[j]] = 1;
        }
        for (int j = 0; j < A.num_cols; j++)
        {
            out << buffer[j] << " ";
        }
        out << std::endl;
    }
}

void printCoord(std::ostream &out, clsparseBoolCsrMatrix A, std::vector<int> csrRowPtrA_h, std::vector<int> csrColIndA_h)
{
    out << A.num_rows << " " << A.num_cols << " " << A.num_nonzeros << std::endl;

    for (int r = 0; r < A.num_rows; r++)
    {
        for (int c_id = csrRowPtrA_h[r]; c_id < csrRowPtrA_h[r + 1]; c_id++)
        {
            out << r << " " << csrColIndA_h[c_id] << std::endl;
        }
    } 
}

clsparseStatus
clsparseOutputBoolCsrMatrix(std::ostream &out, clsparseBoolCsrMatrix A, cl_command_queue queue, int dense, int add_mtx_header)
{   
    // read data to host
    std::vector<int> csrRowPtrA_h((A.num_rows + 1), 0);
    int run_status = clEnqueueReadBuffer(queue,
                                     A.row_pointer,
                                     1,
                                     0,
                                     (A.num_rows + 1)*sizeof(cl_int),
                                     csrRowPtrA_h.data(),
                                     0,
                                     0,
                                     0);

    std::vector<int> csrColIndA_h(A.num_nonzeros, 0);
    run_status = clEnqueueReadBuffer(queue,
                                     A.col_indices,
                                     1,
                                     0,
                                     A.num_nonzeros*sizeof(cl_int),
                                     csrColIndA_h.data(),
                                     0,
                                     0,
                                     0);

    if (add_mtx_header)
    {
        out << "%%MatrixMarket matrix coordinate pattern general" << std::endl;
    }
    
    if (dense) {
        printDense(out, A, csrRowPtrA_h, csrColIndA_h);
    } else {
        printCoord(out, A, csrRowPtrA_h, csrColIndA_h);
    }

    return clsparseSuccess;
}

clsparseStatus
clsparseBoolCsrMatrixDumpToMtx(clsparseBoolCsrMatrix A, cl_command_queue queue, char* filename)
{
    std::ofstream out;
    out.open(filename);
    clsparseStatus status = clsparseOutputBoolCsrMatrix(out, A, queue, 0, 1);
    out.close();
    return status;
}

clsparseStatus
clsparsePrintBoolCsrMatrix(clsparseBoolCsrMatrix A, cl_command_queue queue, int dense = 1)
{
    return clsparseOutputBoolCsrMatrix(std::cout, A, queue, dense, 0);
}