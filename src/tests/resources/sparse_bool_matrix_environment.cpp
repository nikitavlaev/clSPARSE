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
#include "sparse_bool_matrix_environment.h"

CSRSparseBoolEnvironment::sMatrixType CSRSparseBoolEnvironment::ublasSCsr = CSRSparseBoolEnvironment::sMatrixType();

clsparseIdx_t CSRSparseBoolEnvironment::n_rows = 0;
clsparseIdx_t CSRSparseBoolEnvironment::n_cols = 0;
clsparseIdx_t CSRSparseBoolEnvironment::n_vals = 0;

clsparseBoolCsrMatrix CSRSparseBoolEnvironment::csrSMatrix = clsparseBoolCsrMatrix();

std::string CSRSparseBoolEnvironment::file_name = std::string();