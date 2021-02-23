R"(
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
#ifdef cl_khr_fp64
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
      #error "Double precision floating point not supported by OpenCL implementation."
#endif


#define TUPLE_QUEUE 6
// typedef double   vT;
#define vT float

inline
void siftDown(__local int   *s_key,
              const int start,
              const int stop,
              const int local_id,
              const int local_size)
{
    int root = start;
    int child, swap;

    int temp_swap_key;

    while (root * 2 + 1 <= stop)
    {
        child = root * 2 + 1;
        swap = root;

        if (s_key[swap * local_size + local_id] < s_key[child * local_size + local_id])
            swap = child;

        if (child + 1 <= stop && s_key[swap * local_size + local_id] < s_key[(child + 1) * local_size + local_id])
            swap = child + 1;

        if (swap != root)
        {
            const int index1 = root * local_size + local_id;
            const int index2 = swap * local_size + local_id;

            //swap root and swap
            temp_swap_key = s_key[index1];
            s_key[index1] = s_key[index2];
            s_key[index2] = temp_swap_key;

            root = swap;
        }
        else
            return;
    }
}

inline
int heapsort(__local int   *s_key,
             const int segment_size,
             const int local_id,
             const int local_size)
{
    // heapsort - heapify max-heap
    int start = (segment_size - 1) / 2;
    int stop  = segment_size - 1;

    int index1, index2;

    while (start >= 0)
    {
        siftDown(s_key, start, stop, local_id, local_size);
        start--;
    }

    // inject root element to the end

    int temp_swap_key;

    index1 = stop * local_size + local_id;

    temp_swap_key = s_key[local_id];
    s_key[local_id] = s_key[index1];
    s_key[index1] = temp_swap_key;

    stop--;

    siftDown(s_key, 0, stop, local_id, local_size);

    // this start is compressed list's start
    start = segment_size - 1;

    // heapsort - remove-max and compress
    while (stop >= 0)
    {
        index2 = stop * local_size + local_id;

        if (s_key[local_id] == s_key[start * local_size + local_id])
        {
            s_key[local_id] = s_key[index2];
        }
        else
        {
            start--;

            index1 = start * local_size + local_id;

            if (stop == start)
            {
                temp_swap_key = s_key[local_id];
                s_key[local_id] = s_key[index2];
                s_key[index2] = temp_swap_key;
            }
            else
            {
                s_key[index1] = s_key[local_id];

                s_key[local_id] = s_key[index2];
            }
        }

        stop--;

        siftDown(s_key, 0, stop, local_id, local_size);
    }

    return start;
}

__kernel
void ESC_2heap_noncoalesced_local(__global const int   *d_queue,
                    __global const int   *d_csrRowPtrA,
                    __global const int   *d_csrColIndA,
                    __global const int   *d_csrRowPtrB,
                    __global const int   *d_csrColIndB,
                    __global int         *d_csrRowPtrC,
                    __global const int   *d_csrRowPtrCt,
                    __global int         *d_csrColIndCt,
                    __local  int         *s_key,          // SEGMENTSIZE * LOCALSIZE
                    const int queue_size,
                    const int queue_offset)
{
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int global_id = get_global_id(0);
    const int local_size = get_local_size(0);
    int index = 0;

    if (global_id < queue_size)
    {
        int i, counter = 0;
        int start_col_index_A, stop_col_index_A;
        int rowidB, start_col_index_B, stop_col_index_B;

        int rowidC = d_queue[TUPLE_QUEUE * (queue_offset + global_id)];

        start_col_index_A = d_csrRowPtrA[rowidC];
        stop_col_index_A  = d_csrRowPtrA[rowidC + 1];

        // i is both col index of A and row index of B
        for (i = start_col_index_A; i < stop_col_index_A; i++)
        {
            rowidB = d_csrColIndA[i];

            start_col_index_B = d_csrRowPtrB[rowidB];
            stop_col_index_B  = d_csrRowPtrB[rowidB + 1];

            for (int j = start_col_index_B; j < stop_col_index_B; j++)
            {
                index = counter * local_size + local_id;
                s_key[index] = d_csrColIndB[j];

                counter++;
            }
        }

        // heapsort in each work-item
        int local_start = heapsort(s_key, counter, local_id, local_size);

        counter -= local_start;
        d_csrRowPtrC[rowidC] = counter;

        const int base_index = d_queue[TUPLE_QUEUE * (queue_offset + group_id * local_size + local_id) + 1];;
        for (int i = 0; i < counter; i++)
        {
            d_csrColIndCt[base_index + i] = s_key[(local_start+i) * local_size + local_id];
        }
    }
}


)"
