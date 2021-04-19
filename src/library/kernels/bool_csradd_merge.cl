R"(
#define WG_SIZE 64
#define warp_size 64
#define blocksize WG_SIZE
#define uint unsigned int
// #include "assert.cl"  // include assert* macros

__kernel
void merge_count(
        __global const int *rpt_a,
        __global const int *col_a,
        __global const int *rpt_b,
        __global const int *col_b,
        __global int *rpt_c)
{
    // WHAT IS THE DIFFERENCE BETWEEN blockDim.x and blocksize?

    // const int blocksize = WG_SIZE;

    //workgroup for row
    const int row = get_group_id(0);

    const int max_val = INT_MAX;

    const  global_offset_a = rpt_a[row];
    const int sz_a = rpt_a[row + 1] - global_offset_a;

    const int global_offset_b = rpt_b[row];
    const int sz_b = rpt_b[row + 1] - global_offset_b;

    const int block_count = (sz_a + sz_b + blocksize - 1) / blocksize;

    int begin_a = 0;
    int begin_b = 0;

    __local int raw_a[blocksize + 2];
    __local int raw_b[blocksize + 2];
    __local int res[blocksize];

    bool dir = true;
    int item_from_prev_chank = max_val;

    for (auto i = 0; i < block_count; i++) {
        __local int max_x_index;
        __local int max_y_index;

        int max_x_index_per_thread = 0;
        int max_y_index_per_thread = 0;

        // assert(sz_a >= begin_a);
        // assert(sz_b >= begin_b);

        int buf_a_size = min(sz_a - begin_a, blocksize);
        int buf_b_size = min(sz_b - begin_b, blocksize);

        if (get_local_id(0) == 0) {
            max_x_index = 0;
            max_y_index = 0;
        }

        for (auto j = get_local_id(0); j < blocksize + 2; j += WG_SIZE) {
            if (j > 0 && j - 1 < buf_a_size) {
                raw_a[j] = col_a[global_offset_a + j - 1 + begin_a];
            } else {
                raw_a[j] = max_val;
            }
            if (j > 0 && j - 1 < buf_b_size) {
                raw_b[j] = col_b[global_offset_b + j - 1 + begin_b];
            } else {
                raw_b[j] = max_val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int to_process = min(buf_b_size + buf_a_size, blocksize);

        for (auto j = get_local_id(0); j < to_process; j += WG_SIZE) {
            const int y = j + 2;
            const int x = 0;

            int l = 0;
            int r = j + 2;

            while (r - l > 1) {
                bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

                l += (r - l) / 2 * ans;
                r -= (r - l) / 2 * !ans;
            }

            int ans_x = x + l;
            int ans_y = y - l;

            if (ans_y == 1 || ans_x == 0) {
                if (ans_y == 1) {
                res[j] = raw_a[ans_x];
                max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                } else {
                res[j] = raw_b[ans_y - 1];
                max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                }
            } else {
                if (raw_b[ans_y - 1] > raw_a[ans_x]) {
                res[j] = raw_b[ans_y - 1];
                max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                } else {
                res[j] = raw_a[ans_x];
                max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                }
            }
        }

        atomic_max(&max_x_index, max_x_index_per_thread);
        atomic_max(&max_y_index, max_y_index_per_thread);

        barrier(CLK_LOCAL_MEM_FENCE);

        int counter = 0;

        if (dir) {
            for (auto m = get_local_id(0); m < to_process; m += WG_SIZE) {
                if (m > 0)
                counter += (res[m] - res[m - 1]) != 0;
                else
                counter += (res[0] - item_from_prev_chank) != 0;
                item_from_prev_chank = res[m];  
            }
        } else {
            for (auto m = WG_SIZE - 1 - get_local_id(0); m < to_process; m += WG_SIZE) {
                if (m > 0)
                counter += (res[m] - res[m - 1]) != 0;
                else
                counter += (res[0] - item_from_prev_chank) != 0;
                item_from_prev_chank = res[m];
            }
        }

        dir = !dir;

        atomic_add(rpt_c + row, counter);

        begin_a += max_x_index;
        begin_b += max_y_index;

        // local -> global
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // int global_id = get_global_id(0);
    // int group_id = get_group_id(0);
    // printf("Hello %d %d\n", group_id, global_id);
}




__kernel
void merge_fill(
    __global const int *rpt_a,
    __global const int *col_a,
    __global const int *rpt_b,
    __global const int *col_b,
    __global const int *rpt_c,
    __global int *col_c)
{
    const int row = get_group_id(0);

    const int max_val = INT_MAX;

    const  global_offset_a = rpt_a[row];
    const int sz_a = rpt_a[row + 1] - global_offset_a;

    const int global_offset_b = rpt_b[row];
    const int sz_b = rpt_b[row + 1] - global_offset_b;

    int global_offset_c = rpt_c[row];

    const int block_count = (sz_a + sz_b + WG_SIZE - 1) / WG_SIZE;

    int begin_a = 0;
    int begin_b = 0;

    __local int raw_a[WG_SIZE + 2];
    __local int raw_b[WG_SIZE + 2];
    __local int res[WG_SIZE];

    bool dir = true;
    int item_from_prev_chank = max_val;

    for (auto i = 0; i < block_count; i++) {
        __local int max_x_index;
        __local int max_y_index;

        int max_x_index_per_thread = 0;
        int max_y_index_per_thread = 0;

        // assert(sz_a >= begin_a);
        // assert(sz_b >= begin_b);

        int buf_a_size = min(sz_a - begin_a, WG_SIZE);
        int buf_b_size = min(sz_b - begin_b, WG_SIZE);

        if (get_local_id(0) == 0) {
            max_x_index = 0;
            max_y_index = 0;
        }

        for (auto j = get_local_id(0); j < WG_SIZE + 2; j += WG_SIZE) {
            if (j > 0 && j - 1 < buf_a_size) {
                raw_a[j] = col_a[global_offset_a + j - 1 + begin_a];
            } else {
                raw_a[j] = max_val;
            }
            if (j > 0 && j - 1 < buf_b_size) {
                raw_b[j] = col_b[global_offset_b + j - 1 + begin_b];
            } else {
                raw_b[j] = max_val;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int to_process = min(buf_b_size + buf_a_size, WG_SIZE);
        int answer = max_val;

        const auto j = dir ? get_local_id(0) : WG_SIZE - 1 - get_local_id(0);

        if (j < to_process) {
            const int y = j + 2;
            const int x = 0;

            int l = 0;
            int r = j + 2;

            while (r - l > 1) {
                bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

                l += (r - l) / 2 * ans;
                r -= (r - l) / 2 * !ans;
            }

            int ans_x = x + l;
            int ans_y = y - l;

            if (ans_y == 1 || ans_x == 0) {
                if (ans_y == 1) {
                    answer = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                } else {
                    answer = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                }
            } else {
                if (raw_b[ans_y - 1] > raw_a[ans_x]) {
                    answer = raw_b[ans_y - 1];
                    max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
                } else {
                    answer = raw_a[ans_x];
                    max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
                }
            }
        }

        atomic_max(&max_x_index, max_x_index_per_thread);
        atomic_max(&max_y_index, max_y_index_per_thread);

        res[j] = answer;

        barrier(CLK_LOCAL_MEM_FENCE);

        bool take = j < to_process;
        if (j > 0)
            take = take && (answer - res[j - 1]) != 0;
        else
            take = take && (answer - item_from_prev_chank) != 0;

        item_from_prev_chank = answer;

        barrier(CLK_LOCAL_MEM_FENCE);

        res[j] = take;

        barrier(CLK_LOCAL_MEM_FENCE);

        // TODO: optimize prefix sum
        // however it is only WG_SIZE elems...
        // if (j == 0) {
        //     for (int i = 1; i < WG_SIZE; i++) {
        //         int accum = res[i-1];
        //         res[i] += accum;
        //     }
        // }

        // EXACT COPY OF SCAN_BELLOCH
        // TODO: move scan_belloch to reuse here

        uint local_id = get_local_id(0);
        uint block_size = get_local_size(0);
        uint dp = 1;

        for(uint s = block_size>>1; s > 0; s >>= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(local_id < s)
            {
                uint i = dp*(2 * local_id + 1) - 1;
                uint j = dp*(2 * local_id + 2) - 1;
                res[j] += res[i];
            }
 
            dp <<= 1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id == block_size - 1) {
            res[local_id] = 0;
        }

        barrier(CLK_GLOBAL_MEM_FENCE);

        for(uint s = 1; s < block_size; s <<= 1)
        {
            dp >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if(local_id < s)
            {
                uint i = dp*(2 * local_id + 1) - 1;
                uint j = dp*(2 * local_id + 2) - 1;

                unsigned int t = res[j];
                res[j] += res[i];
                res[i] = t;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        //END OF EXACT COPY

        barrier(CLK_LOCAL_MEM_FENCE);

        if (take) {
            col_c[global_offset_c + (j == 0 ? 0 : res[j - 1])] = answer;
        }

        global_offset_c += res[WG_SIZE - 1];

        dir = !dir;

        begin_a += max_x_index;
        begin_b += max_y_index;
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
    

)"
