#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <pthread.h>
#include <CL/cl.h>
#include "../opencl-platform/cl_common.h"
#include "perftime.h"

#define MAX_NUM_GPUS 	1
#define MAX_QUEUE_SIZE 	1


typedef struct {
    cl_mem in_out_hashes;
    cl_mem in_num_hashes_arr;
    size_t in_num_elems;

    pthread_mutex_t mutex;
} gpu_ctx_t;


static gpu_ctx_t g_gpu_ctx[MAX_NUM_GPUS][MAX_QUEUE_SIZE] = {0};
static uint32_t g_cur_gpu = 0;
static uint32_t g_cur_queue[MAX_NUM_GPUS] = {0};
static int32_t g_total_gpus = -1;

extern bool cl_check_init();
static pthread_mutex_t clg_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;

static bool cl_crypt_init_locked() {
    if (g_total_gpus == -1) {
        g_total_gpus = MAX_NUM_GPUS;
        LOG("total_gpus: %d\n", g_total_gpus);
        for (int gpu = 0; gpu < g_total_gpus; gpu++) {
            for (int queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
                int err = pthread_mutex_init(&g_gpu_ctx[gpu][queue].mutex, NULL);
                if (err != 0) {
                    fprintf(stderr, "pthread_mutex_init error %d gpu: %d queue: %d\n",
                            err, gpu, queue);
                    g_total_gpus = 0;
                    return false;
                }
            }
        }
    }
    return g_total_gpus > 0;
}

gpu_ctx_t* get_gpu_ctx() {
    int32_t cur_gpu, cur_queue;

    LOG("getting gpu_ctx\n");

    cur_gpu = g_cur_gpu;
    g_cur_gpu++;
    g_cur_gpu %= g_total_gpus;
    cur_queue = g_cur_queue[cur_gpu];
    g_cur_queue[cur_gpu]++;
    g_cur_queue[cur_gpu] %= MAX_QUEUE_SIZE;

    gpu_ctx_t* cur_ctx = &g_gpu_ctx[cur_gpu][cur_queue];
    LOG("locking contex mutex queue: %d gpu: %d\n", cur_queue, cur_gpu);
    pthread_mutex_lock(&cur_ctx->mutex);

    LOG("selecting gpu: %d queue: %d\n", cur_gpu, cur_queue);

    return cur_ctx;
}

void setup_gpu_ctx(gpu_ctx_t *cur_ctx,
                    uint8_t* hashes,
                    const uint64_t* num_hashes_arr,
                    size_t num_elems,
                    size_t nr_bytes_hashes,
                    size_t nr_bytes_num_hashes_arr
                   ) {
	int ret;

    LOG("device allocate. num hashes: %lu nr_bytes_hashes in MB: %f nr_bytes_num_hashes_arr: %f\n",
        num_elems, (double)nr_bytes_hashes/(1024*1024), (double)nr_bytes_num_hashes_arr/(1024*1024));

    if (cur_ctx->in_out_hashes == NULL || cur_ctx->in_num_elems < num_elems) {
        clReleaseMemObject(cur_ctx->in_out_hashes);
        cur_ctx->in_out_hashes = clCreateBuffer(context, CL_MEM_READ_WRITE, nr_bytes_hashes, NULL, &ret);
        CL_ERR( ret );
        cur_ctx->in_num_elems = num_elems;
    }

    if (cur_ctx->in_num_hashes_arr == NULL || cur_ctx->in_num_elems < num_elems) {
        clReleaseMemObject(cur_ctx->in_num_hashes_arr);
        cur_ctx->in_num_hashes_arr = clCreateBuffer(context, CL_MEM_READ_ONLY, nr_bytes_num_hashes_arr, NULL, &ret);
        CL_ERR( ret );
    }

    CL_ERR( clEnqueueWriteBuffer(cmd_queue, cur_ctx->in_out_hashes, CL_TRUE, 0, nr_bytes_hashes, hashes, 0, NULL, NULL));
    CL_ERR( clEnqueueWriteBuffer(cmd_queue, cur_ctx->in_num_hashes_arr, CL_TRUE, 0, nr_bytes_num_hashes_arr, num_hashes_arr, 0, NULL, NULL));
}

void release_gpu_ctx(gpu_ctx_t* cur_ctx) {
    pthread_mutex_unlock(&cur_ctx->mutex);
}

void static inline save_out(uint8_t* hashes,
    size_t num_elems) {

    FILE * fp;

    char *file_name = "test_hashes_output_cl";
    char temp_string[35];
    sprintf(temp_string, "%s", file_name);

    fp = fopen (temp_string, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not create file %s\n", temp_string);
        exit(-1);
    }

    for (size_t i = 0; i < num_elems*32*8; ++i) {
        //fprintf(stderr, "alexNNN inainte de inchidere %s writing hashes[%lu]=%hhu \n", file_name, i, hashes[i]);
        fprintf(fp, "%hhu ", hashes[i]);
    }
    fclose(fp);
}

extern "C" {
int poh_verify_many(uint8_t* hashes,
                    const uint64_t* num_hashes_arr,
                    size_t num_elems,
                    uint8_t use_non_default_stream)
{
    pthread_mutex_lock(&clg_ctx_mutex);
    bool success = false;

    if (initialized == false) {
        success = cl_check_init();
        DIE(success == false, "OpenCL could not be init");
        DIE(cl_crypt_init_locked() == false, "cl_crypt_init_locked failed");
        initialized = true;
    } else {
        fprintf(stderr, "cl_poh_verify_many already initialized\n");
    }

    gpu_ctx_t *cur_ctx = get_gpu_ctx();
    pthread_mutex_unlock(&clg_ctx_mutex);
    
    size_t nr_bytes_hashes = num_elems * 32 * sizeof(uint8_t);
    size_t nr_bytes_num_hashes_arr = num_elems * sizeof(uint64_t);
    setup_gpu_ctx(cur_ctx,
                    hashes,
                    num_hashes_arr,
                    num_elems,
                    nr_bytes_hashes,
                    nr_bytes_num_hashes_arr);

    if (num_elems == 0) {
        return 0;
    }

    if (success) {
        fprintf(stderr, "cl_poh_verify_many initialized with success\n");
    } else {
        fprintf(stderr, "cl_poh_verify_many failed to initialize\n");
    }
    
    size_t num_threads_per_block = 64;
    size_t num_blocks = ROUND_UP_DIV(num_elems, num_threads_per_block) * num_threads_per_block;
    LOG("num_blocks: %zu threads_per_block: %zu nr hashes: %lu\n",
           num_blocks, num_threads_per_block, num_elems);                     
                             
    CL_ERR( clSetKernelArg(poh_verify_kernel, 0, sizeof(cl_mem), (void *)&cur_ctx->in_out_hashes) );
    CL_ERR( clSetKernelArg(poh_verify_kernel, 1, sizeof(cl_mem), (void *)&cur_ctx->in_num_hashes_arr) );
    CL_ERR( clSetKernelArg(poh_verify_kernel, 2, sizeof(cl_uint), (void *)&cur_ctx->in_num_elems) );

	perftime_t start, end;
    get_time(&start);

    size_t globalSize[2] = {num_blocks * num_threads_per_block, 0};
    size_t localSize[2] = {num_threads_per_block, 0};    
    cl_int ret = clEnqueueNDRangeKernel(cmd_queue, poh_verify_kernel, 1, NULL,
        globalSize, localSize, 0, NULL, NULL);
    CL_ERR( ret );
    if(ret != CL_SUCCESS) {
        fprintf(stderr, "ret is not CL_SUCCESS after NDRANGE\n");
    }

    ret = clFinish(cmd_queue);
    if(ret != CL_SUCCESS) {
        fprintf(stderr, "ret is not CL_SUCCESS after clFinish\n");
    }        
    ret = clEnqueueReadBuffer(cmd_queue, cur_ctx->in_out_hashes, CL_TRUE, 0, nr_bytes_hashes, hashes, 0, NULL, NULL);
    if(ret != CL_SUCCESS) {
        fprintf(stderr, "ret is not CL_SUCCESS after clEnqueueReadBuffer\n");
    }

    get_time(&end);
    LOG("time diff: %f\n", get_diff(&start, &end));
    fprintf(stderr, "alexNNN\n");    
    save_out(hashes, num_elems);
    fprintf(stderr, "poh_verify_many not implemented.\n");
    release_gpu_ctx(cur_ctx);
    //exit(1);
    return 0;
}

// 0 == success
static int
get_checked_scalar(unsigned char* scalar, const unsigned char* signature) {
    // Check if top 4-bits are clear
    // then scalar is reduced.
    // if ((signature[31] & 0xf0) == 0) {
    //     for (int i = 0; i < 32; i++) {
    //         scalar[i] = signature[i];
    //     }
    //     return 0;
    // }

    // if ((signature[31] >> 7) != 0) {
    //     return 1;
    // }

    // scalar32_reduce(scalar);
    // if (!consttime_equal(scalar, signature)) {
    //     return 1;
    // }
    fprintf(stderr, "get_checked_scalar not implemented.\n");
    exit(0);
    return 0;

}

int ed25519_get_checked_scalar(unsigned char* out_scalar, const unsigned char* in_scalar) {
    return get_checked_scalar(out_scalar, in_scalar);
}

// Return 0=success if ge unpacks and is not small order
static int
check_packed_ge_small_order(const unsigned char* packed_group_element) {
    // ge_p3 signature_R;

    // fail if ge does not unpack
    // if (0 != ge_frombytes_negate_vartime(&signature_R, packed_group_element)) {
    //     return 1;
    // }

    // // fail if ge is small order
    // if (0 != ge_is_small_order(&signature_R)) {
    //     return 1;
    // }
    fprintf(stderr, "check_packed_ge_small_order not implemented.\n");
    exit(0);
    return 0;
}

int ed25519_check_packed_ge_small_order(const unsigned char* packed_group_element) {
    return check_packed_ge_small_order(packed_group_element);
}

}
