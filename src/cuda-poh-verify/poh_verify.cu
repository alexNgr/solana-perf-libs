#include <stddef.h>
#include <inttypes.h>
#include <pthread.h>
#include "gpu_common.h"
#include "sha256.cu"

#define MAX_NUM_GPUS 8
#define MAX_QUEUE_SIZE 8
#define NUM_THREADS_PER_BLOCK 64


__global__ void poh_verify_kernel(uint8_t* hashes, uint64_t* num_hashes_arr, size_t num_elems) {
    size_t idx = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= num_elems) return;

    uint8_t hash[SHA256_BLOCK_SIZE];

    memcpy(hash, &hashes[idx * SHA256_BLOCK_SIZE], SHA256_BLOCK_SIZE);
    if (idx == 0 ) {
        printf("inceput hashes[0]= %u\n", hashes[0]);
    }

    if (idx == 0 ) {
        printf("hashes[32] = %u\n", hashes[32]);
        printf("inceput hash[32]= %u\n", hash[32]);
    }

    for (size_t i = 0; i < num_hashes_arr[idx]; i++) {
        hash_state sha_state;
        sha256_init(&sha_state);
        if (idx == 0 && (i == 0 || i == 1)) {
            printf("sha_state.curlen = %u\n", sha_state.sha256.curlen);
            printf("sha_state.length = %lu\n", sha_state.sha256.length); 
            printf("sha_state.state[0] = %u\n", sha_state.sha256.state[0]);
            printf("sha_state.state[1] = %u\n", sha_state.sha256.state[1]);
            printf("sha_state.state[2] = %u\n", sha_state.sha256.state[2]);
            printf("sha_state.state[3] = %u\n", sha_state.sha256.state[3]);
            printf("sha_state.state[4] = %u\n", sha_state.sha256.state[4]);
            printf("sha_state.state[5] = %u\n", sha_state.sha256.state[5]);
            printf("sha_state.state[6] = %u\n", sha_state.sha256.state[6]);
            printf("sha_state.state[7] = %u\n", sha_state.sha256.state[7]);
        }
        sha256_process(&sha_state, hash, SHA256_BLOCK_SIZE);
        if (idx == 0 && ((i == 0 || i == 1))) {
            printf("dupa sha256_process hash[0]= %u\n", hash[0]);
            printf("dupa sha256_process hash[1]= %u\n", hash[1]);
            printf("dupa sha256_process hash[2]= %u\n", hash[2]);
            printf("dupa sha256_process hash[3]= %u\n", hash[3]);
            printf("dupa sha256_process hash[4]= %u\n", hash[4]);
            printf("dupa sha256_process hash[5]= %u\n", hash[5]);
            printf("dupa sha256_process hash[6]= %u\n", hash[6]);
            printf("dupa sha256_process hash[7]= %u\n", hash[7]);
            printf("dupa sha256_process hash[8]= %u\n", hash[8]);
            printf("dupa sha256_process hash[9]= %u\n", hash[9]);
            printf("dupa sha256_process hash[10]= %u\n", hash[10]);
            printf("dupa sha256_process hash[11]= %u\n", hash[11]);
            printf("dupa sha256_process hash[12]= %u\n", hash[12]);
            printf("dupa sha256_process hash[13]= %u\n", hash[13]);
            printf("dupa sha256_process hash[14]= %u\n", hash[14]);
            printf("dupa sha256_process hash[15]= %u\n", hash[15]);
            printf("dupa sha256_process hash[16]= %u\n", hash[16]);
            printf("dupa sha256_process hash[17]= %u\n", hash[17]);
            printf("dupa sha256_process hash[18]= %u\n", hash[18]);
            printf("dupa sha256_process hash[19]= %u\n", hash[19]);
            printf("dupa sha256_process hash[20]= %u\n", hash[20]);
            printf("dupa sha256_process hash[21]= %u\n", hash[21]);
            printf("dupa sha256_process hash[22]= %u\n", hash[22]);
            printf("dupa sha256_process hash[23]= %u\n", hash[23]);
            printf("dupa sha256_process hash[24]= %u\n", hash[24]);
            printf("dupa sha256_process hash[25]= %u\n", hash[25]);
            printf("dupa sha256_process hash[26]= %u\n", hash[26]);
            printf("dupa sha256_process hash[27]= %u\n", hash[27]);
            printf("dupa sha256_process hash[28]= %u\n", hash[28]);
            printf("dupa sha256_process hash[29]= %u\n", hash[29]);
            printf("dupa sha256_process hash[30]= %u\n", hash[30]);
            printf("dupa sha256_process hash[31]= %u\n", hash[31]);
        }
        if ( i == 0 && idx == 0)
            sha256_done(&sha_state, hash, 0);
        else 
            sha256_done(&sha_state, hash, 1);
        if (idx == 0 && ((i == 0 || i == 1))) {
            printf("dupa sha256_done hash[0]= %u\n", hash[0]);
            printf("dupa sha256_done hash[1]= %u\n", hash[1]);
            printf("dupa sha256_done hash[2]= %u\n", hash[2]);
            printf("dupa sha256_done hash[3]= %u\n", hash[3]);
            printf("dupa sha256_done hash[4]= %u\n", hash[4]);
            printf("dupa sha256_done hash[5]= %u\n", hash[5]);
            printf("dupa sha256_done hash[6]= %u\n", hash[6]);
            printf("dupa sha256_done hash[7]= %u\n", hash[7]);
            printf("dupa sha256_done hash[8]= %u\n", hash[8]);
            printf("dupa sha256_done hash[9]= %u\n", hash[9]);
            printf("dupa sha256_done hash[10]= %u\n", hash[10]);
            printf("dupa sha256_done hash[11]= %u\n", hash[11]);
            printf("dupa sha256_done hash[12]= %u\n", hash[12]);
            printf("dupa sha256_done hash[13]= %u\n", hash[13]);
            printf("dupa sha256_done hash[14]= %u\n", hash[14]);
            printf("dupa sha256_done hash[15]= %u\n", hash[15]);
            printf("dupa sha256_done hash[16]= %u\n", hash[16]);
            printf("dupa sha256_done hash[17]= %u\n", hash[17]);
            printf("dupa sha256_done hash[18]= %u\n", hash[18]);
            printf("dupa sha256_done hash[19]= %u\n", hash[19]);
            printf("dupa sha256_done hash[20]= %u\n", hash[20]);
            printf("dupa sha256_done hash[21]= %u\n", hash[21]);
            printf("dupa sha256_done hash[22]= %u\n", hash[22]);
            printf("dupa sha256_done hash[23]= %u\n", hash[23]);
            printf("dupa sha256_done hash[24]= %u\n", hash[24]);
            printf("dupa sha256_done hash[25]= %u\n", hash[25]);
            printf("dupa sha256_done hash[26]= %u\n", hash[26]);
            printf("dupa sha256_done hash[27]= %u\n", hash[27]);
            printf("dupa sha256_done hash[28]= %u\n", hash[28]);
            printf("dupa sha256_done hash[29]= %u\n", hash[29]);
            printf("dupa sha256_done hash[30]= %u\n", hash[30]);
            printf("dupa sha256_done hash[31]= %u\n", hash[31]);
        }
    }  
    memcpy(&hashes[idx * SHA256_BLOCK_SIZE], hash, SHA256_BLOCK_SIZE);
    if (idx == 0 ) {
       printf("sfarsit aici hash[0] = %u\n", hashes[idx]);
    } 
}

typedef struct {
    uint8_t* hashes;
    uint64_t* num_hashes_arr;
    size_t num_elems_alloc;
    pthread_mutex_t mutex;
    cudaStream_t stream;
} gpu_ctx;

static size_t index_file=0;

static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

static gpu_ctx g_gpu_ctx[MAX_NUM_GPUS][MAX_QUEUE_SIZE] = {0};
static uint32_t g_cur_gpu = 0;
static uint32_t g_cur_queue[MAX_NUM_GPUS] = {0};
static int32_t g_total_gpus = -1;

static bool poh_init_locked() {
    if (g_total_gpus == -1) {
        cudaGetDeviceCount(&g_total_gpus);
        g_total_gpus = min(MAX_NUM_GPUS, g_total_gpus);
        LOG("total_gpus: %d\n", g_total_gpus);
        for (int gpu = 0; gpu < g_total_gpus; gpu++) {
            CUDA_CHK(cudaSetDevice(gpu));
            for (int queue = 0; queue < MAX_QUEUE_SIZE; queue++) {
                int err = pthread_mutex_init(&g_gpu_ctx[gpu][queue].mutex, NULL);
                if (err != 0) {
                    fprintf(stderr, "pthread_mutex_init error %d gpu: %d queue: %d\n",
                            err, gpu, queue);
                    g_total_gpus = 0;
                    return false;
                }
                CUDA_CHK(cudaStreamCreate(&g_gpu_ctx[gpu][queue].stream));
            }
        }
    }
    return g_total_gpus > 0;
}

bool poh_init() {
    cudaFree(0);
    pthread_mutex_lock(&g_ctx_mutex);
    bool success = poh_init_locked();
    pthread_mutex_unlock(&g_ctx_mutex);
    return success;
}

void save_input(uint8_t* hashes,
    const uint64_t* num_hashes_arr,
    size_t num_elems) {

    FILE * fp;

    char *file_name = "test_hashes";
    char temp_string[20];
    sprintf(temp_string, "%s_%zu", file_name, index_file);

    //sprintf(temp_string, "%zu", num_elems);

    fp = fopen (temp_string, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not create file %s\n",temp_string);
        exit(-1);
    }

    FILE * fp2;

    char *file_name2 = "test_num_hashes_arr";
    char temp_string2[25];
    sprintf(temp_string2, "%s_%zu", file_name2, index_file);
    fp2 = fopen (temp_string2, "w");
    if (fp2 == NULL) {
        fprintf(stderr, "Could not create file %s\n", temp_string2);
        exit(-2);
    }

    FILE * fp3;

    char *file_name3 = "test_num_elems";
    char temp_string3[25];
    sprintf(temp_string3, "%s_%zu", file_name3, index_file);
    fp3 = fopen (temp_string3, "w");
    if (fp3 == NULL) {
        fprintf(stderr, "Could not create file %s\n", file_name3);
        exit(-3);
    }

    fprintf(fp3, "%lu", num_elems*32*8);
    fclose(fp3);

    for (size_t i = 0; i < num_elems*32*8; ++i) {
        fprintf(fp, "%hhu ", hashes[i]);
    }
    fclose(fp);


    for (size_t i = 0; i < num_elems; ++i) {
        fprintf(fp2, "%lu ", num_hashes_arr[i]);
    }
    fclose(fp2);
    index_file++;

}

void static inline save_out(uint8_t* hashes,
    size_t num_elems) {

    FILE * fp;
    printf("in save out");
    char *file_name = "test_hashes_output";
    char temp_string[25];
    sprintf(temp_string, "%s_%zu", file_name, index_file-1);

    fp = fopen (temp_string, "w");
    if (fp == NULL) {
        fprintf(stderr, "Could not create file %s\n", temp_string);
        exit(-1);
    }

    for (size_t i = 0; i < num_elems*32*8; ++i) {
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
    LOG("Starting poh_verify_many: num_elems: %zu\n", num_elems);
    save_input(hashes, num_hashes_arr, num_elems);

    if (num_elems == 0) return 0;

    int32_t cur_gpu, cur_queue;

    pthread_mutex_lock(&g_ctx_mutex);
    if (!poh_init_locked()) {
        pthread_mutex_unlock(&g_ctx_mutex);
        LOG("No GPUs, exiting...\n");
        return 1;
    }
    cur_gpu = g_cur_gpu;
    g_cur_gpu++;
    g_cur_gpu %= g_total_gpus;
    cur_queue = g_cur_queue[cur_gpu];
    g_cur_queue[cur_gpu]++;
    g_cur_queue[cur_gpu] %= MAX_QUEUE_SIZE;
    pthread_mutex_unlock(&g_ctx_mutex);

    gpu_ctx* cur_ctx = &g_gpu_ctx[cur_gpu][cur_queue];
    pthread_mutex_lock(&cur_ctx->mutex);

    CUDA_CHK(cudaSetDevice(cur_gpu));

    LOG("cur gpu: %d cur queue: %d\n", cur_gpu, cur_queue);

    size_t hashes_size = num_elems * SHA256_BLOCK_SIZE * sizeof(uint8_t);
    size_t num_hashes_size = num_elems * sizeof(uint64_t);

    // Ensure there is enough memory allocated
    if (cur_ctx->hashes == NULL || cur_ctx->num_elems_alloc < num_elems) {
        CUDA_CHK(cudaFree(cur_ctx->hashes));
        CUDA_CHK(cudaMalloc(&cur_ctx->hashes, hashes_size));
        CUDA_CHK(cudaFree(cur_ctx->num_hashes_arr));
        CUDA_CHK(cudaMalloc(&cur_ctx->num_hashes_arr, num_hashes_size));
        printf("s-a alocat \n");
        cur_ctx->num_elems_alloc = num_elems;
    }

    cudaStream_t stream = 0;
    if (0 != use_non_default_stream) {
        stream = cur_ctx->stream;
    }

    CUDA_CHK(cudaMemcpyAsync(cur_ctx->hashes, hashes, hashes_size, cudaMemcpyHostToDevice, stream));
    CUDA_CHK(cudaMemcpyAsync(cur_ctx->num_hashes_arr, num_hashes_arr, num_hashes_size, cudaMemcpyHostToDevice, stream));

    int num_blocks = ROUND_UP_DIV(num_elems, NUM_THREADS_PER_BLOCK);

    poh_verify_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(cur_ctx->hashes, cur_ctx->num_hashes_arr, num_elems);
    CUDA_CHK(cudaPeekAtLastError());
    printf("DUPA kernel apelat"); 
    CUDA_CHK(cudaMemcpyAsync(hashes, cur_ctx->hashes, hashes_size, cudaMemcpyDeviceToHost, stream));
    printf("DUPA kernel apelat2");
    CUDA_CHK(cudaStreamSynchronize(stream));
    save_out(hashes, num_elems);
    pthread_mutex_unlock(&cur_ctx->mutex);

    return 0;
}
}
