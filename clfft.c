#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cl-helper.h"
#include "timing.h"

cl_context ctx;

cl_int work_size;

cl_command_queue queue;

cl_kernel knl;

cl_mem cl_src, cl_dst;

cl_mem *in, *out;

long N;

void naivefft() {
    char *knl_text = read_file("kernels/naivefft.cl");
    knl = kernel_from_string(ctx, knl_text, "naivefft", NULL);
    free(knl_text);
    
    in = &cl_src; out = &cl_dst;
    size_t ldim[] = {2};
    size_t gdim[] = {N};
   
    SET_3_KERNEL_ARGS(knl, *in, *out, N);

    timestamp_type start, end;
    get_timestamp(&start);

    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
        (queue, knl,
        /*dimensions*/ 1, NULL, gdim, ldim,
        0, NULL, NULL));
    CALL_CL_GUARDED(clFinish, (queue));
    get_timestamp(&end);
    double elapsed = timestamp_diff_in_seconds(start, end);
    printf("Elapsed time: %fs\n", elapsed);
}

void cooleyTukey() {
    char *knl_text = read_file("kernels/cooleyTukey.cl");
    knl = kernel_from_string(ctx, knl_text, "cooleyTukeyfft", NULL);
    free(knl_text);

    int block_size = 16;
    int logN = (int) log2(N);
    //SET_6_KERNEL_ARGS(knl, cl_src_re, cl_src_im, N, logN, block_size, N);
    
    size_t ldim[] = {block_size};
    size_t gdim[] = {N};
    
    timestamp_type start, end;
    get_timestamp(&start);
    
    CALL_CL_GUARDED(clEnqueueNDRangeKernel,
        (queue, knl,
         /*dimensions*/ 1, NULL, gdim, ldim,
         0, NULL, NULL));
    
    CALL_CL_GUARDED(clFinish, (queue));
    get_timestamp(&end);
    double elapsed = timestamp_diff_in_seconds(start, end);
    printf("Elapsed time: %fs\n", elapsed);
}

void swap(cl_mem** a, cl_mem** b) {
    cl_mem* temp = *a;
    *a = *b;
    *b = temp;
}

void stockham() {
    char *knl_text = read_file("kernels/fft_radix8.cl");
    knl = kernel_from_string(ctx, knl_text, "fft_radix8", NULL);
    //char *knl_text = read_file("kernels/fft_radix4.cl");
    //knl = kernel_from_string(ctx, knl_text, "fft_radix4", NULL);
    //char *knl_text = read_file("kernels/fft_radix2.cl");
    //knl = kernel_from_string(ctx, knl_text, "fft_radix2", NULL); 
    free(knl_text);
    
    int logN = (int) log2(N);
    int block_size = 1;
    int thread_count = N / 8;
    size_t ldim[] = {512};
    size_t gdim[] = {thread_count};
    in = &cl_src; out = &cl_dst;
    
    timestamp_type start, end;
    get_timestamp(&start);
   
    for (int t = 1; t <= logN / 3; ++t) {
        SET_4_KERNEL_ARGS(knl, *in, *out, block_size, thread_count);
        
        CALL_CL_GUARDED(clEnqueueNDRangeKernel,
            (queue, knl,
            /*dimensions*/ 1, NULL, gdim, ldim,
            0, NULL, NULL));
        swap(&in, &out);
        block_size *= 8;
    }
    CALL_CL_GUARDED(clFinish, (queue));
    get_timestamp(&end);
    double elapsed = timestamp_diff_in_seconds(start, end);
    printf("Elapsed time: %fs\n", elapsed);
    printf("Performance: %.2f Gflops\n", 5 * N * logN / elapsed / 1e9);
}


int clFFT(const float* src, float* dst, int n, char* device) {
    N = n;
    print_devices();
    
    create_context_on(device, NULL, 0, &ctx, &queue, 0);
   
    cl_int status;

    cl_src = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
        sizeof(cl_float2) * N, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");

    cl_dst = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
        sizeof(cl_float2) * N, 0, &status);
    CHECK_CL_ERROR(status, "clCreateBuffer");
    
    CALL_CL_GUARDED(clEnqueueWriteBuffer, (
        queue, cl_src, CL_TRUE, 0, sizeof(float) * N * 2, 
        src, 0, NULL, NULL));
    
    CALL_CL_GUARDED(clFinish, (queue));
    

    //naivefft();
    //cooleyTukey();
    stockham();

    CALL_CL_GUARDED(clEnqueueReadBuffer, (
        queue, *out, CL_TRUE, 0, sizeof(float) * N * 2, 
        dst, 0, NULL, NULL));
    

    CALL_CL_GUARDED(clFinish, (queue));


    CALL_CL_GUARDED(clReleaseMemObject, (cl_src));
    CALL_CL_GUARDED(clReleaseMemObject, (cl_dst));
    CALL_CL_GUARDED(clReleaseKernel, (knl));
    CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
    CALL_CL_GUARDED(clReleaseContext, (ctx));
}







