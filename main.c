#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "clfft.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: clfft N device_name\n");
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]);
    char* device = argv[2];

    float* src = (float*) malloc(sizeof(float) * n * 2);
    float* dst = (float*) malloc(sizeof(float) * n * 2);
    
    if (!src || !dst) {
        fprintf(stderr, "Not enough memory!\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 2 * n; i += 2) {
        src[i] = 1.0;
        src[i + 1] = 1.0;
    }

    clFFT(src, dst, n, device);

    for (int i = 0; i < 2 * n; i += 2) {
        printf("%d %9f %9f\n", i / 2, dst[i], dst[i + 1]);
    }

    free(src);
    free(dst);
    exit(EXIT_SUCCESS);
}
