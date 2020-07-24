#ifndef _INCL_CORE
#define _INCL_CORE

#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define CALLBACK1 v0::cudaCallbackCPU
#define CALLBACK2 v1::cudaCallbackCPU
#define CALLBACK3 v2::cudaCallbackCPU
#define CALLBACK4 v3::cudaCallbackCPU
#define CALLBACK5 v4::cudaCallbackCPU
#define CALLBACK6 v5::cudaCallbackCPU
#define CALLBACK7 v6::cudaCallbackCPU
#define CALLBACK8 v7::cudaCallbackCPU
#define CALLBACK9 v8::cudaCallbackCPU
#define CALLBACK10 v9::cudaCallbackCPU

// The main function would invoke the "cudaCallback"s on each sample. Note that
// you don't have to (and shouldn't) free the space of searchPoints,
// referencePoints, and result by yourself since the main function have included
// the free statements already.
//
// To make the program work, you shouldn't modify the signature of\
// "cudaCallback"s.
namespace v0
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v1
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v2
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v3
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v4
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v5
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v6
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v7
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v8
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};
namespace v9
{
    extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);
};

extern void cudaCallback(int k, int m, int n, float *searchPoints, float *referencePoints, int **results);

// divup calculates n / m and would round it up if the remainder is non-zero.
extern int divup(int n, int m);

// CHECK macro from Grossman and McKercher, "Professional CUDA C Programming"
#define CHECK(call)                                       \
    {                                                     \
        const cudaError_t error = call;                   \
        if (error != cudaSuccess)                         \
        {                                                 \
            printf("Error: %s:%d, ", __FILE__, __LINE__); \
            printf("code:%d, reason: %s \n",              \
                   error, cudaGetErrorString(error));     \
            exit(1);                                      \
        }                                                 \
    }

#endif