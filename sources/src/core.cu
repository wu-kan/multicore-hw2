#include "core.h"
#include <thrust/device_vector.h>
struct WuKTimer
{
    cudaEvent_t beg, end;
    WuKTimer()
    {
        cudaEventCreate(&beg);
        cudaEventCreate(&end);
        cudaEventRecord(beg);
    }
    ~WuKTimer()
    {
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        float elapsed_time;
        cudaEventElapsedTime(
            &elapsed_time,
            beg,
            end);
        printf("%f\n", elapsed_time);
    }
};
namespace v0
{
    void cudaCallback(
        int k,
        int m,
        int n,
        float *searchPoints,
        float *referencePoints,
        int **results)
    {

        int *tmp = (int *)malloc(sizeof(int) * m);
        int minIndex;
        float minSquareSum, diff, squareSum;

        // Iterate over all search points
        for (int mInd = 0; mInd < m; mInd++)
        {
            minSquareSum = -1;
            // Iterate over all reference points
            for (int nInd = 0; nInd < n; nInd++)
            {
                squareSum = 0;
                for (int kInd = 0; kInd < k; kInd++)
                {
                    diff = searchPoints[k * mInd + kInd] - referencePoints[k * nInd + kInd];
                    squareSum += (diff * diff);
                }
                if (minSquareSum < 0 || squareSum < minSquareSum)
                {
                    minSquareSum = squareSum;
                    minIndex = nInd;
                }
            }
            tmp[mInd] = minIndex;
        }

        *results = tmp;
        // Note that you don't have to free searchPoints, referencePoints, and
        // *results by yourself
    }
} // namespace v0
namespace v1
{
    __global__ void
    get_dis_kernel(
        const int k,
        const int m,
        const int n,
        const float *__restrict__ searchPoints,
        const float *__restrict__ referencePoints,
        float *__restrict__ dis)
    {
        const int
            nInd = threadIdx.x + blockIdx.x * blockDim.x,
            mInd = threadIdx.y + blockIdx.y * blockDim.y;
        if (nInd < n && mInd < m)
        {
            float ans = 0;
            for (int kInd = 0; kInd < k; ++kInd)
            {
                const float d = searchPoints[kInd + mInd * k] - referencePoints[kInd + nInd * k];
                ans += d * d;
            }
            dis[nInd + mInd * n] = ans;
        }
    }
    void cudaCallback(
        int k,
        int m,
        int n,
        float *searchPoints,
        float *referencePoints,
        int **results)
    {
        thrust::device_vector<float>
            s_d(searchPoints, searchPoints + k * m),
            r_d(referencePoints, referencePoints + k * n),
            dis_d(m * n);
        const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
        {
            WuKTimer t1;
            get_dis_kernel<<<
                dim3(divup(n, BLOCK_DIM_X), divup(m, BLOCK_DIM_Y)),
                dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
                k,
                m,
                n,
                thrust::raw_pointer_cast(s_d.data()),
                thrust::raw_pointer_cast(r_d.data()),
                thrust::raw_pointer_cast(dis_d.data()));
        }
        *results = (int *)malloc(sizeof(int) * m);
        {
            WuKTimer t2;
            for (int i = 0; i < m; ++i)
                (*results)[i] = thrust::min_element(dis_d.begin() + n * i, dis_d.begin() + n * i + n) - dis_d.begin() - n * i;
        }
    }
}; // namespace v1
namespace v2
{
    __global__ void
    get_dis_kernel(
        const int k,
        const int m,
        const int n,
        const float *__restrict__ searchPoints,
        const float *__restrict__ referencePoints,
        float *__restrict__ dis)
    {
        const int
            nInd = threadIdx.x + blockIdx.x * blockDim.x,
            mInd = threadIdx.y + blockIdx.y * blockDim.y;
        if (nInd < n && mInd < m)
        {
            float ans = 0;
            for (int kInd = 0; kInd < k; ++kInd)
            {
                const float d = searchPoints[kInd + mInd * k] - referencePoints[kInd + nInd * k];
                ans += d * d;
            }
            dis[nInd + mInd * n] = ans;
        }
    }
    void cudaCallback(
        int k,
        int m,
        int n,
        float *searchPoints,
        float *referencePoints,
        int **results)
    {
        thrust::device_vector<float>
            s_d(searchPoints, searchPoints + k * m),
            r_d(referencePoints, referencePoints + k * n),
            dis_d(m * n);
        const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
        {
            WuKTimer t1;
            get_dis_kernel<<<
                dim3(divup(n, BLOCK_DIM_X), divup(m, BLOCK_DIM_Y)),
                dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
                k,
                m,
                n,
                thrust::raw_pointer_cast(s_d.data()),
                thrust::raw_pointer_cast(r_d.data()),
                thrust::raw_pointer_cast(dis_d.data()));
        }
        *results = (int *)malloc(sizeof(int) * m);
        {
            WuKTimer t2;
#pragma omp parallel for
            for (int i = 0; i < m; ++i)
                (*results)[i] = thrust::min_element(dis_d.begin() + n * i, dis_d.begin() + n * i + n) - dis_d.begin() - n * i;
        }
    }
}; // namespace v2
void cudaCallback(
    int k,
    int m,
    int n,
    float *searchPoints,
    float *referencePoints,
    int **results)
{
    v1::cudaCallback(
        k,
        m,
        n,
        searchPoints,
        referencePoints,
        results);
}