#include <math.h>
#include <thrust/device_vector.h>
#include "core.h"

__constant__ float const_mem[(64 << 10) / sizeof(float)];

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
		thrust::device_vector<float> dis_d(m * n);
		{
			thrust::device_vector<float>
				s_d(searchPoints, searchPoints + k * m),
				r_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t1;
			get_dis_kernel<<<
				dim3(divup(n, BLOCK_DIM_X), m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				thrust::raw_pointer_cast(s_d.data()),
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(dis_d.data()));
		}
		*results = (int *)malloc(sizeof(int) * m);
		{
			//WuKTimer t2;
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
		thrust::device_vector<float> dis_d(m * n);
		{
			thrust::device_vector<float>
				s_d(searchPoints, searchPoints + k * m),
				r_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t1;
			get_dis_kernel<<<
				dim3(divup(n, BLOCK_DIM_X), m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				thrust::raw_pointer_cast(s_d.data()),
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(dis_d.data()));
		}
		*results = (int *)malloc(sizeof(int) * m);
		{
			//WuKTimer t2;
#pragma omp parallel for
			for (int i = 0; i < m; ++i)
				(*results)[i] = thrust::min_element(dis_d.begin() + n * i, dis_d.begin() + n * i + n) - dis_d.begin() - n * i;
		}
	}
}; // namespace v2
namespace v3
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
	template <int BLOCK_DIM_X>
	__global__ void
	get_min_kernel(
		const int result_size,
		const int m,
		const int n,
		const float *__restrict__ dis,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x + blockIdx.y * gridDim.x;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			const float d = dis[nInd + blockIdx.y * n];
			if (dis_s[threadIdx.x] > d)
			{
				dis_s[threadIdx.x] = d;
				ind_s[threadIdx.x] = nInd;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM_X >> 1; offset > 0; offset >>= 1)
		{
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset])
				{
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[ans_id] = ind_s[0];
	}
	void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		thrust::device_vector<float> dis_d(m * n);
		{
			thrust::device_vector<float>
				s_d(searchPoints, searchPoints + k * m),
				r_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t1;
			get_dis_kernel<<<
				dim3(divup(n, BLOCK_DIM_X), m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				thrust::raw_pointer_cast(s_d.data()),
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(dis_d.data()));
		}
		thrust::device_vector<int> results_d(m);
		{
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t2;
			get_min_kernel<
				BLOCK_DIM_X><<<
				dim3(results_d.size() / m, m),
				BLOCK_DIM_X>>>(
				results_d.size(),
				m,
				n,
				thrust::raw_pointer_cast(dis_d.data()),
				thrust::raw_pointer_cast(results_d.data()));
		}
		thrust::copy(
			results_d.begin(),
			results_d.begin() + m,
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v3
namespace v4
{
	template <int BLOCK_DIM_X>
	__global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ searchPoints,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x + blockIdx.y * gridDim.x;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float dis = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float d = searchPoints[kInd + mInd * k] - referencePoints[kInd + nInd * k];
				dis += d * d;
			}
			if (dis_s[threadIdx.x] > dis)
			{
				dis_s[threadIdx.x] = dis;
				ind_s[threadIdx.x] = nInd;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM_X >> 1; offset > 0; offset >>= 1)
		{
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset])
				{
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[ans_id] = ind_s[0];
	}
	void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		thrust::device_vector<int> results_d(m);
		{
			thrust::device_vector<float>
				s_d(searchPoints, searchPoints + k * m),
				r_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 1024;
			WuKTimer t1;
			cudaCallbackKernel<
				BLOCK_DIM_X><<<
				dim3(results_d.size() / m, m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				results_d.size(),
				thrust::raw_pointer_cast(s_d.data()),
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(results_d.data()));
		}
		thrust::copy(
			results_d.begin(),
			results_d.begin() + m,
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v4
namespace v5
{
	template <int BLOCK_DIM_X>
	__global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x + blockIdx.y * gridDim.x;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float dis = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float d = const_mem[kInd + mInd * k] - referencePoints[kInd + nInd * k];
				dis += d * d;
			}
			if (dis_s[threadIdx.x] > dis)
			{
				dis_s[threadIdx.x] = dis;
				ind_s[threadIdx.x] = nInd;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM_X >> 1; offset > 0; offset >>= 1)
		{
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset])
				{
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[ans_id] = ind_s[0];
	}
	void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		assert(k * m <= (64 << 10) / sizeof(float));
		CHECK(cudaMemcpyToSymbol(const_mem, searchPoints, sizeof(float) * k * m));
		thrust::device_vector<int> results_d(m);
		{
			thrust::device_vector<float>
				r_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 1024;
			WuKTimer t1;
			cudaCallbackKernel<
				BLOCK_DIM_X><<<
				dim3(results_d.size() / m, m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				results_d.size(),
				thrust::raw_pointer_cast(r_d.data()),
				thrust::raw_pointer_cast(results_d.data()));
		}
		thrust::copy(
			results_d.begin(),
			results_d.begin() + m,
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v5
namespace v6
{
	template <int BLOCK_DIM_X>
	__global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		cudaTextureObject_t texObj, //使用纹理对象
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x + blockIdx.y * gridDim.x;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float dis = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float d = const_mem[kInd + mInd * k] - tex2D<float>(texObj, kInd, nInd);
				dis += d * d;
			}
			if (dis_s[threadIdx.x] > dis)
			{
				dis_s[threadIdx.x] = dis;
				ind_s[threadIdx.x] = nInd;
			}
		}
		__syncthreads();
		for (int offset = BLOCK_DIM_X >> 1; offset > 0; offset >>= 1)
		{
			if (threadIdx.x < offset)
				if (dis_s[threadIdx.x] > dis_s[threadIdx.x ^ offset])
				{
					dis_s[threadIdx.x] = dis_s[threadIdx.x ^ offset];
					ind_s[threadIdx.x] = ind_s[threadIdx.x ^ offset];
				}
			__syncthreads();
		}
		if (threadIdx.x == 0)
			result[ans_id] = ind_s[0];
	}
	void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		assert(k * m <= (64 << 10) / sizeof(float));
		CHECK(cudaMemcpyToSymbol(const_mem, searchPoints, sizeof(float) * k * m));
		cudaArray *cuArray;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		CHECK(cudaMallocArray(&cuArray, &channelDesc, k, n));
		CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, referencePoints, sizeof(float) * k, sizeof(float) * k, n, cudaMemcpyHostToDevice));

		// 绑定纹理到cudaArray上
		struct cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// 设置纹理为只读
		struct cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		// 创建纹理对象
		cudaTextureObject_t texObj = 0;
		CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

		thrust::device_vector<int> results_d(m);
		{
			const int BLOCK_DIM_X = 1024;
			WuKTimer t1;
			cudaCallbackKernel<
				BLOCK_DIM_X><<<
				dim3(results_d.size() / m, m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				results_d.size(),
				texObj,
				thrust::raw_pointer_cast(results_d.data()));
		}
		thrust::copy(
			results_d.begin(),
			results_d.begin() + m,
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v6
void cudaCallback(
	int k,
	int m,
	int n,
	float *searchPoints,
	float *referencePoints,
	int **results)
{
	v6::cudaCallback(
		k,
		m,
		n,
		searchPoints,
		referencePoints,
		results);
}