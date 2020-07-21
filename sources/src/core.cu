#include "core.h"

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
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		int *tmp = (int *)malloc(sizeof(int) * m);
		// Iterate over all search points
		for (int mInd = 0; mInd < m; ++mInd)
		{
			float minSquareSum = INFINITY;
			int minIndex = 0;
			// Iterate over all reference points
			for (int nInd = 0; nInd < n; ++nInd)
			{
				float squareSum = 0;
				for (int kInd = 0; kInd < k; ++kInd)
				{
					const float diff = searchPoints[k * mInd + kInd] - referencePoints[k * nInd + kInd];
					squareSum += diff * diff;
				}
				if (minSquareSum > squareSum)
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
	static __global__ void
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
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - referencePoints[kInd + nInd * k];
				squareSum += diff * diff;
			}
			dis[nInd + mInd * n] = squareSum;
		}
	}
	static void cudaCallback(
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
			const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
			//WuKTimer t1;
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
			//WuKTimer t2;
			for (int i = 0; i < m; ++i)
				(*results)[i] = thrust::min_element(dis_d.begin() + n * i, dis_d.begin() + n * i + n) - dis_d.begin() - n * i;
		}
	}
}; // namespace v1
namespace v2
{
	__global__ void static get_dis_kernel(
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
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - referencePoints[kInd + nInd * k];
				squareSum += diff * diff;
			}
			dis[nInd + mInd * n] = squareSum;
		}
	}
	template <int BLOCK_DIM_X>
	static __global__ void
	get_min_kernel(
		const int result_size,
		const int m,
		const int n,
		const float *__restrict__ dis,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			const float squareSum = dis[nInd + blockIdx.y * n];
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
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
			const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
			//WuKTimer t1;
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
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v2
namespace v3
{
	template <int BLOCK_DIM_X>
	static __global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ searchPoints,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - referencePoints[kInd + nInd * k];
				squareSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
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
			//WuKTimer t1;
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
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v3
namespace v4
{
	static __global__ void
	mat_inv_kernel(
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output)
	{
		const int
			nInd = threadIdx.x + blockIdx.x * blockDim.x,
			kInd = threadIdx.y + blockIdx.y * blockDim.y;
		if (nInd < n && kInd < k)
		{
			const float a = input[nInd * k + kInd];
			output[nInd + kInd * n] = a;
		}
	}
	template <int BLOCK_DIM_X>
	static __global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ searchPoints,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - referencePoints[kInd * n + nInd];
				squareSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		thrust::device_vector<int> results_d(m);
		thrust::device_vector<float>
			s_d(searchPoints, searchPoints + k * m),
			r_d(k * n);
		{
			thrust::device_vector<float>
				rr_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
			//WuKTimer t1;
			mat_inv_kernel<<<
				dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
				dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
				k,
				n,
				thrust::raw_pointer_cast(rr_d.data()),
				thrust::raw_pointer_cast(r_d.data()));
		}
		{
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t1;
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
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v4
namespace v5
{
	template <int BLOCK_DIM_X>
	static __global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ searchPoints,
		cudaTextureObject_t texObj, //使用纹理对象
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - tex2D<float>(texObj, kInd, nInd);
				squareSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		if (n > 65536)
		{
			v4::cudaCallback(k, m, n, searchPoints, referencePoints, results);
			return;
		}
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
		thrust::device_vector<float>
			s_d(searchPoints, searchPoints + k * m);
		{
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t1;
			cudaCallbackKernel<
				BLOCK_DIM_X><<<
				dim3(results_d.size() / m, m),
				BLOCK_DIM_X>>>(
				k,
				m,
				n,
				results_d.size(),
				thrust::raw_pointer_cast(s_d.data()),
				texObj,
				thrust::raw_pointer_cast(results_d.data()));
		}
		thrust::copy(
			results_d.begin(),
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v5
namespace v6
{
	static __constant__ float const_mem[(64 << 10) / sizeof(float)];
	static __global__ void
	mat_inv_kernel(
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output)
	{
		const int
			nInd = threadIdx.x + blockIdx.x * blockDim.x,
			kInd = threadIdx.y + blockIdx.y * blockDim.y;
		if (nInd < n && kInd < k)
		{
			const float a = input[nInd * k + kInd];
			output[nInd + kInd * n] = a;
		}
	}
	template <int BLOCK_DIM_X>
	static __global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = const_mem[kInd + mInd * k] - referencePoints[kInd * n + nInd];
				squareSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		if (k * m > (64 << 10) / sizeof(float))
		{
			v4::cudaCallback(k, m, n, searchPoints, referencePoints, results);
			return;
		}
		CHECK(cudaMemcpyToSymbol(const_mem, searchPoints, sizeof(float) * k * m));
		thrust::device_vector<int> results_d(m);
		thrust::device_vector<float> r_d(k * n);
		{
			thrust::device_vector<float>
				rr_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
			//WuKTimer t1;
			mat_inv_kernel<<<
				dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
				dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
				k,
				n,
				thrust::raw_pointer_cast(rr_d.data()),
				thrust::raw_pointer_cast(r_d.data()));
		}
		{
			const int BLOCK_DIM_X = 1024;
			//WuKTimer t2;
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
			results_d.end(),
			*results = (int *)malloc(sizeof(int) * m));
	}
}; // namespace v6
namespace v7
{
	static __global__ void
	mat_inv_kernel(
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output)
	{
		const int
			nInd = threadIdx.x + blockIdx.x * blockDim.x,
			kInd = threadIdx.y + blockIdx.y * blockDim.y;
		if (nInd < n && kInd < k)
		{
			const float a = input[nInd * k + kInd];
			output[nInd + kInd * n] = a;
		}
	}
	template <int BLOCK_DIM_X>
	static __global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ searchPoints,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		ind_s[threadIdx.x] = 0;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - referencePoints[kInd * n + nInd];
				squareSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		thrust::device_vector<float>
			s_d(searchPoints, searchPoints + k * m),
			r_d(k * n);
		{
			thrust::device_vector<float>
				rr_d(referencePoints, referencePoints + k * n);
			const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
			//WuKTimer t1;
			mat_inv_kernel<<<
				dim3(divup(n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
				dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
				k,
				n,
				thrust::raw_pointer_cast(rr_d.data()),
				thrust::raw_pointer_cast(r_d.data()));
		}
		const int BLOCK_DIM_X = 1024;
		int numBlocks;
		CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&numBlocks,
			cudaCallbackKernel<BLOCK_DIM_X>,
			BLOCK_DIM_X,
			0));
		thrust::device_vector<int> results_d(m * divup(numBlocks, m));
		{
			//WuKTimer t2;
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
		*results = (int *)malloc(sizeof(int) * m);
		if (results_d.size() == m)
		{
			thrust::copy(
				results_d.begin(),
				results_d.end(),
				*results);
			return;
		}
		thrust::host_vector<int> results_tmp(results_d);
		for (int mInd = 0; mInd < m; ++mInd)
		{
			float minSquareSum = INFINITY;
			int minIndex = 0;
			// Iterate over all reference points
			for (int i = 0; i < results_tmp.size(); i += m)
			{
				const int nInd = results_tmp[i];
				float squareSum = 0;
				for (int kInd = 0; kInd < k; ++kInd)
				{
					const float diff = searchPoints[k * mInd + kInd] - referencePoints[k * nInd + kInd];
					squareSum += diff * diff;
				}
				if (minSquareSum > squareSum)
				{
					minSquareSum = squareSum;
					minIndex = nInd;
				}
			}
			(*results)[mInd] = minIndex;
		}
	}
}; // namespace v7
namespace v8
{
	static __global__ void
	mat_inv_kernel(
		const int k,
		const int n,
		const float *__restrict__ input,
		float *__restrict__ output)
	{
		const int
			nInd = threadIdx.x + blockIdx.x * blockDim.x,
			kInd = threadIdx.y + blockIdx.y * blockDim.y;
		if (nInd < n && kInd < k)
		{
			const float a = input[nInd * k + kInd];
			output[nInd + kInd * n] = a;
		}
	}
	template <int BLOCK_DIM_X>
	static __global__ void
	cudaCallbackKernel(
		const int k,
		const int m,
		const int n,
		const int result_size,
		const float *__restrict__ searchPoints,
		const float *__restrict__ referencePoints,
		int *__restrict__ result)
	{
		const int ans_id = blockIdx.x * gridDim.y + blockIdx.y;
		if (ans_id >= result_size)
			return;
		__shared__ float dis_s[BLOCK_DIM_X];
		__shared__ int ind_s[BLOCK_DIM_X];
		dis_s[threadIdx.x] = INFINITY;
		ind_s[threadIdx.x] = 0;
		for (int mInd = blockIdx.y, nInd = threadIdx.x + blockIdx.x * BLOCK_DIM_X;
			 nInd < n;
			 nInd += gridDim.x * BLOCK_DIM_X)
		{
			float squareSum = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				const float diff = searchPoints[kInd + mInd * k] - referencePoints[kInd * n + nInd];
				squareSum += diff * diff;
			}
			if (dis_s[threadIdx.x] > squareSum)
			{
				dis_s[threadIdx.x] = squareSum;
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
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		thrust::host_vector<int> results_tmp;
		int num_gpus = 0;
		CHECK(cudaGetDeviceCount(&num_gpus));
		if (num_gpus > n)
			num_gpus = n;
		if (num_gpus < 1)
			return v0::cudaCallback(k, m, n, searchPoints, referencePoints, results);
		if (n <= std::min(1 << 18, m << 10))
			return v7::cudaCallback(k, m, n, searchPoints, referencePoints, results);
#pragma omp parallel num_threads(num_gpus)
		{
			int thread_num = omp_get_thread_num(),
				thread_n = divup(n, num_gpus);
			float *thread_referencePoints = referencePoints + thread_num * thread_n * k;
			if (thread_num == num_gpus - 1)
			{
				thread_n = n - thread_num * thread_n;
				if (thread_n == 0)
					thread_n = 1, thread_referencePoints -= k;
			}
			CHECK(cudaSetDevice(thread_num));
			thrust::device_vector<float>
				s_d(searchPoints, searchPoints + k * m),
				r_d(k * thread_n);
			{
				thrust::device_vector<float>
					rr_d(thread_referencePoints,
						 thread_referencePoints + k * thread_n);
				const int BLOCK_DIM_X = 32, BLOCK_DIM_Y = 32;
				//WuKTimer t1;
				mat_inv_kernel<<<
					dim3(divup(thread_n, BLOCK_DIM_X), divup(k, BLOCK_DIM_Y)),
					dim3(BLOCK_DIM_X, BLOCK_DIM_Y)>>>(
					k,
					thread_n,
					thrust::raw_pointer_cast(rr_d.data()),
					thrust::raw_pointer_cast(r_d.data()));
			}
			const int BLOCK_DIM_X = 1024;
			int numBlocks;
			CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&numBlocks,
				cudaCallbackKernel<BLOCK_DIM_X>,
				BLOCK_DIM_X,
				0));
			thrust::device_vector<int> results_d(m * divup(numBlocks, m));
			{
				//WuKTimer t2;
				cudaCallbackKernel<
					BLOCK_DIM_X><<<
					dim3(results_d.size() / m, m),
					BLOCK_DIM_X>>>(
					k,
					m,
					thread_n,
					results_d.size(),
					thrust::raw_pointer_cast(s_d.data()),
					thrust::raw_pointer_cast(r_d.data()),
					thrust::raw_pointer_cast(results_d.data()));
			}
			int my_beg, my_end;
#pragma omp critical
			{
				my_beg = results_tmp.size();
				results_tmp.insert(results_tmp.end(), results_d.begin(), results_d.end());
				my_end = results_tmp.size();
			}
#pragma omp barrier
			for (int offset = (thread_referencePoints - referencePoints) / k; my_beg < my_end; ++my_beg)
				results_tmp[my_beg] += offset;
		}
		*results = (int *)malloc(sizeof(int) * m);
		for (int mInd = 0; mInd < m; ++mInd)
		{
			float minSquareSum = INFINITY;
			int minIndex = 0;
			// Iterate over all reference points
			for (int i = 0; i < results_tmp.size(); i += m)
			{
				const int nInd = results_tmp[i];
				float squareSum = 0;
				for (int kInd = 0; kInd < k; ++kInd)
				{
					const float diff = searchPoints[k * mInd + kInd] - referencePoints[k * nInd + kInd];
					squareSum += diff * diff;
				}
				if (minSquareSum > squareSum)
				{
					minSquareSum = squareSum;
					minIndex = nInd;
				}
			}
			(*results)[mInd] = minIndex;
		}
	}
}; // namespace v8
namespace v9
{
	float *searchPoints, *referencePoints;
	int k;
	struct DimCmp
	{
		int dim;
		bool operator()(int lhs, int rhs) const
		{
			return referencePoints[lhs * k + dim] < referencePoints[rhs * k + dim];
		}
	};
	struct KDTreeCPU
	{
		typedef float lf;
		int n;
		thrust::host_vector<int> p, dim;
		KDTreeCPU(int n)
			: n(n),
			  p(n << 2, -1),
			  dim(p)
		{
			thrust::host_vector<int> se(n);
			for (int i = 0; i < n; ++i)
				se[i] = i;
			build(se.begin(), se.end());
		}
		void build(
			thrust::host_vector<int>::iterator beg,
			thrust::host_vector<int>::iterator end,
			int rt = 1)
		{
			if (beg >= end)
				return;
			lf sa_max = -INFINITY;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				lf sum = 0, sa = 0;
				for (thrust::host_vector<int>::iterator it = beg; it != end; ++it)
				{
					lf val = referencePoints[(*it) * k + kInd];
					sum += val, sa += val * val;
				}
				sa = (sa - sum * sum / (end - beg)) / (end - beg);
				if (sa_max < sa)
					sa_max = sa, dim[rt] = kInd;
			}
			thrust::host_vector<int>::iterator mid = beg + (end - beg) / 2;
			std::nth_element(beg, mid, end, DimCmp{dim[rt]});
			p[rt] = *mid;
			build(beg, mid, rt << 1);
			build(++mid, end, rt << 1 | 1);
		}
		std::pair<lf, int> ask(int x, int rt = 1)
		{
			if (dim[rt] < 0)
				return {INFINITY, 0};
			lf d = searchPoints[x * k + dim[rt]] - referencePoints[p[rt] * k + dim[rt]];
			int w = d > 0;
			std::pair<lf, int> ans = ask(x, (rt << 1) ^ w);
			lf tmp = 0;
			for (int kInd = 0; kInd < k; ++kInd)
			{
				lf d = searchPoints[x * k + kInd] - referencePoints[p[rt] * k + kInd];
				tmp += d * d;
			}
			ans = min(ans, {tmp, p[rt]});
			if (ans.first > d * d - 1e-6)
				ans = min(ans, ask(x, (rt << 1) ^ w ^ 1));
			return ans;
		}
	};
	static void cudaCallback(
		int k,
		int m,
		int n,
		float *searchPoints,
		float *referencePoints,
		int **results)
	{
		if (k > 16)
			return v0::cudaCallback(k, m, n, searchPoints, referencePoints, results);
		v9::k = k;
		v9::searchPoints = searchPoints;
		v9::referencePoints = referencePoints;
		KDTreeCPU kd(n);
		*results = (int *)malloc(sizeof(int) * m);
		printf("\n\n---\nsearch on KD-Tree: ");
		{
			WuKTimer timer;
			for (int i = 0; i < m; ++i)
				(*results)[i] = kd.ask(i).second;
		}
		printf("---\n\n");
	}
} // namespace v9
struct WarmUP
{
	WarmUP(int k, int m, int n)
	{
		void (*cudaCallback[])(int, int, int, float *, float *, int **) = {
			v0::cudaCallback,
			v1::cudaCallback,
			v2::cudaCallback,
			v3::cudaCallback,
			v4::cudaCallback,
			v5::cudaCallback,
			v6::cudaCallback,
			v7::cudaCallback,
			v8::cudaCallback,
			v9::cudaCallback};
		float *searchPoints = (float *)malloc(sizeof(float) * k * m);
		float *referencePoints = (float *)malloc(sizeof(float) * k * n);

#pragma omp parallel
		{
			unsigned seed = omp_get_thread_num(); //每个线程使用不同的随机数种子
#pragma omp for
			for (int i = 0; i < k * m; ++i)
				searchPoints[i] = rand_r(&seed) / double(RAND_MAX); //使用线程安全的随机数函数
#pragma omp for
			for (int i = 0; i < k * n; ++i)
				referencePoints[i] = rand_r(&seed) / double(RAND_MAX);
		}

		for (int i = 0; i < sizeof(cudaCallback) / sizeof(cudaCallback[0]); ++i)
		{
			int *result;
			cudaCallback[i](k, m, n, searchPoints, referencePoints, &result);
			free(result);
		}
		free(searchPoints);
		free(referencePoints);
	}
};
struct BenchMark
{
	BenchMark(int k, int m, int n)
	{
		void (*cudaCallback[])(int, int, int, float *, float *, int **) = {
			v0::cudaCallback,
			v1::cudaCallback,
			v2::cudaCallback,
			v3::cudaCallback,
			v4::cudaCallback,
			v5::cudaCallback,
			v6::cudaCallback,
			v7::cudaCallback,
			v8::cudaCallback};
		float *searchPoints = (float *)malloc(sizeof(float) * k * m);
		float *referencePoints = (float *)malloc(sizeof(float) * k * n);

#pragma omp parallel
		{
			unsigned seed = omp_get_thread_num(); //每个线程使用不同的随机数种子
#pragma omp for
			for (int i = 0; i < k * m; ++i)
				searchPoints[i] = rand_r(&seed) / double(RAND_MAX); //使用线程安全的随机数函数
#pragma omp for
			for (int i = 0; i < k * n; ++i)
				referencePoints[i] = rand_r(&seed) / double(RAND_MAX);
		}
		printf("\n\nStart benchmark with (k, m, n) = (%d, %d, %d):\n\n", k, m, n); //开始benchnmark
		for (int i = 0; i < sizeof(cudaCallback) / sizeof(cudaCallback[0]); ++i)
		{
			int *result;
			printf("Version %d: ", i);
			{
				WuKTimer t1;
				cudaCallback[i](k, m, n, searchPoints, referencePoints, &result);
			}
			free(result);
			fflush(stdout);
		}
		printf("\nFinish benchmark with (k, m, n) = (%d, %d, %d).\n\n", k, m, n);
		free(searchPoints);
		free(referencePoints);
	}
};
static WarmUP warm_up(1, 1, 1 << 20);
static BenchMark
	benchmark8(3, 1, 1 << 24),
	benchmark9(16, 1, 1 << 24),
	benchmark10(3, 1024, 1 << 20),
	benchmark11(16, 1024, 1 << 20);
void cudaCallback(
	int k,
	int m,
	int n,
	float *searchPoints,
	float *referencePoints,
	int **results)
{
	v8::cudaCallback(
		k,
		m,
		n,
		searchPoints,
		referencePoints,
		results);
}