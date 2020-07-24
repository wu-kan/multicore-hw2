# multicore hw2

<style>html body{ font-family: "华文中宋"; }</style>

|   学校   |         学院         |             专业             | 年级 |   学号   | 姓名 |
| :------: | :------------------: | :--------------------------: | :--: | :------: | :--: |
| 中山大学 | 数据科学与计算机学院 | 计算机科学与技术（超算方向） |  17  | 17341163 | 吴坎 |

[toc]

计算高维空间中的最近邻（nearest neighbor）

- 输入：查询点集，参考点集，空间维度
- 输出：对每个查询点，输出参考点集中最近邻的序号

实验中分别针对 CPU、GPU、多 GPU 三种计算环境上实现了上述实验要求，并实现了若干种优化版本进行对比，同时也使用空间划分结构 KD-Tree 对查询进行加速。实验使用 TA 提供的八组测试数据和自己构造的四组更大的测试数据对各个版本进行 benchmark 测试，并从结果分析得到了一些影响本次程序性能的因素。

如果要运行我的实验程序，可以终端直接在当前目录运行如下指令：

```bash
#chmod 777 ./RUNME.sh # 可用于解决有时出现文件权限错误的问题
./RUNME.sh | tee screen.log
```

该条语句会自动编译并运行实验程序，同时保存屏幕日志到 `screen.log` 文件中。我的运行结果可以直接看当前目录下的 `screen.log`。

## 介绍程序整体逻辑，包含的函数，每个函数完成的内容。对于核函数，应该说明每个线程块及每个线程所分配的任务

我分别实现了串行的 v0 版本、 基于 CUDA 的 v1\~v8 版本、使用 CPU 上的 KD-Tree 的 v9 版本、使用 GPU 上的 KD-Tree 的 v10 版本。如下，如果要在它们之间进行切换，只需要修改 `sources/src/core.cu` 最后几行 `cudaCallback` 函数中实际调用的版本即可。

```cpp
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
```

### v0

v0 版本是 TA 提供的串行版本，用于检验运行结果的正确性。其逻辑如下：

1. 枚举查询点集所有的点
   1. 设查询点下标 `mInd`
   2. 初始化当前最优距离为无穷大
   3. 枚举参考点集所有点
      1. 设参考点下标 `nInd`
      2. 计算查询点和参考点的距离
         - 为了减少一次不必要的开方运算，这里保存的是距离的平方
      3. 若查询点到参考点距离比当前最优距离更优
         1. 更新当前最优距离
         2. 记录当前参考点坐标 `nInd`
   4. 保存最优参考点下标到答案向量中
2. 返回结果

v0 版本中仅有一个 `cudaCallback` 函数作为外部调用的接口。

### v1

v1 版本是我实现的 CUDA 上的 baseline 版本，下面对其逻辑稍作解释。

首先，因为串行的 v0 版本中在枚举参考点集所有下标 `nInd` 时有明显的循环依赖性（当前最优距离依赖于之前的循环迭代结果），因此不可以直接移植到多线程上。为此，我将求解的过程划分成如下两个阶段，从而使每个阶段都没有循环依赖：

1. 计算查询点到参考点的距离矩阵 `dis`，大小为 $m\times n$
   - 为了减少一次不必要的开方运算，这里保存的是距离的平方
2. 对于距离矩阵 `dis` 的每一行，求出其最小值点下标，并保存在答案向量对应位置

为了简化开发过程，在这个 baseline 版本中我使用了 [thrust](https://docs.nvidia.com/cuda/thrust/index.html) 库进行显存管理，其优点是提供了接近于 STL 的高级抽象，同时又是**运行时零开销**的。

对于第一阶段计算距离矩阵，我实现了一个核函数 `get_dis_kernel`，将线程按照 $32\times 32$ 分块并一一映射到距离矩阵的每个位置上。作为 baseline 版本，此处每个线程都直接使用串行逻辑，线程之间没有交互。由于第一阶段程序用时实际上相当小（见下），此处暂且不对 blockDim 的大小进行调整，直接按照经验选择了 1024。

对于第二阶段求距离矩阵最小点，实际上是一个非常经典的区间规约问题，我已经在 CUDA-8 这一节课上学过对应的优化方法。当然，还是有一些小的区别，课件上的规约操作是求和，而我这里是区间最小值点下标。恰好 thrust 库中也有封装好的算法 `thrust::min_element` ，我这里直接用其作为实现，也可作为后续优化的标杆。

v1 版本中有一个 `cudaCallback` 函数作为外部调用的接口，还有一个 `get_dis_kernel` 核函数用于计算上文提到的距离矩阵。

### v2

注意到 v1 版本中第二阶段的规约算法占据了九成以上的时间，而且多次使用规约算法时，第二层规约的元素数量会远小于第一层元素的数量，猜测没有充分利用显卡的算力因此效果很差。因此我在 v2 版本中增加了一个核函数 `get_min_kernel`，按照 CUDA-8 课程的 ppt，使用 share memory 实现了一个 block 内部的树形规约算法。显卡上使用 share memory 进行树形规约的算法已经非常经典，此处不做详细展开了。

调用核函数时，每个 block 内有 1024 个线程（显卡 v100 的上限）按照一维分布；而每个 grid 内有 m 个 block，每个 block 对应一个查询点的求解。程序运行时，每个线程先读入一部分距离向量的元素（此处是跳跃读取，从而使在运行的时候访存连续）并在线程内部进行规约；随后 block 内部的线程使用树形规约求出最小值和对应下标，并保存在 0 号线程中，最后由 0 号线程写回结果。

v2 版本相对于 v1 版本增加了一个 `get_min_kernel` 核函数，实现了上文提到的树形规约算法。

### v3

重新看一下 v2 版本，可以发现这个距离矩阵其实是不必要去计算的，我大可以边计算查询点到参考点的距离边规约求最小值，于是得到了 v3 版本，从而减少了一次启动核函数的开销，和一次对距离矩阵的读写。

v3 版本在启动核函数时，线程在 block 和 grid 中的分布方式与 v2 版本中 `get_min_kernel` 的方式完全相同。每个线程先串行计算查询点和一部分参考点的距离并规约，随后在 block 内部执行树形规约。

v3 版本中有一个 `cudaCallback` 函数作为外部调用的接口，还有一个 `cudaCallBackKernel` 核函数执行上文提到的求距离并执行规约过程。

### v4

注意到 TA 提供的接口中，点的存储方式是 Array of Structures（AoS，点的每个维度上的坐标值存储连续），这种存储方式在串行上有空间局部性，比较适合 CPU 的缓存方式，但是在 GPU 上访问的时候却会导致严重的访存不连续问题，且会随着 k 的增加不断的变严重（比如，在 v2 版本中，k 从 3 增加到 16 时，求解距离矩阵的时间从 0.5ms 增加到了 5ms，增加的时间远多于显卡带宽读入对应数据所需的时间）。

我在 v3 版本上额外增加了一个 `mat_inv_kernel` 核函数，将 TA 提供的参考点坐标进行了一次转置（如果把输入看成一个矩阵的话），从而让点的存储方式变成 Structure of Arrays（SoA，每个维度上点的坐标值存储连续），这样 `cudaCallBackKernel` 对参考点的访存就连续了。

虽然说矩阵转置的最优实现是使用 shared memory 分块并合并访存，实际上这里不使用 shared memory 运行的时间也只有 0.5ms。并且由于 k 很小（3\~16），因此这个矩阵非常「细长」，使用 shared memory 的分块方法未必就有好的效果。

v4 版本中有一个 `cudaCallback` 函数作为外部调用的接口，一个 `cudaCallBackKernel` 核函数执行上文提到的求距离并执行规约过程，还有一个 `mat_inv_kernel` 核函数用于将参考点矩阵进行转置。

### v5

v5 版本尝试用 2d texture memory 优化 v3 版本中对参考点的不连续访存，从而避免像 v4 版本中一样做一次额外的矩阵转置。

v5 版本中有一个 `cudaCallback` 函数作为外部调用的接口，还有一个 `cudaCallBackKernel` 核函数执行上文提到的求距离并执行规约过程。注意到使用的 v100 显卡计算能力为 7.0 ，对应的 [2d texture memory 中矩阵的宽度不能超过 65536](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability)，因此 v5 版本仅能处理 $n\le65536$ 大小的输入。如果超过这个限制，v5 版本会直接调用 v4 版本。

### v6

v6 版本在 v4 版本上使用 constant memory 优化对查询点的访问。

v6 版本中有一个 `cudaCallback` 函数作为外部调用的接口，还有一个 `cudaCallBackKernel` 核函数执行上文提到的求距离并执行规约过程。注意到 const memory 有大小为 64k 的限制，因此 v6 版本仅能处理 $k\times m\le16384$ 大小的输入。如果超过这个限制，v6 版本会直接调用 v4 版本。

### v7

注意到 v3\~v6 版本都没有处理这个问题：每个查询点仅由一个 block 完成，如果查询点非常少的时候，启动的 block 也非常少，这将导致很多 SM 空置。

v7 版本在 v4 版本的基础上，对每个查询点将有多个 block 进行查询，启动的 block 数量从 cuda 中的 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 函数获得，从而保证启动的`cudaCallback`核函数能够使 SM 满载。（求矩阵转置的核函数跑得很快，没有必要用这个了）

由于多个 block 会求到多个结果，这里多个结果拷贝回 CPU 进行二级规约。由于实际上二级规约的元素非常少，在 CPU 上进行规约的收益高于 GPU。

v7 版本中有一个 `cudaCallback` 函数作为外部调用的接口，一个 `cudaCallBackKernel` 核函数执行上文提到的求距离并执行规约过程，还有一个 `mat_inv_kernel` 核函数用于将参考点矩阵进行转置。

### v8

v8 版本在 v7 版本的基础上增加了多卡的实现。具体来说，就是将参考点集均分到每张显卡上，这样每个显卡执行对参考集一个子集的求解，然后在对每张显卡的结果再进行一次规约。

v8 版本中有一个 `cudaCallback` 函数作为外部调用的接口，一个 `cudaCallBackKernel` 核函数执行上文提到的求距离并执行规约过程，还有一个 `mat_inv_kernel` 核函数用于将参考点矩阵进行转置。

### v9

v9 版本在 CPU 上构建了一棵 KD-Tree 用于加速最近邻求解。

v9 中有一个 `cudaCallback` 函数作为外部调用的接口，还有一个 `struct KDTreeCPU`，其构造函数即为建树过程；还有一个接口 `ask` ，返回查询点到参考点的最小距离及下标。

### v10

v10 版本在 CPU 上构造了一棵 KD-Tree，然后拷贝到 GPU 上，对最近邻的查询在 GPU 上。

v10 中有一个 `cudaCallback` 函数作为外部调用的接口，还有一个 `struct KDTreeGPU`，其构造函数即为建树过程；还有一个接口 `range_ask` ，返回查询点集合到参考点的最小距离及下标。

## 解释程序中涉及哪些类型的存储器（如，全局内存，共享内存等），并通过分析数据的访存模式及该存储器的特性说明为何使用该种存储器

对于运行在 CPU 上的版本，缓存结构在 C 语言的层面几乎是不可见的，通常来说数据是被存放在内存上，CPU 通过缓存读取；少部分经常访问的数据编译器会将其存放在寄存器中。在 GPU 上的版本，我主要使用了 global memory、shared memory、texture memory、constant memory。

v2~v8 版本中，我在树形规约算法中使用了 shared memory。shared memory 是容量很小，但是低延迟的 on-chip memory，比 global memory 拥有高得多的带宽，可以把它当做可编程的 cache。物理上，每个 SM 包含一个当前正在执行的 block 中所有 thread 共享的低延迟的内存池。shared memory 使得同一个 block 中的 thread 能够相互合作，重用 on-chip 数据，并且能够显著减少 kernel 需要的 global memory 带宽。在本问题中，shared memory 可以被同一个 block 中的线程访问，从而实现线程间的通信，完成多线程并行的的树形规约算法。

v4 版本中，我将 TA 提供的参考点坐标进行了一次转置（如果把输入看成一个矩阵的话），从而让点的存储方式变成 Structure of Arrays（SoA，每个维度上点的坐标值存储连续），这样对 global memory 的访存就连续了。

v5 版本中，我使用 2d texture memory 加速对参考点的访问。texture memory 不需要满足 global memory 的合并访问条件也可以优化邻域上的数据读取。

v6 版本中，我使用 constant memory 加速对查询点的访问。对 constant memory 的单次读操作可以广播到同个半线程束的其他 $15$ 个线程，这种方式产生的内存流量只是使用全局内存时的 $\frac{1}{16}$。同时硬件主动把 constant memory 缓存在 GPU 上，在第一次从常量内存的某个地址上读取后，当其他半线程束请求同一个地址时，那么将命中缓存，这同样减少了额外的内存流量。

## 针对查询点集为 1 个点及 1024 个点给出两个版本，并说明设计逻辑（例如，任务分配方式）的异同。如两个版本使用同一逻辑，需说明原因

刚看到作业题的时候，我想的是「先实现一个查询点的逻辑，然后扩展到多个点上」。然而实际上在实现的时候，我却先实现了在多个查询点上效果良好而单点效果很差的 v2\~v6 版本，然后再增加数据的并行性得到了 v7\~v8 版本，优化了其在 1 个查询点上的表现。

原因其实也很明显：多个查询点的版本天生就有很高的并行性，每个查询其实是互不影响的，每个任务分给一个 block 即可，可以同时进行，这样能够充分利用显卡上的大量线程。然而只有一个查询点时，实际上只启动了一个 block，造成了大量计算资源的浪费。

因此，设计的 v7 版本则将之前每个 block 的任务进一步细分，每个 block 只执行原先任务的一个子集；得到的结果经过二次规约之后才是最终的结果。

v8 版本和 v7 版本的任务划分方式完全相同，只不过要先通过多线程划分一次任务，而各线程得到结果在规约时也要设置临界区防止读写冲突。

## 请给出一个基础版本（baseline）及至少一个优化版本。并分析说明每种优化对性能的影响

我一共实现了 v0\~v10 共 11 个版本。由于使用数据结构 KD-Tree 的 v9、v10 版本比较特殊，有一个建树的过程，其 benchmark 方式和其他版本不同，因此对其性能的测试和评价在最后一部分进行（见下文）。

### 测试方法

测试使用的编译指令如下（打开所有常见编译优化）：

```cmake
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -use_fast_math -Xcompiler -Ofast -Xcompiler -fopenmp")
```

TA 提供的几组测试数据如下：

| id  |  k  |  m   |   n   |
| :-: | :-: | :--: | :---: |
|  0  |  3  |  1   |   2   |
|  1  |  3  |  2   |   8   |
|  2  |  3  |  1   | 1024  |
|  3  |  3  |  1   | 65536 |
|  4  | 16  |  1   | 65536 |
|  5  |  3  | 1024 | 1024  |
|  6  |  3  | 1024 | 65536 |
|  7  | 16  | 1024 | 65536 |

由于我最后优化的版本在最大的测试数据上已经跑到了 2.689ms，而此时系统噪声对结果的影响已经不能忽视，上述几组数据实际上并不足够反映我优化的效果，我也构造了四组更大的测试：

| id  |  k  |  m   |    n     |
| :-: | :-: | :--: | :------: |
|  8  |  3  |  1   | 16777216 |
|  9  | 16  |  1   | 16777216 |
| 10  |  3  | 1024 | 1048576  |
| 11  | 16  | 1024 | 1048576  |

其中，第 8、9 组数据是对第 3、4 组数据的加强，适用于衡量 1 个参考点集的情况；第 10、11 组数据是对第 6、7 组数据的加强，适用于衡量 1024 个参考点集的情况。我将 benchmark 的过程封装成 class，只需要在 `sources/src/core.cu` 最后几行中的添加相关内容即可运行。

```cpp
static WarmUP warm_up(1, 1, 1 << 20);
static BenchMark
	benchmark8(3, 1, 1 << 24),
	benchmark9(16, 1, 1 << 24),
	benchmark10(3, 1024, 1 << 20),
	benchmark11(16, 1024, 1 << 20);
```

此外，实验中发现对于一张显卡上首次运行的核函数会有 30ms 左右的冷启动时间。为排除这部分对于 benchmark 的影响，我也封装了一个 `WarmUP` class，提前进行「热身」操作。

### 测试环境

- [Intel(R) Xeon(R) Gold 6242 CPU](https://ark.intel.com/content/www/cn/zh/ark/products/192440/intel-xeon-gold-6242-processor-22m-cache-2-80-ghz.html)@2.80GHz \*2
- 128GB Memory
- NVIDIA(R) Tesla(R) V100 32GB \* 4

```bash
$ cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
     32  Intel(R) Xeon(R) Gold 6242 CPU @ 2.80GHz
$ cat /proc/meminfo | grep Mem
MemTotal:       131660272 kB
MemFree:        97517704 kB
MemAvailable:   126997860 kB
$ nvidia-smi
Sat Jul 18 09:05:07 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:1A:00.0 Off |                    0 |
| N/A   38C    P0    41W / 300W |      0MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:3D:00.0 Off |                    0 |
| N/A   37C    P0    42W / 300W |      0MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:89:00.0 Off |                    0 |
| N/A   37C    P0    42W / 300W |      0MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:B2:00.0 Off |                    0 |
| N/A   38C    P0    41W / 300W |      0MiB / 32510MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 测试结果

可以查看当前目录下的 `screen.log`。

#### v8（v7）版本对第 0\~7 组数据的结果

由于这几组数据都没有让单卡跑满，因此 v8 版本实际上是直接调用 v7 版本实现的，其结果如下（单位为 ms，下同）：

|   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.929 | 0.620 | 0.600 | 1.832 | 3.294 | 0.403 | 0.925 | 2.689 |

可以看到我的程序跑的还是相当快的。而且运行结果受系统噪声过大，甚至出现了 1 个参考点的数据 4 运行时间比 1024 个参考点的数据 7 还要大的情况，因此此处结果仅供参考，不用于衡量实际测试结果。

#### v0\~v8 版本对第 8\~11 组数据的结果

| version |    8    |    9    |    10    |    11     |
| :-----: | :-----: | :-----: | :------: | :-------: |
|   v0    | 46.044  | 201.456 | 2804.439 | 12104.106 |
|   v1    | 48.321  | 236.344 | 384.369  |  488.961  |
|   v2    | 61.228  | 262.765 |  40.041  |  148.049  |
|   v3    | 58.445  | 338.374 |  14.915  |  128.129  |
|   v4    | 97.289  | 410.671 |  17.890  |  46.449   |
|   v5    | 107.971 | 399.344 |  18.036  |  47.162   |
|   v6    | 67.248  | 346.049 |  17.656  |  64.690   |
|   v7    | 58.224  | 343.083 |  20.012  |  48.639   |
|   v8    | 25.650  | 100.346 |  9.971   |  17.292   |

对上述结果做可视化如下：

![可视化]

上图中纵轴是程序的运行时间，横轴对应各个版本，不同颜色的折线代表不同数据集的情况。由于 v0 版本是 CPU 上的串行版本，在部分测试数据上运行时间非常久，超过了纵轴的范围，为了图像的直观性其值在图中不显示。

### 结果分析

从可视化的折线图中可以看出，无论哪一组数据，多卡上的版本都有最优的性能，多卡加速还是非常有效的。下面再对单卡上的 v1\~v7 版本进行分析。

首先，对于 1 个查询点的数据来看，直接使用 `thrust::min_element` 库的 v1 版本就是最优秀的。可以看到，`thrust` 库能够号称「Code at the speed of light」，还是做了很多优化的。然而，由于这个函数实际上并没有针对多个查询作优化，本身也是线程不安全的（直接 `#pragma omp parallel for` 会报错），因此其对 1024 个查询点的结果非常不佳。根据测试数据来看，`{k, m, n} == {3, 1024, 65536}` 时两个阶段所用时间分别为 `0.556032ms, 53.586945ms`；`{k, m, n} == {16, 1024, 65536}` 时两个阶段所用时间分别为 `5.749760ms, 53.504002ms`。这说明九成计算时间都在第二阶段，也为后面的优化方法提供了思路和指导。

直接调库的结果未免有点「胜之不武」，因此在 v2 版本中用我自己的代码代替了 `thrust::min_element` ，也同时对多个查询进行优化。当然 v2 版本也有一些问题。v2 版本区间规约的核函数只启动了一次，实际上只有一层规约就得到最后结果，而且一个查询点实际上只由一个 block 负责求解，对于 1024 个点的版本来说还好，对于 1 个点的查询实际上只启动了 1 个 block，显然远远不能充分利用显卡的多个 SM！不过相对于 v1 版本，v2 版本中区间规约的时间减少到 3 毫秒以内，相对于 v1 版本已经提高太多了。

v3 版本相当于将 v2 版本中两个核函数融合，因此也继承了 v2 版本的缺点，在查询点只有 1 个的时候实际上只启动了 1 个 block。从运行时间上来看，对于 `{k, m, n} == {16, 1024, 65536}` 有一定的效果（11ms 减为 9ms），而在输入点只有 1 个的时候时间却增加了非常多（因为求距离时的并行性大大降低了）。

v4 版本相对于 v3 版本，增加的矩阵转置过程本身只用了不到 0.5ms，却让程序在 `{k, m, n} == {16, 1024, 65536}` 的总运行时间从 9ms 下降到 4ms，效果非常明显！然而，由于每个 block 对只对应一个查询点，因此没有办法通过同样的方法调整查询点的访存。

由于 v5 版本中使用了 texture memory，所以上面第 8\~11 组数据上的结果实际上是调用 v4 版本实现的。而从 `{k, m, n} == {16, 1024, 65536}` 这组测试数据来看，v5 版本的运行时间大概在 6ms 左右，相对于 v3 版本有提升但是不如 v4 版本好。这说明纹理内存的访问速度还是比不上合并访存之后的 global memory。

v6 版本在 v4 版本的基础上使用 constant memory 优化了对查询点的访问，在 v4 版本上少许提升了性能。然而 v6 版本也有缺陷：constant memory 有 64K 的限制，因此只能处理 $k\times m\le 16384$ 的输入。因此后续优化仍然是在 v4 版本上做的。

v7 版本在 v4 版本上增加了算法的并行性，对 SM 的利用率更高，因此同样在 v4 版本的基础上提升了性能表现，同时效果优于 v6。

## 选做：使用空间划分数据结构加速查询（如 KD-Tree, BVH-Tree）

我实现了使用 CPU 上的 KD-Tree 的 v9 版本、使用 GPU 上的 KD-Tree 的 v10 版本。

KD-Tree 是每个节点都为 k 维点的二叉树。所有非叶子节点可以视作用一个超平面把空间分割成两个半空间。节点左边的子树代表在超平面左边的点，节点右边的子树代表在超平面右边的点。最邻近搜索用来找出在树中与输入点最接近的点，在 KD 树上最邻近搜索的过程如下：

1. 从根节点开始，递归的往下移。往左还是往右的决定方法与插入元素的方法一样（如果输入点在分区面的左边则进入左子节点，在右边则进入右子节点）。
2. 一旦移动到叶节点，将该节点当「当前最佳点」。
3. 解开递归，并对每个经过的节点运行下列步骤：
   1. 如果当前所在点比当前最佳点更靠近输入点，则将其变为当前最佳点。
   2. 检查另一边子树有没有更近的点，如果有则从该节点往下找。
4. 当根节点搜索完毕后完成最邻近搜索。

首先我实现了纯 CPU 上的 v9 版本，此版本用于检验算法的正确性，且通过了[在线评测网站的测试](https://vjudge.net/solution/26421345)。这个版本中建树有一些小技巧：树的根节点是 1 号点，而节点 `i` 的左孩子节点是 `i << 1`，右孩子 `i << 1 | 1`，这样直接用数学公式就可以推出每个孩子的编号，减少了寻址过程，而空间也没有浪费特别多（对于 n 个参考点的集合，树上顶点编号不超过 $n\times 4$）。此外，在建树时我选择方差最大的维度作为分割，并选择这一维的中位数作为分割点，这样生成的 KD 树是平衡的，每个叶节点的高度都十分接近，并且将空间切割更均匀，在查找时更容易被剪枝。

然后我实现了 GPU 上的 v10 版本。此版本使用 GPU 的多线程加速查找过程，每个线程对应一个查询点，使得其可以同时查询原问题中的多个顶点；其他算法与 CPU 上的一致。

由于使用 KD-Tree 会有一个比较耗时的建树过程，在本问题中如果直接同其他版本比较的话其实是不太公正的，应当把建树和查询的过程分开来看。这样评价也有现实意义，例如某 MOBA 游戏中玩家释放的技能自动定位到最近的野怪，此时单次查询时的「延迟感」对游戏体验的影响非常大，此时使用 KD-Tree 可以大大减少这个延迟，而建树的过程可以放在游戏加载的时候，相对不那么重要。此外，对于 KD-Tree 版本的代码，我发现当数据范围过大时，其建树时间会漫长到让人难以接受。由于时间所限，我使用 `{k, m, n} == {3, 1024, 65536}` 和 `{k, m, n} == {16, 1024, 65536}` 这两组数据评价最终的效果。

这里同样放出 CPU 上的 v0 版本和 GPU 上的 v7 版本进行对照。

| 版本 | $k=3$ 查询时间 | $k=3$ 总时间 | $k=16$ 查询时间 | $k=16$ 总时间 |
| :--: | :------------: | :----------: | :-------------: | :-----------: |
|  v0  |    176.126     |   176.126    |     705.098     |    705.098    |
|  v7  |     0.925      |    0.925     |      2.689      |     2.689     |
|  v9  |     1.073      |    19.804    |    2419.928     |   2460.956    |
| v10  |     0.433      |    18.755    |     24.292      |    69.975     |

从结果来看，在维数比较低的时候，创建的 KD 树能够有效减少查询时的时间，然而在高维情况下这个空间划分结构显得花哨且不实用，可以说是「维数灾难」了。在高维空间中，KD 树并不能做很高效的最近邻搜索（难以触发搜索时的剪枝条件），从而使大部分的点都会被查询，最终算法效率也不会比全体查询一遍要好到哪里去。

[可视化]: data:image/png;base64,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
