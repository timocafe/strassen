# Heterogenous computing 

Heterogenous computing is the art to utilize multiple "device" simultaneously. It becomes very popular the last decade by the intensive 
usage of the GPU into HPC. But more interesting, embedded devices have introduced new microchips which are super powerful and reach
the TerraFlops (Apple M1, etc ...). Such devices have many compute units: CPUs, GPUs and TPUs. Contrary to big HPC node, all the
ship are integrated into the same socket with unify space memory. Therefor, the data transfert (bottleneck in HPC - PCI or NVlink)
can be overrided. 

In this tiny note, I explore the possibility of a Jetson TX2 device, and the faculty to run all units simultaneously (CPU/GPU) on
a well know example of HPC: the floating point matrix multiplication single precision: SGEMM, which is the keystone of the Deep Learning.
 
# A SGEMM needs a SGEMM ...

The most popular library of linear algebra has been released in 1978, BLAS. Since the API does not
have changed at all, and vendors and universities make tough work to get the highest performance.
In this post, I will re-implement a SGEMM following the classical and the STRASSEN scheme using tiling and recursive method.

The combination of the two approaches is interesting because the recursive process will always end up in one the basic following operations: 
BLAS3 - GEMM or BLAS1 - SAXPY (vector operations) on a single block of memory: a tile. Just need to specify the execution device to
make heterogenous computing.

# What do we need ?

Two thinks:
1. A good linear algebra implementation for CPU and GPU. For CPU, I have selected the [Eigen library](http://eigen.tuxfamily.org/index.php?title=Main_Page) 
, and CUBLAS for the CUDA execution. For the multithreading
my choice is [Intel TBB](https://github.com/oneapi-src/oneTBB),
Intel did a lof of effort since a decade to release tool for multithreading (I stop to make my own threading cooking). They did an amazing work. 
Here I will focus on TBB::Task programming and Pool TBB::allocator. 
2. The programming focus on the implementation of vector/matrix and tile_matrix. Here nothing fancy basic c++14, with important usage of 
the move semantic and a sweet allocator to encapsulate the unify_memory allocator of NVidia.

# Jetson device

The platform test is Jetson TX-2, force to run on 8 cores ARM A-57 under the mode `nvpmodel -m 0` and 256 core NVIDIA Pascal. I measure the performance for 
a square matrix of size (8192, 8192) the following Flops number (from my benchmark not from the SDK): 

| nvpmodel | CPUs   (8 cores) | GPUs          |
|----------|------------------|---------------|
|      0   |  160.  [GFlop/s]   | 722. [GFlop/s] |

The memory stream benchmark indicates a bandwidth of 40. [GByte/s].
So can we make nice mixup of CPU and GPU, and get higher performance like 900 [Gflops] :rofl: ?

# Matrice multiplication algorithms

Two algorithms of matrix multiplication are implemented are: the [classical](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm) one
and the [Strassen](https://en.wikipedia.org/wiki/Strassen_algorithm) one. For both I apply a divide and conquer algorithm, and stop the recursive process when 
the wanted tile size is reached. For simplification all my matrices are squared and multiple of 2<sup>n</sup>. 

# Implementation details

## Memory

Implement matrix and vector classes is a long and endless story, it exists an infinite number of possibilities, but for a divide and conquere algorithm,
it is the keypoint, the memory management must be efficient:

1. The [vector class](https://github.com/timocafe/strassen/blob/main/memory/vector.h) is based on the composition of a std::vector<T>, 
 I provide a C++11 allocator to encapsulate the cuda-unify malloc. I remind minimum
 C++11 (much simpler than the C03 allocator, it consists to declare the functions `T *allocate(std::size_t size)`  and `void deallocate(T *ptr, std::size_t n)`
 into a class). The full description  of the allocator is [here](https://github.com/timocafe/strassen/blob/main/memory/allocator.h). The class also implements
 the overload of the operator `+=`, `-=`, `+`, and `-` (alias of SAXPY) based on **Eigen**. These operators will be used by the matrix class. 
 Moreover I did not make an additional binding for the GPU for the vector operations, because all these operations are limited by the bandwidth, 
 which is shared between the devices (CPU/GPU).
2. The [matrix class](https://github.com/timocafe/strassen/blob/main/memory/matrix.h) is build upon the previous vector class. The column order has been 
 chosen because integration with BLAS3-SGEMM API is much simpler (1978 API :confounded:). Operators `+=`, `-=`, `+`, and `-` forward to the vector class. And the operator '*'
 is bind to **Eigen** for the CPU and **cuBlas** for the GPU.
3. Finally the [tile matrix](https://github.com/timocafe/strassen/blob/main/memory/tile_matrix.h) is build upon a `std::vector` of the previous
 matrix class using column order. This approach is well compatible with divide and conquers algorithm. Moreover the `move` semantics of modern C++
 is mandatory to move block without copy.

## Algorithms

The [classical algorithm](https://github.com/timocafe/strassen/blob/main/algo/classic.h) is recursive. To perform well the multithreading programming
task from [Intel-TBB](https://link.springer.com/chapter/10.1007/978-1-4842-4398-5_10) has been selected. I do a short description of the [classical](https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm) algorithm with TBB task based on the wikipedia page.

```C++
template <class T>
auto classic(const tile_matrix<T> &A, const tile_matrix<T> &B,
             const uint32_t lbs = 64) {
  const uint32_t n = A.rows();
  const uint32_t k = A.rows() / 2;
  if (lbs == n) // limit blocksize end of the recursive process
    return std::move(mul(A, B));
  // allocate new tile matrices twice smaller
  tile_matrix<T> A11(k, k, lbs);
  tile_matrix<T> B11(k, k, lbs);
  …
  // middle tile
  const uint32_t mt = A.tile_rows() / 2;

  // used move semantic inside
  copy_block(A11, A, 0, 0); 
  copy_block(B11, B, 0, 0);

  // results of the recursive process
  tile_matrix<T> M1(k, k, lbs);
  tile_matrix<T> M5(k, k, lbs);


  // start the tasks
  tbb::task_group g;
  // first task for M1, 7 more are needed for M2 to M8
  g.run([&] { M1 = classic(A11, B11, lbs); });
  …
  // wait all taks
  g.wait(); //
  // recombine all matrix to get the results
  M1 += M5;
```

The approach is similar for the [Strassen Implementation](https://github.com/timocafe/strassen/blob/main/algo/strassen.h).

Last and not least,  the binaries are link to the TBB pool manager extremely useful for the reuse the memory blocks A simple link to 
`TBB::tbbmalloc_proxy` is enough, all `new` and `delete` will be overload by the Intel library. We need now, a tiny scheduler for the
CPU/GPU.

## Scheduling CPU/GPU

The scheduling is basic. When a TBB::task will execute a single SGEMM, it will check if the GPU is free. If the GPU is free, it is locked and the task SGEMM is executed, when the SGEMM is over, the GPU is released. This pattern can be easily implemented using simple atomic and the `compare_exchange_strong` C++11 functionality:

```C++
  // gpu_ready_ is an std::atomic<int>
  int b(0); 
  if (gpu_ready_.compare_exchange_strong(b, 1)) { // check if the GPU is free, if yes, take it
    mul_matrix_gpu(mC, mA, mB); // perform SGEMM on GPU (CUBLAS)
    gpu_ready_ = 0; // Release GPU
  } else
    mul_matrix_cpu(mC, mA, mB); // perform SGEMM on CPU (EIGEN)

``` 

## Setup

The jetson works under the mode `nvpmodel -m 0`, and the number of TBB worker is [controled](https://github.com/timocafe/strassen/blob/main/sandbox/main.cpp#L54) per [task](https://www.threadingbuildingblocks.org/docs/help/reference/task_scheduler/tbb_global_control.html)

## Benchmarks

The benchmark consists to perform multiple SGEMM EIGEN full threads (blue), SGEMM cublass (green), 
Classical mix (CPU/GPU) grey and Strassen mix (CPU/GPU) orange. Results are reported in FLOPS (HPC’s metric) 
and time to solution because the Strassen algorithm makes less « macho » flops. For clarity, only significative results are presented.
The recursive process is performed only once, thus the tile size is equal to the original dimension divided by 2.
The mix classical or mix Strassen show the best results for matrices size lower then 4096, beyond a pure GPU version is faster. 
The noise for low value comes from the scheduler that does not reproduce the same pattern for every run. The number of CPU or GPU 
SGEMM varies between each run. I was not able to beat the CUBLAS, however during the development I remark:

<img src=https://github.com/timocafe/strassen/blob/main/images/flops.png align="center" height="318" width="410" />
<img src="https://github.com/timocafe/strassen/blob/main/images/time.png" align="center" height="318" width="410" />

* CudaMallocManaged is not stable in the asynchronous recursive multithread environment. It look likes, it is impossible to read a block of memory
on the CPU and GPU simultaneously without a segfault. Therefore for the GPU, the old method is applied GPU memory buffers are [allocated](https://github.com/timocafe/strassen/blob/main/memory/matrix.h#L221),
and a [data transfert](https://github.com/timocafe/strassen/blob/main/memory/matrix.h#L227) (here copy) is needed. It is an additional workload significative for low matrix size in the recursive process. Therefor I have lost all the advantages of the unify memory.

* I have no clue of the execution of the threads on the CPU and GPU simultaneously inside the microchip. I fear, there is a significative workload. 
Some profiling on a x86 machine does not show a big impact. However on such tiny machine, it could be again significative. 

# Conclusions

On this basic example of HPC on embedded device, I demonstrated that heterogenous is feasible and descent.
Embedded devices look like not relevant for HPC, but I remember you
during last decade, BlueGen supercomputers were leader in HPC. The last version [BG/Q](https://en.wikipedia.org/wiki/IBM_Blue_Gene) has A2 processor
`in order` for a peak of 220 [GFlops] for 15 [Watt]. The jetson can provide 1 [TFlops] for the same energy…
I consider these microchips like a serious option for next HPC cluster.


# Questions

* Cool ! Does it work on Apple Silicon M1 ?

There is no contradiction to apply this programming model on M1 processor. However for developer like me (C/C++), 
Apple does not provide any C++-API about their scheduler (grand central), 
Metal, and Neural Engine. It is a bit sad, because these technologies are developed in C/C++ but not open to the public.

* Cool ! So what for deep learning ?

Indeed the key operation of deep learning is SGEMM, however the wait of legacy and BLAS/LAPACK has privileged contiguous 
buffer for matrix representation. Inside Tensor Flow the heterogenous computing is performed on node  which represent 
operation coming from layer, this node can be executed on devices. I can imagine 
TF working on tile matrices where every nodes proposed heterogenous solver.

* Cool ! But is it really useful in real life ?

Well Every vendor (will) proposes heterogenous microchip. The last example of Apple is the most significant. 
After the question of the programming model and granularity is open.

* Memory, memory but algorithms are not more important ??

I consider the management of the memory the most important, specially in this kind of algorithm (Strassen). 
If the memory is not correctly managed get good performance is impossible. In the CUDA-sdk it is well demonstrated that
the memory management is the most important (stride, etc ..) 

* Nice, but could you do better ?

Indeed I could but it is very time consuming, and it will be more an academic work, not the purpose of this note.
