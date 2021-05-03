#include <iostream>
#include <iomanip>
#include <cassert>

#include "prefix_sum_cuda.h"

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
#include <iostream>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Handle CUDA errors

void handle_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		std::cout << cudaGetErrorString(err) << " in " << file << " at line "
				<< line << std::endl;
		exit(EXIT_FAILURE);
	}
}

// Round a / b to nearest higher integer value
inline int div_up(int a, int b) {
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void inclusive_prefix_sum(int const n, int *a) {
	int acc = a[0];
	for (int i = 1; i < n; i++) {
		acc += a[i];
		a[i] = acc;
	}
}

void exclusive_prefix_sum(int const n, int *a) {
	int acc = 0;
	for (int i = 0; i < n; i++) {
		int a_i = a[i];
		a[i] = acc;
		acc += a_i;
	}
}

void blelloch_iter_cpu_up(int const n, int * const a) {
	for (int size = 1; size < n; size *= 2) {
		int num_tasks = n / (2 * size);
		for (int task = 0; task < num_tasks; task++) {
			int k = (2 * size) * task + size - 1;
			a[k + size] += a[k];
		}
	}
}

void blelloch_iter_cpu_down(int const n, int * const a) {
	for (int size = n / 2; size > 0; size /= 2) {
		int num_tasks = n / (2 * size);
		for (int task = 0; task < num_tasks; task++) {
			int k = (2 * size) * task + size - 1;
			int a_k = a[k];
			a[k] = a[k + size];
			a[k + size] += a_k;
		}
	}
}

void blelloch_iter_cpu(int const n, int *a) {
	assert((n != 0) && ((n & (n - 1)) == 0));

	blelloch_iter_cpu_up(n, a);
	a[n - 1] = 0;
	blelloch_iter_cpu_down(n, a);
}

__device__
void blelloch_cuda_up(int const n, int *a) {
//	cg::grid_group g = cg::this_grid();
	cg::thread_group g = cg::this_thread_block();
	unsigned task = g.thread_rank();

	for (int size = 1; size < n; size *= 2) {
		g.sync();
		int num_tasks = n / (2 * size);
		if (task < num_tasks) {
			int k = (2 * size) * task + size - 1;
			a[k + size] += a[k];
		}
	}
}

__device__
void blelloch_cuda_down(int const n, int *a) {
//	cg::grid_group g = cg::this_grid();
	cg::thread_group g = cg::this_thread_block();
	unsigned task = g.thread_rank();

	for (int size = n / 2; size > 0; size /= 2) {
		g.sync();
		int num_tasks = n / (2 * size);
		if (task < num_tasks) {
			int k = (2 * size) * task + size - 1;
			int a_k = a[k];
			a[k] = a[k + size];
			a[k + size] += a_k;
		}
	}
}

__global__
void blelloch_cuda_kernel(int const n, int *a) {
//	cg::grid_group g = cg::this_grid();
	cg::thread_group g = cg::this_thread_block();
	unsigned task = g.thread_rank();

	blelloch_cuda_up(n, a);
	if (task == 0)
		a[n - 1] = 0;
	blelloch_cuda_down(n, a);
}

void blelloch_cuda(int const n, int *a) {
	assert((n != 0) && ((n & (n - 1)) == 0));

//	int block_sz = 128;
	int block_sz = n / 2;
	int grid_sz = (n / 2) / block_sz;
	assert(grid_sz == 1);

	blelloch_cuda_kernel<<<grid_sz, block_sz>>>(n, a);
}

// --- Scan then Fan.

template<class T, bool is_zero_padded>
inline __device__ T scan_warp(volatile T *s_partials) {
	T t = s_partials[0];
	if (is_zero_padded) {
		t += s_partials[-1];
		s_partials[0] = t;
		t += s_partials[-2];
		s_partials[0] = t;
		t += s_partials[-4];
		s_partials[0] = t;
		t += s_partials[-8];
		s_partials[0] = t;
		t += s_partials[-16];
		s_partials[0] = t;
	} else {
		const int tid = threadIdx.x;
		const int lane = tid & 31;
		if (lane >= 1) {
			t += s_partials[-1];
			s_partials[0] = t;
		}
		if (lane >= 2) {
			t += s_partials[-2];
			s_partials[0] = t;
		}
		if (lane >= 4) {
			t += s_partials[-4];
			s_partials[0] = t;
		}
		if (lane >= 8) {
			t += s_partials[-8];
			s_partials[0] = t;
		}
		if (lane >= 16) {
			t += s_partials[-16];
			s_partials[0] = t;
		}
	}
	return t;
}

template<class T, bool is_zero_padded>
inline __device__ T scan_block(volatile T *s_partials) {
	extern __shared__ T w_partials[];
	const int tid = threadIdx.x;
	const int lane = tid & 31;
	const int warpid = tid >> 5;

	// Compute this thread's partial sum
	T sum = scan_warp<T, is_zero_padded>(s_partials);
	__syncthreads();

	// Write each warp's reduction to shared memory
	if (lane == 31) {
		w_partials[16 + warpid] = sum;
	}
	__syncthreads();

	// Have one warp scan reductions
	if (warpid == 0) {
		scan_warp<T, is_zero_padded>(16 + w_partials + tid);
	}
	__syncthreads();

	// Fan out the exclusive scan element (obtained
	// by the conditional and the decrement by 1)
	// to this warp's pending output
	if (warpid > 0) {
		sum += w_partials[16 + warpid - 1];
	}
	__syncthreads();

	// Write this thread's scan output
	*s_partials = sum;
	__syncthreads();

	// The return value will only be used by caller if it
	// contains the spine value (i.e., the reduction
	// of the array we just scanned).
	return sum;
}

template<class T, bool write_spine>
__global__ void scan_and_write_partials(T *g_partials, const T *a, size_t n,
		size_t num_blocks, T *r) {
	extern volatile __shared__ T s_partials[];
	const int t = threadIdx.x;
	volatile T *t_partials = s_partials + t;
	for (size_t b = blockIdx.x; b < num_blocks; b += gridDim.x) {
		size_t i = b * blockDim.x + t;
		*t_partials = (i < n) ? a[i] : 0;
		__syncthreads();
		T sum = scan_block<T, false>(t_partials);
		__syncthreads();
		if (i < n) {
			r[i] = *t_partials;
		}

		// write the spine value to global memory
		if (write_spine && (threadIdx.x == (blockDim.x - 1))) {
			g_partials[b] = sum;
		}
	}
}

template<class T>
__global__ void scan_add_base_sums(T *g_base_sums, size_t n, size_t num_blocks,
		T *out) {
	const int t = threadIdx.x;

	T fan_value = 0;
	for (size_t b = blockIdx.x; b < num_blocks; b += gridDim.x) {
		size_t index = b * blockDim.x + t;
		if (b > 0) {
			fan_value = g_base_sums[b - 1];
		}
		out[index] += fan_value;
	}
}

template<class T>
void scan_fan(int block_sz, size_t n, T const *a, T *r) {
	if (n <= block_sz) {
		scan_and_write_partials<T, false> <<<1, block_sz, block_sz * sizeof(T)>>>(
				0, a, n, 1, r);
		return;
	}

	// device pointer to array of partial sums in global memory
	T *g_partials = 0;

	// ceil(N/b)
	unsigned int num_partials = (n + block_sz - 1) / block_sz;

	// number of CUDA threadblocks to use. The kernels are
	// blocking agnostic, so we can clamp to any number
	// within CUDA's limits and the code will work.
	const unsigned int max_blocks = 150;
	// maximum blocks to launch
	unsigned int num_blocks = min(num_partials, max_blocks);
	CHECK(cudaMalloc(&g_partials, num_partials * sizeof(T)));
	scan_and_write_partials<T, true> <<<num_blocks, block_sz,
			block_sz * sizeof(T)>>>(g_partials, a, n, num_partials, r);
	scan_fan<T>(block_sz, num_partials, g_partials, g_partials);
	scan_add_base_sums<T> <<<num_blocks, block_sz>>>(g_partials, n, num_partials,
			r);
	CHECK(cudaFree(g_partials));
}

void scan_fan_cuda128(int const n, int const *a, int *r) {
	scan_fan<int>(128, n, a, r);
}

void scan_fan_cuda256(int const n, int const *a, int *r) {
	scan_fan<int>(256, n, a, r);
}

void scan_fan_cuda512(int const n, int const *a, int *r) {
	scan_fan<int>(512, n, a, r);
}
