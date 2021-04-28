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
