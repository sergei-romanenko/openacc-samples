#include <stdlib.h>
#include <cassert>

#include "prefix_sum_acc.hpp"

void loop_fusion(int const n, int *__restrict a, int *__restrict b) {
#pragma acc data present(a[0:n], b[0:n])
#pragma acc kernels
	{
#pragma acc loop gang, vector(128)
		for (int i = 0; i < n; i++) {
			a[i] = a[i] + 1;
		}
#pragma acc loop gang, vector(128)
		for (int i = 0; i < n; i++) {
			a[i] = a[i] * 2;
		}
	}
}

inline bool is_power_of_2(int const n) {
	return (n >= 1) && ((n & (n - 1)) == 0);
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

// Kogge-Stone
void ks_cpu(int const n, int *a) {
	assert(is_power_of_2(n));

	int *z = (int*) malloc(2 * n * sizeof(int));
	int p = 1, q = 0;

	for (int i = 0; i < n; i++)
		z[i] = a[i];

	for (unsigned int stride = 1; stride < n; stride *= 2) {
		p = 1 - p, q = 1 - q;
		int *x = z + p * n;
		int *y = z + q * n;

		for (int i = 0; i < n; i++) {
			if (i >= stride)
				y[i] = x[i - stride] + x[i];
			else
				y[i] = x[i];
		}
	}

	for (int i = 0; i < n; i++)
		a[i] = z[q * n + i];

	free(z);
}

void dc_rec(int const n, int *a) {
	if (n <= 1)
		return;
	assert(is_power_of_2(n));
	int m = n / 2;
	dc_rec(m, a);
	dc_rec(m, a + m);
	int d = a[m - 1];
	for (int i = m; i < n; i++)
		a[i] += d;
}

void dc_iter_cpu(int const n, int *a) {
	assert(is_power_of_2(n));

	// size - the size of a chunk
	for (int size = 1; size < n; size *= 2) {
		int num_tasks = n / (2 * size);
		for (int task = 0; task < num_tasks; task++)
			for (int j = 0; j < size; j++) {
				int k = (2 * size) * task + size;
				a[k + j] += a[k - 1];
			}
	}
}

void dc_iter_acc(int const n, int *a) {
	assert(is_power_of_2(n));

	// size - the size of a chunk
	for (int size = 1; size < n; size *= 2) {
		int num_tasks = n / (2 * size);
#pragma acc data present(a[0:n])
#pragma acc parallel loop collapse(2) independent
		for (int task = 0; task < num_tasks; task++)
			for (int j = 0; j < size; j++) {
				int k = (2 * size) * task + size;
				a[k + j] += a[k - 1];
			}
	}
}

void blelloch_rec_up(int const n, int *a) {
	if (n <= 1)
		return;
	int m = n / 2;
	blelloch_rec_up(m, a);
	blelloch_rec_up(m, a + m);
	a[n - 1] += a[m - 1];
}

void blelloch_rec_down(int const n, int *a) {
	if (n <= 1)
		return;
	int m = n / 2;
	int r = a[m - 1];
	a[m - 1] = a[n - 1];
	a[n - 1] += r;
	blelloch_rec_down(m, a);
	blelloch_rec_down(m, a + m);
}

void blelloch_rec(int const n, int *a) {
	assert(is_power_of_2(n));

	blelloch_rec_up(n, a);
	a[n - 1] = 0;
	blelloch_rec_down(n, a);
}

void blelloch_iter_cpu_up(int const n, int *const a) {
	for (int size = 1; size < n; size *= 2) {
		int num_tasks = n / (2 * size);
		for (int task = 0; task < num_tasks; task++) {
			int k = (2 * size) * task + size - 1;
			a[k + size] += a[k];
		}
	}
}

void blelloch_iter_cpu_down(int const n, int *const a) {
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
	assert(is_power_of_2(n));

	blelloch_iter_cpu_up(n, a);
	a[n - 1] = 0;
	blelloch_iter_cpu_down(n, a);
}

//constexpr int TILE_SZ = 16;

void blelloch_iter_acc_up(int const n, int *const a) {
	for (int size = 1; size < n; size *= 2) {
		int num_tasks = n / (2 * size);
#pragma acc data present(a[0:n])
#pragma acc parallel loop independent
		for (int task = 0; task < num_tasks; task++) {
			int k = (2 * size) * task + size - 1;
			a[k + size] += a[k];
		}
	}
}

void blelloch_iter_acc_down(int const n, int *const a) {
	for (int size = n / 2; size > 0; size /= 2) {
		int num_tasks = n / (2 * size);
//#pragma acc parallel loop independent present(a[0:n]) tile(TILE_SZ)
//#pragma acc cache(a[0:n])
#pragma acc parallel loop independent present(a[0:n])
		for (int task = 0; task < num_tasks; task++) {
			int k = (2 * size) * task + size - 1;
			int a_k = a[k];
			a[k] = a[k + size];
			a[k + size] += a_k;
		}
	}
}

void blelloch_iter_acc(int const n, int *const a) {
	assert(is_power_of_2(n));

	blelloch_iter_acc_up(n, a);

#pragma acc parallel present(a[0:n])
	a[n - 1] = 0;

	blelloch_iter_acc_down(n, a);
}

inline int div_up(int n, int block_sz) {
	return (n % block_sz == 0) ? (n / block_sz) : (n / block_sz + 1);
}

// --- dc_scan_fan_cpu

void dc_scan_seq_cpu(const int n, int *const a) {
	int acc = a[0];
	for (int i = 1; i < n; i++) {
		acc += a[i];
		a[i] = acc;
	}
}

void dc_scan_incr_cpu(const int chunk_sz, int num_chunks, int *h,
		int *const a) {
	for (int k = 1; k < num_chunks; k++)
		for (int i = 0; i < chunk_sz; i++)
			a[chunk_sz * k + i] += h[k - 1];
}

void dc_scan_fan_cpu(int const chunk_sz, int const n, int *const a) {
	assert(is_power_of_2(chunk_sz));
	assert(is_power_of_2(n));

	if (n <= chunk_sz) {
		dc_scan_seq_cpu(n, a);
	} else {
		int num_chunks = n / chunk_sz;
		assert(num_chunks >= 2);
		int *h = new int[num_chunks];
		for (int k = 0; k < num_chunks; k++) {
			dc_scan_seq_cpu(chunk_sz, a + chunk_sz * k);
			h[k] = a[chunk_sz * k + (chunk_sz - 1)];
		}
		dc_scan_fan_cpu(chunk_sz, num_chunks, h);
		dc_scan_incr_cpu(chunk_sz, num_chunks, h, a);
		delete[] h;
	}
}

void dc_scan_fan_cpu256(int const n, int *const a) {
	dc_scan_fan_cpu(256, n, a);
}

// dc_scan_fan_acc

#pragma acc routine seq
void dc_scan_seq_acc(const int n, int *const a) {
#pragma acc data present(a[0:n])
	{
		int acc = a[0];
		for (int i = 1; i < n; i++) {
			acc += a[i];
			a[i] = acc;
		}
	}
}

void dc_scan_incr_acc(const int chunk_sz, int num_chunks, int *h,
		int *const a) {
#pragma acc data present(h[0:num_chunks], a[0:chunk_sz*num_chunks])
#pragma acc kernels
#pragma acc loop collapse(2) independent
	for (int k = 1; k < num_chunks; k++)
		for (int i = 0; i < chunk_sz; i++)
			a[chunk_sz * k + i] += h[k - 1];
}

void dc_scan_fan_acc(int const chunk_sz, int const n, int *const a) {
	assert(is_power_of_2(chunk_sz));
	assert(is_power_of_2(n));

	if (n <= chunk_sz) {
#pragma acc data present(a[0:n])
#pragma acc kernels
		dc_scan_seq_acc(n, a);
	} else {
		int num_chunks = n / chunk_sz;
		assert(num_chunks >= 2);
		int *h = new int[num_chunks];
#pragma acc enter data create(h[0:num_chunks])
#pragma acc data present(a[0:n], h[0:num_chunks])
#pragma acc kernels
#pragma acc loop independent
		for (int k = 0; k < num_chunks; k++) {
			dc_scan_seq_acc(chunk_sz, a + chunk_sz * k);
			h[k] = a[chunk_sz * k + (chunk_sz - 1)];
		}
		dc_scan_fan_acc(chunk_sz, num_chunks, h);
		dc_scan_incr_acc(chunk_sz, num_chunks, h, a);
#pragma acc exit data delete(h)
		delete[] h;
	}
}

void dc_scan_fan_acc256(int const n, int *const a) {
	dc_scan_fan_acc(256, n, a);
}
