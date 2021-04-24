#include <stdlib.h>
#include <cassert>

#include "prefix_sum_acc.hpp"

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
	assert((n != 0) && ((n & (n - 1)) == 0));

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
	assert((n != 0) && ((n & (n - 1)) == 0));
	int m = n / 2;
	dc_rec(m, a);
	dc_rec(m, a + m);
	int d = a[m - 1];
	for (int i = m; i < n; i++)
		a[i] += d;
}

void dc_iter_cpu(int const n, int *a) {
	assert((n != 0) && ((n & (n - 1)) == 0));

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
	assert((n != 0) && ((n & (n - 1)) == 0));

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
	assert((n != 0) && ((n & (n - 1)) == 0));

	blelloch_rec_up(n, a);
	a[n - 1] = 0;
	blelloch_rec_down(n, a);
}

void blelloch_iter_cpu(int const n, int *a) {
	assert((n != 0) && ((n & (n - 1)) == 0));

	for (int size = 1; size < n; size *= 2) {
		int num_tasks = n / (2 * size);
		for (int task = 0; task < num_tasks; task++) {
			int k = (2 * size) * task + size - 1;
			a[k + size] += a[k];
		}
	}

	a[n - 1] = 0;

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
	assert((n != 0) && ((n & (n - 1)) == 0));

	blelloch_iter_acc_up(n, a);

#pragma acc parallel present(a[0:n])
	a[n - 1] = 0;

	blelloch_iter_acc_down(n, a);
}
