#include <stdlib.h>
#include <cassert>

#include "prefix_sum_acc.hpp"

void prefix_sum_cpu(int const n, int *a) {
	int acc = a[0];
	for (int i = 1; i < n; i++) {
		acc += a[i];
		a[i] = acc;
	}
}

// Kogge-Stone
void ks_cpu(int const n, int const *a, int *r) {
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
		r[i] = z[q * n + i];

	free(z);
}

void dc_rec_cpu(int const n, int *a) {
	if (n <= 1)
		return;
	assert((n != 0) && ((n & (n - 1)) == 0));
	int m = n / 2;
	dc_rec_cpu(m, a);
	dc_rec_cpu(m, a + m);
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
