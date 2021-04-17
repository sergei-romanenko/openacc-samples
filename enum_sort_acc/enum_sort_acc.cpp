#include "enum_sort_acc.hpp"

void bubble_sort(int n, int *a) {
	for (int m = n - 1; m > 0; m--) {
		for (int i = 0; i < m; ++i) {
			if (a[i] > a[i + 1]) {
				int temp = a[i];
				a[i] = a[i + 1];
				a[i + 1] = temp;
			}
		}
	}
}

void enum_sort_cpu(int const n, int const *a, int *r) {

	for (int i = 0; i < n; i++) {
		int v = a[i];
		int rank = 0;
		for (int j = 0; j < n; j++) {
			if (a[j] < v || (a[j] == v && j < i))
				rank++;
		}
		r[rank] = v;
	}
}

void enum_sort_acc(int const n, int const *a, int *r) {

#pragma acc parallel loop present(a[:n], r[:n])
	for (int i = 0; i < n; i++) {
		int v = a[i];
		int rank = 0;
		for (int j = 0; j < n; j++) {
			if (a[j] < v || (a[j] == v && j < i))
				rank++;
		}
		r[rank] = v;
	}
}
