#include "enum_sort_acc.hpp"

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
