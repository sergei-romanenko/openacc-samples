#include "vect_add_acc.hpp"

#include <iostream>

void vector_add_cpu(int const n, int const *a, int const *b, int *r) {
	for (int i = 0; i < n; i++)
		r[i] = a[i] + b[i];
}

void vector_add_acc(int const n, int const *a, int const *b, int *r) {
#pragma acc parallel loop copyin(a[:n], b[:n]) copyout(r[:n])
	for (int i = 0; i < n; i++)
		r[i] = a[i] + b[i];
}
