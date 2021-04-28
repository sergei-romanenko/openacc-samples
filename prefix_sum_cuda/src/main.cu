#include <iostream>
#include <algorithm>
#include <ctime>

#include "prefix_sum_cuda.h"

using std::cout;
using std::cerr;
using std::endl;
using std::fixed;

inline double get_wtime() {
	return ((double) clock()) / CLOCKS_PER_SEC;
}

void verify_result(char const *msg, const int num_elem, int *a, int *r) {
	for (int i = 0; i < num_elem; ++i) {
		if (a[i] != r[i]) {
			cerr << msg << ": result verification failed at element " << i
					<< "!" << endl;
			exit(EXIT_FAILURE);
		}
	}
//	cout << msg << ": OK" << endl;
}

void run_scan(char const *fn_name, void fn(int const, int*), int const num_elem,
		int *r) {
	double t1 = get_wtime();
	fn(num_elem, r);
	double t2 = get_wtime();
	cout << fn_name << ": " << fixed << (t2 - t1) << endl;
}

void run_test_cpu(char const *fn_name, void fn(int const, int*),
		int const num_elem, int *a, int *s, int *r) {
	std::copy(a, a + num_elem, r);
	run_scan(fn_name, fn, num_elem, r);
	verify_result(fn_name, num_elem, s, r);
}

void run_test_gpu(char const *fn_name, void fn(int const, int*),
		int const num_elem, int *a, int *s, int *r) {
	std::copy(a, a + num_elem, r);

	int *d_r;
	size_t size = num_elem * sizeof(int);
	CHECK(cudaMalloc(&d_r, size));
	CHECK(cudaMemcpy(d_r, r, size, cudaMemcpyHostToDevice));
	run_scan(fn_name, fn, num_elem, d_r);
	CHECK(cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost));
	CHECK(cudaFree(d_r));
	verify_result(fn_name, num_elem, s, r);
}

int main(int argc, char **argv) {
//	int const num_elem = 8;
//	int const num_elem = 16;
	int const num_elem = 1024;
//	int const num_elem = 32768;
//	int const num_elem = 1024 * 1024 * 64;
	cout << "Computing prefix sum for " << num_elem << " elements" << endl;

	int *a = new int[num_elem];
	int *r = new int[num_elem];
	int *s = new int[num_elem];

	// Verify that allocations succeeded.
	if (a == nullptr || r == nullptr || s == nullptr) {
		cerr << "Failed to allocate host vectors!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vector.
	for (int i = 0; i < num_elem; ++i) {
		a[i] = rand() % num_elem;
	}

	std::copy(a, a + num_elem, s);
	run_scan("inclusive_prefix_sum", inclusive_prefix_sum, num_elem, s);
//
//	run_test_cpu("ks_cpu", ks_cpu, num_elem, a, s, r);
//
//	run_test_cpu("dc_rec", dc_rec, num_elem, a, s, r);
//	run_test_cpu("dc_iter_cpu", dc_iter_cpu, num_elem, a, s, r);
//	run_test_acc("dc_iter_acc", dc_iter_acc, num_elem, a, s, r);
//
	std::copy(a, a + num_elem, s);
	run_scan("exclusive_prefix_sum", exclusive_prefix_sum, num_elem, s);
//
//	run_test_cpu("blelloch_rec", blelloch_rec, num_elem, a, s, r);
	run_test_cpu("blelloch_iter_cpu", blelloch_iter_cpu, num_elem, a, s, r);
	run_test_gpu("blelloch_cuda", blelloch_cuda, num_elem, a, s, r);

	cout << "Test PASSED" << endl;

	// Free host memory.
	delete[] s;
	delete[] a;
	delete[] r;

	cout << "Done" << endl;
	return 0;
}

