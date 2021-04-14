#include <iostream>
#include <algorithm>
#include <omp.h>

#include "prefix_sum_acc.hpp"

using std::cout;
using std::cerr;
using std::endl;
using std::fixed;

void verify_result(char const *msg, const int num_elem, int *a, int *r) {
	for (int i = 0; i < num_elem; ++i) {
		if (a[i] != r[i]) {
			cerr << msg << ": result verification failed at element " << i
					<< "!" << endl;
			exit (EXIT_FAILURE);
		}
	}
//	cout << msg << ": OK" << endl;
}

int main(int argc, char **argv) {
//	int const num_elem = 16;
//	int const num_elem = 32768;
	int const num_elem = 1024 * 1024 * 64;
	cout << "Computing inclusive prefix sum for " << num_elem << " elements"
			<< endl;

	int *a = new int[num_elem];
	int *r = new int[num_elem];
	int *s = new int[num_elem];

	// Verify that allocations succeeded.
	if (a == nullptr || r == nullptr || s == nullptr) {
		cerr << "Failed to allocate host vectors!" << endl;
		exit (EXIT_FAILURE);
	}

	// Initialize the host input vector.
	for (int i = 0; i < num_elem; ++i) {
		a[i] = rand() % num_elem;
	}

	double prefix_sum_cpu_t1 = omp_get_wtime();
	prefix_sum_cpu(num_elem, a, s);
	double prefix_sum_cpu_t2 = omp_get_wtime();
	cout << "prefix_sum_cpu duration: " << fixed
			<< (prefix_sum_cpu_t2 - prefix_sum_cpu_t1) << endl;

	double ks_cpu_t1 = omp_get_wtime();
	ks_cpu(num_elem, a, r);
	double ks_cpu_t2 = omp_get_wtime();
	cout << "ks_cpu duration: " << fixed << (ks_cpu_t2 - ks_cpu_t1) << endl;
	verify_result("ks_cpu", num_elem, s, r);

	std::copy(a, a + num_elem, r);
	double dc_rec_cpu_t1 = omp_get_wtime();
	dc_rec_cpu(num_elem, r);
	double dc_rec_cpu_t2 = omp_get_wtime();
	cout << "dc_rec_cpu duration: " << fixed << (dc_rec_cpu_t2 - dc_rec_cpu_t1)
			<< endl;
	verify_result("dc_rec_cpu", num_elem, s, r);

	std::copy(a, a + num_elem, r);
	double dc_iter_cpu_t1 = omp_get_wtime();
	dc_iter_cpu(num_elem, r);
	double dc_iter_cpu_t2 = omp_get_wtime();
	cout << "dc_iter_cpu duration: " << fixed
			<< (dc_iter_cpu_t2 - dc_iter_cpu_t1) << endl;
	verify_result("dc_iter_cpu", num_elem, s, r);

	std::copy(a, a + num_elem, r);
#pragma acc enter data copyin(r[0:num_elem])
	double dc_iter_acc_t1 = omp_get_wtime();
	dc_iter_acc(num_elem, r);
	double dc_iter_acc_t2 = omp_get_wtime();
#pragma acc exit data copyout(r[0:num_elem])
	cout << "dc_iter_acc duration: " << fixed
			<< (dc_iter_acc_t2 - dc_iter_acc_t1) << endl;
	verify_result("dc_iter_acc", num_elem, s, r);

	cout << "Test PASSED" << endl;

	// Free host memory.
	delete[] s;
	delete[] a;
	delete[] r;

	cout << "Done" << endl;
	return 0;
}
