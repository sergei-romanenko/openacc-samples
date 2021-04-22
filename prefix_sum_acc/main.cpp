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
//	int const num_elem = 8;
//	int const num_elem = 16;
//	int const num_elem = 32768;
	int const num_elem = 1024 * 1024 * 64;
	cout << "Computing prefix sum for " << num_elem << " elements" << endl;

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

	std::copy(a, a + num_elem, s);
	double inclusive_prefix_sum_t1 = omp_get_wtime();
	inclusive_prefix_sum(num_elem, s);
	double inclusive_prefix_sum_t2 = omp_get_wtime();
	cout << "inclusive_prefix_sum duration: " << fixed
			<< (inclusive_prefix_sum_t2 - inclusive_prefix_sum_t1) << endl;

	double ks_cpu_t1 = omp_get_wtime();
	ks_cpu(num_elem, a, r);
	double ks_cpu_t2 = omp_get_wtime();
	cout << "ks_cpu duration: " << fixed << (ks_cpu_t2 - ks_cpu_t1) << endl;
	verify_result("ks_cpu", num_elem, s, r);

	std::copy(a, a + num_elem, r);
	double dc_rec_t1 = omp_get_wtime();
	dc_rec(num_elem, r);
	double dc_rec_t2 = omp_get_wtime();
	cout << "dc_rec_cpu duration: " << fixed << (dc_rec_t2 - dc_rec_t1) << endl;
	verify_result("dc_rec", num_elem, s, r);

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

	std::copy(a, a + num_elem, s);
	double exclusive_prefix_sum_t1 = omp_get_wtime();
	exclusive_prefix_sum(num_elem, s);
	double exclusive_prefix_sum_t2 = omp_get_wtime();
	cout << "exclusive_prefix_sum duration: " << fixed
			<< (exclusive_prefix_sum_t2 - exclusive_prefix_sum_t1) << endl;

	std::copy(a, a + num_elem, r);
	double blelloch_rec_t1 = omp_get_wtime();
	blelloch_rec(num_elem, r);
	double blelloch_rec_t2 = omp_get_wtime();
	cout << "blelloch_rec duration: " << fixed
			<< (blelloch_rec_t2 - blelloch_rec_t1) << endl;
	verify_result("blelloch_rec", num_elem, s, r);

	std::copy(a, a + num_elem, r);
	double blelloch_iter_cpu_t1 = omp_get_wtime();
	blelloch_iter_cpu(num_elem, r);
	double blelloch_iter_cpu_t2 = omp_get_wtime();
	cout << "blelloch_iter_cpu: " << fixed
			<< (blelloch_iter_cpu_t2 - blelloch_iter_cpu_t1) << endl;
	verify_result("blelloch_iter_cpu", num_elem, s, r);

	std::copy(a, a + num_elem, r);
#pragma acc enter data copyin(r[0:num_elem])
	double blelloch_iter_acc_t1 = omp_get_wtime();
	blelloch_iter_acc(num_elem, r);
	double blelloch_iter_acc_t2 = omp_get_wtime();
#pragma acc exit data copyout(r[0:num_elem])
	cout << "blelloch_iter_acc: " << fixed
			<< (blelloch_iter_acc_t2 - blelloch_iter_acc_t1) << endl;
	verify_result("blelloch_iter_acc", num_elem, s, r);

	cout << "Test PASSED" << endl;

	// Free host memory.
	delete[] s;
	delete[] a;
	delete[] r;

	cout << "Done" << endl;
	return 0;
}
