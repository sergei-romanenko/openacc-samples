#include <iostream>
#include <algorithm>

#include "enum_sort_acc.hpp"

using std::cout;
using std::cerr;
using std::endl;

void verify_result(const int num_elem, int *a, int *r) {
	for (int i = 0; i < num_elem; ++i) {
		if (a[i] != r[i]) {
			cerr << "Result verification failed at element " << i << "!"
					<< endl;
			exit (EXIT_FAILURE);
		}
	}
}

int main(int argc, char **argv) {
	int const num_elem = 10000;
	cout << "Sorting " << num_elem << " elements" << endl;

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
	std::sort(s, s + num_elem);

	std::copy(a, a + num_elem, r);
	bubble_sort(num_elem, r);
	verify_result(num_elem, s, r);

	enum_sort_cpu(num_elem, a, r);
	verify_result(num_elem, s, r);

#pragma acc enter data copyin(a[:num_elem]) create(r[:num_elem])
	enum_sort_acc(num_elem, a, r);
#pragma acc exit data delete(a) copyout(r[:num_elem])
	verify_result(num_elem, s, r);

	cout << "Test PASSED" << endl;

	// Free host memory.
	delete[] s;
	delete[] a;
	delete[] r;

	cout << "Done" << endl;
	return 0;
}
