#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

#include "vect_add_acc.hpp"

int main(void) {
	// Print the vector length to be used.
	int num_elem = 50000;
	cout << "[Vector addition of " << num_elem << " elements]" << endl;

	int *h_a = new int[num_elem];
	int *h_b = new int[num_elem];
	int *h_r = new int[num_elem];
	int *r = new int[num_elem];

	// Verify that allocations succeeded.
	if (h_a == nullptr || h_b == nullptr || h_r == nullptr || r == nullptr) {
		cerr << "Failed to allocate host vectors!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors.
	for (int i = 0; i < num_elem; ++i) {
		h_a[i] = rand();
		h_b[i] = rand();
	}

	// Launch the Vector Add Kernel.
	vector_add_acc(num_elem, h_a, h_b, h_r);

	// Verify that the result vector is correct.
	vector_add_cpu(num_elem, h_a, h_b, r);

	for (int i = 0; i < num_elem; ++i) {
		if (h_r[i] != r[i]) {
			cerr << "Result verification failed at element " << i << "!"
					<< endl;
			exit(EXIT_FAILURE);
		}
	}

	cout << "Test PASSED" << endl;

	// Free host memory.
	delete[] h_a;
	delete[] h_b;
	delete[] h_r;
	delete[] r;

	cout << "Done" << endl;
	return 0;
}
