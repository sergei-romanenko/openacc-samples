#include <iostream>

using std::cout;
using std::cerr;
using std::endl;

#include "vect_add_acc.hpp"

int main(void) {
	// Print the vector length to be used, and compute its size
	int numElements = 50000;
	size_t size = numElements * sizeof(int);
	cout << "[Vector addition of " << numElements << " elements]" << endl;

	int *h_a = new int[size];
	int *h_b = new int[size];
	int *h_r = new int[size];
	int *r = new int[size];

	// Verify that allocations succeeded
	if (h_a == nullptr || h_b == nullptr || h_r == nullptr || r == nullptr) {
		cerr << "Failed to allocate host vectors!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_a[i] = rand();
		h_b[i] = rand();
	}

	// Launch the Vector Add CUDA Kernel
	vector_add_acc(numElements, h_a, h_b, h_r);

	// Verify that the result vector is correct
	vector_add_cpu(numElements, h_a, h_b, r);

	for (int i = 0; i < numElements; ++i) {
		if (h_r[i] != r[i]) {
			cerr << "Result verification failed at element " << i << "!"
					<< endl;
			exit(EXIT_FAILURE);
		}
	}

	cout << "Test PASSED" << endl;

	// Free host memory
	delete[] h_a;
	delete[] h_b;
	delete[] h_r;
	delete[] r;

	cout << "Done" << endl;
	return 0;
}
