// Vector addition: r = a + b.

#include <stdio.h>
#include <iostream>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;

// Handle CUDA errors

void handle_error(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << file << " at line " << line
				<< endl;
		exit(EXIT_FAILURE);
	}
}

#define CHECK(err) (handle_error(err, __FILE__, __LINE__))

// This will output the proper error string when calling cudaGetLastError

void __handle_last_cuda_error(const char *errorMessage, const char *file,
		const int line) {
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {
		cout << cudaGetErrorString(err) << " in " << file << " at line " << line
				<< endl;
		exit(EXIT_FAILURE);
	}
}

#define handle_last_cuda_error(msg) __handle_last_cuda_error(msg, __FILE__, __LINE__)

void vector_add_cpu(int n, int* a, int* b, int* r) {
	for (int i = 0; i < n; i++)
		r[i] = a[i] + b[i];
}

__global__
void vector_add_kernel(int n, int const *a, int const *b, int *r) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < n) {
		r[i] = a[i] + b[i];
	}
}

void vector_add_gpu(int n, int const *a, int const *b, int *r) {
	// Launch the Vector Add CUDA Kernel
	int TPB = 256; // Threads per block.
	int BPG = (n + TPB - 1) / TPB; // Blocks per grid
	cout << "CUDA kernel launch with " << BPG << " blocks of " << TPB
			<< " threads" << endl;
	vector_add_kernel<<<BPG, TPB>>>(n, a, b, r);
	CHECK(cudaGetLastError());
}

/**
 * Host main routine
 */
int main(void) {
	// Print the vector length to be used, and compute its size
	int numElements = 50000;
	size_t size = numElements * sizeof(int);
	cout << "[Vector addition of " << numElements << " elements]" << endl;

	int *h_a = (int *) malloc(size);
	int *h_b = (int *) malloc(size);
	int *h_r = (int *) malloc(size);
	int *r = (int *) malloc(size);

	// Verify that allocations succeeded
	if (h_a == NULL || h_b == NULL || h_r == NULL || r == NULL) {
		cerr << "Failed to allocate host vectors!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_a[i] = rand();
		h_b[i] = rand();
	}

	cout << "Copy input data from the host memory to the CUDA device" << endl;
	int *d_a = nullptr;
	CHECK(cudaMalloc(&d_a, size));

	int *d_b = nullptr;
	CHECK(cudaMalloc(&d_b, size));

	int *d_r = nullptr;
	CHECK(cudaMalloc(&d_r, size));

	CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

	// Launch the Vector Add CUDA Kernel
	vector_add_gpu(numElements, d_a, d_b, d_r);

	cout << "Copy output data from the CUDA device to the host memory" << endl;
	CHECK(cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost));

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

	// Free device global memory
	CHECK(cudaFree(d_a));
	CHECK(cudaFree(d_b));
	CHECK(cudaFree(d_r));

	// Free host memory
	free(h_a);
	free(h_b);
	free(h_r);
	free(r);

	cout << "Done" << endl;
	return 0;
}

