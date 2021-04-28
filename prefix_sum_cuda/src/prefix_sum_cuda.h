#ifndef _PREFIX_SUM_CUDA_H_
#define _PREFIX_SUM_CUDA_H_


// Handle CUDA errors

void handle_error(cudaError_t err, const char *file, int line);
#define CHECK(err) (handle_error(err, __FILE__, __LINE__))

void inclusive_prefix_sum(int const n, int *a);
void exclusive_prefix_sum(int const n, int *a);

void blelloch_iter_cpu(int const n, int *a);
void blelloch_cuda(int const n, int *a);

#endif
