#ifndef _PREFIX_SUM_ACC_
#define _PREFIX_SUM_ACC_

void prefix_sum_cpu(int const n, int *a);
//void prefix_sum_acc(int const n, int const *a, int *r);
void ks_cpu(int const n, int const *a, int *r);
void dc_rec_cpu(int const n, int *a);
void dc_iter_cpu(int const n, int *a);
void dc_iter_acc(int const n, int *a);

#endif
