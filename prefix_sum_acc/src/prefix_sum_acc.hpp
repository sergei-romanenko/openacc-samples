#ifndef _PREFIX_SUM_ACC_
#define _PREFIX_SUM_ACC_

void inclusive_prefix_sum(int const n, int *a);
void exclusive_prefix_sum(int const n, int *a);

void ks_cpu(int const n, int *a);

void dc_rec(int const n, int *a);
void dc_iter_cpu(int const n, int *a);
void dc_iter_acc(int const n, int *a);

void blelloch_rec(int const n, int *a);
void blelloch_iter_cpu(int const n, int *a);
void blelloch_iter_acc(int const n, int *a);

void dc_scan_fan_cpu256(int const n, int *const a);
void dc_scan_fan_acc256(int const n, int *const a);

#endif
