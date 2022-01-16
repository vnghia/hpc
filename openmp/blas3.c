#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_HPC /* Number of row of matrix A */
#define M 4
#else
#define M M_HPC
#endif

#ifndef K_HPC /* Number of columns of matrix A and rows of matrix B */
#define K 8
#else
#define K K_HPC
#endif

#ifndef N_HPC /* Number of columns of matrix B */
#define N 4
#else
#define N N_HPC
#endif

#ifndef BLOCK_HPC
#define BLOCK 4
#else
#define BLOCK BLOCK_HPC
#endif

/* Init a Matrix A(nrow,ncol) according to Aij = c*(i+j)/nrow/ncol */

void init(int nrow, int ncol, int ld, double *A, double cst) {
  int i, j;
#ifndef NO_OMP
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private(i, j) \
    num_threads(NUM_THREADS_HPC)
#endif
  for (i = 0; i < nrow; i++)
    for (j = 0; j < ncol; j++)
      A[i + j * ld] =
          cst * (double)(i + 1 + j + 1) / (double)nrow / (double)ncol;
}

/* Compute the Frobenius norm of a Matrix A(nrow,ncol) */

double norm(int nrow, int ncol, int ld, double *A) {
  double norm = 0.;
  int i, j;
#ifndef NO_OMP
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private(i, j) reduction(+ : norm) num_threads(NUM_THREADS_HPC)
#endif
  for (i = 0; i < nrow; i++)
    for (j = 0; j < ncol; j++) norm += A[i + j * ld] * A[i + j * ld];
  return sqrt(norm);
}

/* Print in terminal window a Matrix A(nrow,ncol) */

void print_array(int nrow, int ncol, int ld, double *A) {
  for (int i = 0; i < nrow; i++) {
    printf("(");
    for (int j = 0; j < ncol; j++) printf("%5.2f ", A[i + j * ld]);
    printf(")\n");
  }
  printf("\n");
}

/* Perform C = A x B with C a (N,M) matrix, A a (M,K) matrix and B a (K,N)
 * matrix */

void naive_dot(double *A, int lda, double *B, int ldb, double *C, int ldc) {
  int i, j, k;
/* Set the C matrix to zero */
#ifndef NO_OMP
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private(i, j) \
    num_threads(NUM_THREADS_HPC)
#endif
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) C[i + ldc * j] = 0.;
/* Perform the matrix-matrix product */
#ifndef NO_OMP
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private( \
    i, j, k) num_threads(NUM_THREADS_HPC)
#endif
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < K; k++)
#pragma omp atomic
        C[i + ldc * j] += A[i + lda * k] * B[k + ldb * j];
}

/* Perform C = A x B with C a (N,M) matrix, A a (M,K) matrix and B a (K,N)
 * matrix */

void saxpy_dot(double *A, int lda, double *B, int ldb, double *C, int ldc) {
  int i, j, k;
  double temp;
/* Set the C matrix to zero */
#ifndef NO_OMP
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private(i, j) \
    num_threads(NUM_THREADS_HPC)
#endif
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) C[i + ldc * j] = 0.;
/* Perform the matrix-matrix product */
#ifndef NO_OMP
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private( \
    i, j, k) num_threads(NUM_THREADS_HPC)
#endif
  for (k = 0; k < K; k++)
    for (j = 0; j < N; j++)
      for (i = 0; i < M; i++)
#pragma omp atomic
        C[i + ldc * j] += A[i + lda * k] * B[k + ldb * j];
}

/* Perform C = A x B with C a (N,M) matrix, A a (M,K) matrix and B a (K,N)
 * matrix */

void blocking_dot(double *A, int lda, double *B, int ldb, double *C, int ldc) {
  int i, j, k, ii, jj, kk;
  double temp;
/* Set the C matrix to zero */
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private(i, j) \
    num_threads(NUM_THREADS_HPC)
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) C[i + ldc * j] = 0.;

/* Perform the matrix-matrix product */
#ifndef NOT_DIVISIBLE_BY_BLOCK
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private( \
    i, j, k, ii, jj, kk) num_threads(NUM_THREADS_HPC)
  for (k = 0; k < K; k += BLOCK)
    for (j = 0; j < N; j += BLOCK)
      for (i = 0; i < M; i += BLOCK)
        for (kk = 0; kk < BLOCK; kk++)
          for (jj = 0; jj < BLOCK; jj++)
            for (ii = 0; ii < BLOCK; ii++)
#pragma omp atomic
              C[(ii + i) + ldc * (jj + j)] +=
                  A[(ii + i) + lda * (kk + k)] * B[(kk + k) + ldb * (jj + j)];
#else
#pragma omp parallel for schedule(SCHEDULE_HPC) default(shared) private( \
    i, j, k, ii, jj, kk) num_threads(NUM_THREADS_HPC)
  for (k = 0; k < K; k += BLOCK) {
    for (j = 0; j < N; j += BLOCK) {
      for (i = 0; i < M; i += BLOCK) {
        int kmin = fmin(K - k, BLOCK);
        for (kk = 0; kk < kmin; kk++) {
          int jmin = fmin(N - j, BLOCK);
          for (jj = 0; jj < jmin; jj++) {
            int imin = fmin(M - i, BLOCK);
            for (ii = 0; ii < imin; ii++) {
#pragma omp atomic
              C[(ii + i) + ldc * (jj + j)] +=
                  A[(ii + i) + lda * (kk + k)] * B[(kk + k) + ldb * (jj + j)];
            }
          }
        }
      }
    }
  }
#endif
}

int main() {
  int lda = M;
  int ldb = K;
  int ldc = M;

  double *a = (double *)malloc(lda * K * sizeof(double));
  double *b = (double *)malloc(ldb * N * sizeof(double));
  double *c = (double *)malloc(ldc * N * sizeof(double));

  double time;
  double flops = 2. * (double)N * (double)K * (double)M;

  /* Initialization of A and B matrices */

  init(M, K, lda, a, 1.0);
  init(K, N, ldb, b, 2.0);

#ifndef NO_PRINT_ARRAY
  print_array(M, K, lda, a);
  print_array(K, N, ldb, b);
#endif

  /* Naive dot */
#ifndef NO_NAIVE_DOT
  time = omp_get_wtime();
  naive_dot(a, lda, b, ldb, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time naive = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
#ifndef NO_PRINT_ARRAY
  print_array(M, N, ldc, c);
#endif
#endif

  /* Saxpy dot */
#ifndef NO_SAXPY_DOT
  time = omp_get_wtime();
  saxpy_dot(a, lda, b, ldb, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time saxpy = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
#ifndef NO_PRINT_ARRAY
  print_array(M, N, ldc, c);
#endif
#endif

  /* Blocking dot */
#ifndef NO_TILED_DOT
  time = omp_get_wtime();
  blocking_dot(a, lda, b, ldb, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time tiled = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
#ifndef NO_PRINT_ARRAY
  print_array(M, N, ldc, c);
#endif
#endif

  /* BLAS dot */
#ifndef NO_BLAS_DOT
  double alpha = 1.0;
  double beta = 0.0;
  time = omp_get_wtime();
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, lda,
              b, ldb, beta, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time BLAS  = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
#ifndef NO_PRINT_ARRAY
  print_array(M, N, ldc, c);
#endif
#endif

  free(a);
  free(b);
  free(c);
  return 0;
}
