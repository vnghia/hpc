#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define M 4 /* Number of row     of matrix A */
#define K 8 /* Number of columns of matrix A and rows of matrix B */
#define N 4 /* Number of columns of matrix B */

#define BLOCK 4

/* Init a Matrix A(nrow,ncol) according to Aij = c*(i+j)/nrow/ncol */

void init(int nrow, int ncol, int ld, double *A, double cst) {
  int i, j;
#pragma  // TO BE FINISHED
  for (i = 0; i < nrow; i++)
    for (j = 0; j < ncol; j++)
      A[i + j * ld] =
          cst * (double)(i + 1 + j + 1) / (double)nrow / (double)ncol;
}

/* Compute the Frobenius norm of a Matrix A(nrow,ncol) */

double norm(int nrow, int ncol, int ld, double *A) {
  double norm = 0.;
  int i, j;
#pragma  // TO BE FINISHED
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
#pragma  // TO BE FINISHED
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) C[i + ldc * j] = 0.;
/* Perform the matrix-matrix product */
#pragma  // TO BE FINISHED
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < K; k++) C[i + ldc * j] += A[i + lda * k] * B[k + ldb * j];
}

/* Perform C = A x B with C a (N,M) matrix, A a (M,K) matrix and B a (K,N)
 * matrix */

void saxpy_dot(double *A, int lda, double *B, int ldb, double *C, int ldc) {
  int i, j, k;
  double temp;
/* Set the C matrix to zero */
#pragma  // TO BE FINISHED
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) C[i + ldc * j] = 0.;
/* Perform the matrix-matrix product */
#pragma  // TO BE FINISHED
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < K; k++) C[i + ldc * j] += A[i + lda * k] * B[k + ldb * j];
}

/* Perform C = A x B with C a (N,M) matrix, A a (M,K) matrix and B a (K,N)
 * matrix */

void blocking_dot(double *A, int lda, double *B, int ldb, double *C, int ldc) {
  int i, j, k, ii, jj, kk;
  double temp;
/* Set the C matrix to zero */
#pragma  // TO BE FINISHED
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++) C[i + ldc * j] = 0.;
/* Perform the matrix-matrix product */
#pragma  // TO BE FINISHED
  for (i = 0; i < M; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < K; k++) C[i + ldc * j] += A[i + lda * k] * B[k + ldb * j];
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

  /* OpenMP informations */

  omp_sched_t kind;
  int chunk_size;

  printf("\nParallel execution with a maximum of %d threads callable\n\n",
         omp_get_max_threads());

  omp_get_schedule(&kind, &chunk_size);

  if (kind == 1) {
    printf("Scheduling static with chunk = %d\n\n", chunk_size);
  } else if (kind == 2) {
    printf("Scheduling dynamic with chunk = %d\n\n", chunk_size);
  } else if (kind == 3) {
    printf("Scheduling auto with chunk = %d\n\n", chunk_size);
  } else if (kind == 4) {
    printf("Scheduling guided with chunk = %d\n\n", chunk_size);
  }

  /* Initialization of A and B matrices */

  init(M, K, lda, a, 1.0);
  init(K, N, ldb, b, 2.0);
  print_array(M, K, lda, a);
  print_array(K, N, ldb, b);

  /* Naive dot */

  time = omp_get_wtime();
  naive_dot(a, lda, b, ldb, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time naive = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
  print_array(M, N, ldc, c);

  /* Saxpy dot */

  time = omp_get_wtime();
  saxpy_dot(a, lda, b, ldb, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time saxpy = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
  print_array(M, N, ldc, c);

  /* Blocking dot */

  time = omp_get_wtime();
  blocking_dot(a, lda, b, ldb, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time tiled = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
  print_array(M, N, ldc, c);

  /* BLAS dot */

  double alpha = 1.0;
  double beta = 0.0;
  time = omp_get_wtime();
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, a, lda,
              b, ldb, beta, c, ldc);
  time = omp_get_wtime() - time;
  printf("Frobenius Norm   = %f\n", norm(M, N, ldc, c));
  printf("Total time BLAS  = %f\n", time);
  printf("Gflops           = %f\n\n", flops / (time * 1e9));
  print_array(M, N, ldc, c);

  free(a);
  free(b);
  free(c);
  return 0;
}
