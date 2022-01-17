## ---- init ----

library(reticulate)

source("utils.R")
source_python("utils.py")
source_python("openmp.py")

blas3_source_code <- "../openmp/blas3.c"

## ---- naive-dot-shape ----
m_naive_dot <- 1L
k_naive_dot <- 2L
n_naive_dot <- 2L

## ---- naive-dot-output ----
naive_122_output <- BinaryOpenMP$compile_and_run(
  M = m_naive_dot,
  K = k_naive_dot, N = n_naive_dot, print_array = T,
  save_output = "bin/openmp/output/naive_122_output.txt"
)[[1]]$raw_outputs
writeLines(paste(naive_122_output, collapse = "\n"))

## ---- naive-dot-true ----
naive_122_a <- init_matrix(
  m_naive_dot, k_naive_dot,
  1
)
naive_122_b <- init_matrix(
  k_naive_dot, n_naive_dot,
  2
)
naive_122_c <- naive_122_a %*% naive_122_b
writeLines(paste(
  pmatrix(naive_122_a), pmatrix(naive_122_b),
  "=", pmatrix(naive_122_c)
))