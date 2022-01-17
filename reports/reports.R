## ---- init ----

library(reticulate)
library(kableExtra)

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

## ---- naive-saxpy-small-shape ----
m_naive_saxpy_small <- list(4L)
k_naive_saxpy_small <- list(8L)
n_naive_saxpy_small <- list(4L)

## ---- naive-saxpy-small-output ----
naive_saxpy_small_df <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy
  ), Ms = m_naive_saxpy_small,
  Ks = k_naive_saxpy_small, Ns = n_naive_saxpy_small
)$to_df(c(
  "algo",
  "time", "norm", "gflops"
))
naive_saxpy_small_df %>%
  kbl(
    booktabs = T, format.args = list(scientific = FALSE),
    caption = paste("naive vs saxpy when M, N, K is small")
  ) %>%
  kable_styling(latex_options = c("hold_position"))

## ---- naive-saxpy-big-shape ----
default_m <- list(2048L)
default_k <- list(2048L)
default_n <- list(2048L)

## ---- naive-saxpy-big-output ----
naive_saxpy_big_df <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy
  ), Ms = default_m, Ks = default_k,
  Ns = default_n
)$to_df(c(
  "algo", "time",
  "norm", "gflops"
))
naive_saxpy_big_df %>%
  kbl(
    booktabs = T, format.args = list(scientific = FALSE),
    caption = paste("naive vs saxpy when M, N, K is big")
  ) %>%
  kable_styling(latex_options = c("hold_position"))

## ---- naive-saxpy-omp-shape ----
default_omps <- c(T, F)
default_schedule <- list(Schedule$static)
default_chunk <- list(0L)
default_num_threads <- list(4L)

## ---- naive-saxpy-omp-output ----
naive_saxpy_omp_df <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy
  ), Ms = default_m, Ks = default_k,
  Ns = default_n, omps = default_omps,
  schedules = default_schedule, chunks = default_chunk,
  num_threadss = default_num_threads
)$to_df(c(
  "algo",
  "time", "norm", "gflops", "omp"
))
naive_saxpy_omp_df %>%
  kbl(
    booktabs = T, format.args = list(scientific = FALSE),
    caption = "naive vs saxpy with OpenMP"
  ) %>%
  kable_styling(latex_options = c("hold_position"))

## ---- naive-saxpy-tiled-shape ----
default_block <- list(4L)

## ---- naive-saxpy-tiled-output ----
naive_saxpy_tiled_df <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy, Algo$tiled
  ), Ms = default_m,
  Ks = default_k, Ns = default_n, blocks = default_block,
  omps = default_omps, schedules = default_schedule,
  chunks = default_chunk, num_threadss = default_num_threads
)$to_df(c(
  "algo",
  "time", "norm", "gflops", "omp"
))
naive_saxpy_tiled_df %>%
  kbl(
    booktabs = T, format.args = list(scientific = FALSE),
    linesep = c("", "\\addlinespace"),
    caption = "naive vs saxpy vs tiled"
  ) %>%
  kable_styling(latex_options = c("hold_position"))