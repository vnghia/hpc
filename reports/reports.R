## ---- init ----

library(reticulate)
library(kableExtra)
library(ggplot2)
library(hrbrthemes)
library(scales)
library(viridis)

theme_set(theme_ipsum(base_family = "") +
  theme(axis.title.x = element_text(
    hjust = 0.5,
    size = 11
  ), axis.title.y = element_text(
    hjust = 0.5,
    size = 11
  ), ))

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

## ---- all-default-output ----
all_default_df <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy, Algo$tiled, Algo$blas
  ), Ms = default_m,
  Ks = default_k, Ns = default_n, blocks = default_block,
  omps = default_omps, schedules = default_schedule,
  chunks = default_chunk, num_threadss = default_num_threads
)$to_df(c(
  "algo",
  "time", "norm", "gflops", "omp"
))
all_default_df %>%
  kbl(
    booktabs = T, format.args = list(scientific = FALSE),
    linesep = c("", "\\addlinespace"),
    caption = "all techniques with default options"
  ) %>%
  kable_styling(latex_options = c("hold_position"))

## ---- sequential-shape ----
ms_sequential <- 2:11
ks_sequential <- ms_sequential
ns_sequential <- ms_sequential
sequential_num_threads <- list(1L)

## ---- sequential-output ----
sequential_db <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy, Algo$tiled, Algo$blas
  ), Ms = as.integer(2^ms_sequential),
  Ks = as.integer(2^ms_sequential), Ns = as.integer(2^ms_sequential),
  blocks = default_block, omps = default_omps,
  schedules = default_schedule, chunks = default_chunk,
  num_threadss = sequential_num_threads
)

sequential_df <- sequential_db$to_df(c(
  "algo",
  "M", "time", "omp"
), F)
sequential_plot <- sequential_df %>%
  ggplot(aes(
    x = M, y = time, shape = omp,
    color = algo, group = interaction(
      omp,
      algo
    )
  )) +
  geom_line(alpha = 0.75) +
  geom_point(size = 2, alpha = 0.9) +
  scale_x_continuous(
    trans = log2_trans(),
    breaks = trans_breaks("log2", function(x) 2^x),
    labels = trans_format("log2", math_format(2^.x))
  ) +
  xlab("M = K = N") +
  ylab("Time (s)") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = turbo(4))

## ---- sequential-last-shape-output
sequential_db$to_df(c(
  "algo", "M", "time",
  "omp"
))[sequential_df$M == 2^tail(
  ms_sequential,
  1
), -2] %>%
  kbl(
    booktabs = T, format.args = list(scientific = FALSE),
    linesep = c("", "\\addlinespace"),
    caption = paste(
      "Computation time when M = N = K = ",
      2^tail(ms_sequential, 1)
    ), format = "latex",
    row.names = F
  ) %>%
  kable_styling(latex_options = c("hold_position"))