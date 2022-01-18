## ---- init ----

library(reticulate)
library(kableExtra)
library(ggplot2)
library(hrbrthemes)
library(scales)
library(tidyr)
library(viridis)

theme_set(theme_ipsum(base_family = "") +
  theme(axis.title.x = element_text(
    hjust = 0.5
  ), axis.title.y = element_text(
    hjust = 0.5
  ), plot.margin = margin(
    t = 0.5, r = 2, b = 0.5, l = 2, "cm"
  )))

knitr::opts_chunk$set(dev = "tikz")
options(tikzDefaultEngine = "luatex")

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
  scale_color_manual(values = turbo(4)) +
  scale_shape_discrete(labels = c("F", "T"))

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

## ---- threading-shape ----
threading_num_threadss <- 1:7
threading_schedule <- c(
  Schedule$static,
  Schedule$dynamic, Schedule$guided, Schedule$auto
)
threading_chunks <- 0L
threading_params <- expand.grid(
  threading_schedule,
  threading_chunks, as.integer(2^threading_num_threadss)
)
threading_omp <- list(T)

## ---- threading-output ----
threading_db <- DBOpenMP(
  algos = c(
    Algo$naive,
    Algo$saxpy, Algo$tiled
  ), Ms = default_m,
  Ks = default_k, Ns = default_n, blocks = default_block,
  omps = threading_omp, schedules = threading_params[
    ,
    1
  ], chunks = threading_params[, 2],
  num_threadss = threading_params[, 3]
)
threading_blas_df <- DBOpenMP(
  algos = c(Algo$blas),
  Ms = default_m, Ks = default_k, Ns = default_n
)$to_df(c(
  "algo",
  "time", "schedule"
), F) %>%
  crossing(num_threads = 2^threading_num_threadss) %>%
  as.data.frame()
threading_blas_df["schedule"] <- "static"

threading_df <- rbind(
  threading_db$to_df(c(
    "algo",
    "time", "schedule", "num_threads"
  ), F),
  threading_blas_df
)

threading_plot <- threading_df %>%
  ggplot(aes(
    x = num_threads, y = time,
    shape = schedule, color = algo, group = interaction(
      schedule,
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
  xlab("Number of threads") +
  ylab("Time (s)") +
  theme(legend.position = "bottom", legend.box = "vertical") +
  scale_color_manual(values = turbo(4))

## ---- blocking-shape ----
blocking_blocks <- 0:10

## ---- blocking-output ----
blocking_db <- DBOpenMP(
  algos = c(
    Algo$tiled
  ), Ms = default_m,
  Ks = default_k, Ns = default_n, blocks = as.integer(2^blocking_blocks),
  omps = default_omps, schedules = default_schedule, chunks = default_chunk,
  num_threadss = default_num_threads
)
blocking_psuedo_db <- DBOpenMP(
  algos = c(
    Algo$saxpy, Algo$blas
  ), Ms = default_m,
  Ks = default_k, Ns = default_n, blocks = default_block,
  omps = list(T), schedules = default_schedule, chunks = default_chunk,
  num_threadss = default_num_threads
)$to_df(c(
  "algo", "time", "omp"
), F) %>%
  crossing(block = 2^blocking_blocks) %>%
  as.data.frame()

blocking_df <- rbind(blocking_db$to_df(c(
  "algo", "time", "omp", "block"
), F), blocking_psuedo_db)

blocking_plot <- blocking_df %>%
  ggplot(aes(
    x = block, y = time, shape = omp,
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
  xlab("Block size") +
  ylab("Time (s)") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = turbo(4)) +
  scale_shape_discrete(labels = c("F", "T"))