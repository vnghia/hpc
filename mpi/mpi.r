## ---- init ----

library(dplyr)
library(reticulate)
library(kableExtra)
library(ggplot2)
library(ggraph)
library(hrbrthemes)
library(network)
library(stringr)
library(tidyr)
library(viridis)

theme_set(theme_ipsum(base_family = "") + theme(
  axis.title.x = element_text(hjust = 0.5),
  axis.title.y = element_text(hjust = 0.5), plot.margin = margin(
    t = 0.5,
    r = 2, b = 0.5, l = 2, "cm"
  )
))

source(here::here("reports", "utils.r"))
source_python(here::here("reports", "utils.py"))

py_run_string(sprintf(
  "
import os
os.environ['MPI_ROOT_PATH'] = '%s'
os.environ['PYTHON_BIN'] = '%s'
",
  here::here("mpi", ""), py_config()$python
))
source_python(here::here("mpi", "mpi.py"))

## ---- mpi-function-definition ----

read_binary_mpi_output <- function(path, command) {
  py_outputs <- list()
  binary_mpi <- BinaryMPI(
    Session = NULL, np = 0L,
    algo_source = command, multiple_times = 1L
  )
  if (file.exists(path)) {
    py_outputs <- readLines(path)
    binary_mpi$raw_outputs <- py_outputs
    binary_mpi$parse()
  } else {
    binary_mpi$run()
    py_outputs <- binary_mpi$raw_outputs
    py_outputs_str <- paste(py_outputs, collapse = "\n")
    file.create(path)
    writeLines(py_outputs_str, path)
  }
  return(list(outputs = py_outputs, binary = binary_mpi))
}

## ---- init-read-notebook ----
nb_sources <- read_from_notebook(here::here(
  "mpi",
  "ranking.ipynb"
), commit = "b6ec843")

## ---- init-introduction-matrix ----
py_run_string(paste(nb_sources[[1]][-41], collapse = "\n"))
writeLines(pmatrix(as.matrix(py$H), T))

## ---- power-iteration-introduction-matrix
## define norm1/norm2/dot_product
py_run_string(paste(nb_sources[[3]], collapse = "\n"))
# define power_iteration
py_run_string(paste(nb_sources[[4]][2:19], collapse = "\n"))
# define matvec
py_run_string(paste(nb_sources[[4]][31:32], collapse = "\n"))

init_n_u_v_from_h <- paste(nb_sources[[4]][39:41],
  collapse = "\n"
)
py_run_string(init_n_u_v_from_h)

# Run Power Iteration on matrix H
power_introduction_output_path <- here::here(
  "mpi",
  "bin", "output", "power_introduction.txt"
)
power_introduction_result <- read_binary_mpi_output(
  power_introduction_output_path,
  paste(nb_sources[[4]][43:44], collapse = "\n")
)

## ---- power-introduction-matrix-output
writeLines(paste(power_introduction_result[[1]], collapse = "\n"))

## ---- google-matrix ----

# define dangling_nodes/google_matrix_dense
py_run_string(paste(nb_sources[[5]][c(2:11)], collapse = "\n"))
# define matvec_sparse
py_run_string(paste(nb_sources[[7]][c(2:9)], collapse = "\n"))
# define init_H
py_run_string(paste(str_replace_all(
  nb_sources[[6]],
  fixed("load_npz(\""), sprintf(
    "load_npz(\"%s",
    here::here("mpi", "")
  )
), collapse = "\n"))
# define gdense source code
gdense_init <- paste(nb_sources[[5]][15], init_n_u_v_from_h,
  sep = "\n", collapse = "\n"
)
gdense_source <- paste(nb_sources[[5]][20:21], collapse = "\n")
# define gsparse source code
gsparse_init <- paste(nb_sources[[7]][15], init_n_u_v_from_h,
  sep = "\n", collapse = "\n"
)
gsparse_source <- paste(nb_sources[[7]][17:18], collapse = "\n")
# define gdense/gsparse mpi source path
gdense_mpi_source <- here::here("mpi", "pagerank_dense.py")
gsparse_mpi_source <- here::here("mpi", "pagerank_sparse.py")

init_h_matrix_str <- function(p) {
  return(paste0("H = init_H(", p, ")"))
}

init_matrix_params <- function(types) {
  params <- tibble(type = types) %>%
    mutate(
      matrix = mapply(Matrix, as.integer(type)),
      h_init = init_h_matrix_str(as.integer(type)),
      shape = case_when(
        type == 5 ~ 212711L,
        T ~ as.integer(10^type)
      )
    ) %>%
    select(-type)
  return(params)
}

init_algo_params <- function(types, nps) {
  params <- tibble(type = types, np = nps) %>%
    mutate(
      algo = mapply(Algo, as.integer(type)),
      init = case_when(np != 0 ~ "", type ==
        1 ~ gdense_init, type == 2 ~ gsparse_init),
      s = case_when(
        type == 1 & np == 0 ~ gdense_source,
        type == 1 & np != 0 ~ gdense_mpi_source,
        type == 2 & np == 0 ~ gsparse_source,
        type == 2 & np != 0 ~ gsparse_mpi_source
      )
    ) %>%
    select(-type)
  return(params)
}

## ---- google-dense ----
power_gdense_output_path <- here::here(
  "mpi", "bin",
  "output", "power_gdense.txt"
)
py_run_string(gdense_init)
power_gdense_result <- read_binary_mpi_output(
  power_gdense_output_path,
  gdense_source
)
writeLines(paste(power_gdense_result[[1]], collapse = "\n"))

## ---- google-sparse ----
power_gsparse_output_path <- here::here(
  "mpi", "bin",
  "output", "power_gsparse.txt"
)
py_run_string(gsparse_init)
power_gsparse_result <- read_binary_mpi_output(
  power_gsparse_output_path,
  gsparse_source
)
writeLines(paste(power_gsparse_result[[1]], collapse = "\n"))

## ---- google-dense-sparse

matrix_params <- init_matrix_params(c(1, 2, 3, 4, 5))
algo_params <- init_algo_params(c(1, 2), rep(0L, 2))
dense_sparse_params <- matrix_params %>%
  crossing(algo_params) %>%
  unite("init", c(h_init, init), sep = "\n")
google_dense_sparse_db <- DBMPI(
  matrices = dense_sparse_params$matrix,
  shapes = dense_sparse_params$shape, algos = dense_sparse_params$algo,
  nps = dense_sparse_params$np, algo_sources = dense_sparse_params$s,
  inits = dense_sparse_params$init
)
google_dense_sparse_df <- google_dense_sparse_db$to_df(c(
  "algo",
  "time", "matrix", "shape", "important", "density",
  "memory"
))

## ---- google-dense-sparse-output

google_dense_sparse_df %>%
  kbl(booktabs = T, escape = F, align = rep(
    "r",
    6
  ), format.args = list(scientific = F), linesep = c(
    "",
    "\\addlinespace"
  ), caption = "dense vs
sparse approach") %>%
  kable_styling(latex_options = c("hold_position"))

## ---- google-dense-sparse-mpi
mpi_matrix_params <- init_matrix_params(c(
  1, 2, 3,
  4, 5
))
mpi_np_default <- 4L
mpi_algo_params <- init_algo_params(
  rep(1:2, each = 2),
  rep(c(0L, mpi_np_default), 2)
)
mpi_dense_sparse_params <- mpi_matrix_params %>%
  crossing(mpi_algo_params) %>%
  unite("init", c(h_init, init), sep = "\n")
mpi_google_dense_sparse_db <- DBMPI(
  matrices = mpi_dense_sparse_params$matrix,
  shapes = mpi_dense_sparse_params$shape, algos = mpi_dense_sparse_params$algo,
  nps = mpi_dense_sparse_params$np, algo_sources = mpi_dense_sparse_params$s,
  inits = mpi_dense_sparse_params$init
)
mpi_google_dense_sparse_df <- mpi_google_dense_sparse_db$to_df(c(
  "matrix",
  "algo", "np", "time"
), F)
mpi_google_dense_sparse_plot <- mpi_google_dense_sparse_df %>%
  ggplot(aes(
    x = matrix, y = time, shape = as.factor(np),
    color = algo, group = interaction(np, algo)
  )) +
  geom_line(alpha = 0.75) +
  geom_point(
    size = 2,
    alpha = 0.9
  ) +
  xlab("Matrix") +
  ylab("Time (s)") +
  theme(legend.position = "bottom") +
  scale_color_manual(values = turbo(4)) +
  scale_shape_discrete(name = "number of processes") +
  scale_x_discrete(labels = c(paste("r", as.integer(10^c(1:4)),
    sep = ""
  ), "ucam2006"))

## ---- matrix-B-example ----

# define init_B
py_run_string(paste(nb_sources[[8]][c(1:8)], collapse = "\n"))
init_b <- function(size) {
  reticulate::py_run_string(sprintf(
    "B_R = init_B(%d)",
    size
  ))
  return(reticulate::py$B_R)
}
b_ex_np <- 3L
b_ex_size <- 3L
b_ex_shape <- b_ex_np * b_ex_size
b_ex <- as.matrix(init_b(b_ex_shape))

## ---- matrix-B-example-latex ----
writeLines(pmatrix(b_ex, T))

## ---- matrix-B-example-graph ----
b_ex_net <- as.network.matrix(b_ex)
b_ex_frac <- MASS::fractions(b_ex[as.edgelist(b_ex_net)])
b_ex_num_den <- str_split_fixed(attr(b_ex_frac, "fracs"),
  "/",
  n = 2
)
b_ex_label <- paste("$\\frac{", b_ex_num_den[, 1],
  "}{", b_ex_num_den[, 2], "}$",
  sep = ""
)
b_ex_net %e% "prob" <- b_ex_label
b_ex_net %v% "names" <- get.vertex.attribute(
  b_ex_net,
  "vertex.names"
)
set.seed(0)
b_ex_plot <- ggraph(b_ex_net) + geom_node_point(
  size = 12.5,
  colour = "grey"
) + geom_node_text(aes(label = names)) +
  theme_void() + geom_edge_arc(aes(
    x = x, y = y,
    xend = xend, yend = yend, end_cap = circle(
      20,
      "pt"
    ), start_cap = circle(40, "pt"), label = prob
  ),
  arrow = arrow(
    length = unit(5, "pt"), type = "closed",
    angle = 20
  ), strength = 0.2, label_size = 5
  ) +
  coord_fixed()

## ---- matrix-B-example-code ----

# define n (shape of B)
py_run_string(sprintf("n = %d", b_ex_shape))
# init B, u, v by n
py_run_string(paste(nb_sources[[8]][c(12, 16, 17)],
  collapse = "\n"
))
b_ex_output_path <- here::here(
  "mpi", "bin", "output",
  "b_ex.txt"
)
b_ex_result <- read_binary_mpi_output(
  b_ex_output_path,
  paste(nb_sources[[8]][19:20], collapse = "\n")
)

## ---- matrix-B-example-output ----
writeLines(paste(b_ex_result[[1]], collapse = "\n"))

## ---- B-dense ----
b_matrix_params <- tibble(matrix = c(Matrix(6)), init = "")
b_mpi_source <- here::here("mpi", "pagerank_mpi.py")
b_algo_params <- tibble(algo = rep(c(Algo(1), Algo(3)),
  each = 14
), np = rep(4L, 2 * 14), s = rep(c(
  gdense_mpi_source,
  b_mpi_source
), each = 14), shape = rep(2^(2:15), 2))
b_params <- b_matrix_params %>%
  crossing(b_algo_params)
b_db <- DBMPI(
  matrices =
    b_params$matrix, shapes =
    b_params$shape, algos =
    b_params$algo, nps =
    b_params$np, algo_sources =
    b_params$s, inits =
    b_params$init
)
b_df <-
  b_db$to_df(c(
    "matrix",
    "algo", "np", "time"
  ), F)
b_plot <-
  b_df %>% ggplot(aes(
    x =
      matrix, y = time, shape = as.factor(np), color
    = algo, group = interaction(np, algo)
  )) +
  geom_line(alpha = 0.75) +
  geom_point(
    size = 2,
    alpha = 0.9
  ) +
  xlab("Matrix") +
  ylab("Time
(s)") +
  theme(
    legend.position = "bottom",
    legend.box = "vertical"
  ) +
  scale_color_manual(values = turbo(4)) +
  scale_shape_discrete(name = "number of
processes") +
  scale_x_discrete(
    labels =
      c(
        paste("r", as.integer(10^c(1:4)), sep = ""),
        "ucam2006"
      )
  )