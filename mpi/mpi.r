## ---- init ----

library(reticulate)

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