read_from_commit <- function(path, range, commit = "HEAD") {
  system2("git", c("checkout", commit, path),
    stdout = NULL,
    stderr = NULL
  )
  lines <- readLines(path)[range]
  system2("git", c("checkout", "HEAD", path),
    stdout = NULL,
    stderr = NULL
  )
  return(lines)
}

to_float_str <- function(num) {
  return(gsub("\\.", ",", num))
}

replace_inline_code <- function(str, value) {
  return(paste("`", str, " = ", value, "`"))
}