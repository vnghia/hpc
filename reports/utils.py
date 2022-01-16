import numpy as np


def init_matrix(nrow: int, ncol: int, coff: float) -> np.ndarray:
    m = np.empty((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            m[i, j] = coff * (i + 1 + j + 1) / nrow / ncol
    return m


def pmatrix(a: np.ndarray) -> str:
    if len(a.shape) != 2:
        raise ValueError("pmatrix can only display two dimensions")
    latex = r"\begin{pmatrix}" + "\n"
    nrow, ncol = a.shape
    for i in range(nrow):
        for j in range(ncol):
            fraction, integer = np.modf(a[i, j])
            if fraction == 0:
                latex += str(int(integer))
            else:
                latex += str(a[i, j]).replace(".", ",")
            if j != ncol - 1:
                latex += " & "
        if i != nrow - 1:
            latex += r"\\" + "\n"
    latex += "\n" + r"\end{pmatrix}"
    return latex
