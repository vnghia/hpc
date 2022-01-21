import subprocess
from fractions import Fraction
from typing import Optional

import nbformat
import numpy as np


def init_matrix(nrow: int, ncol: int, coff: float) -> np.ndarray:
    m = np.empty((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            m[i, j] = coff * (i + 1 + j + 1) / nrow / ncol
    return m


def pmatrix(a: np.ndarray, is_frac: bool = False) -> str:
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
                if is_frac:
                    frac = Fraction(str(a[i, j])).limit_denominator()
                    latex += (
                        r"\frac{"
                        + str(frac.numerator)
                        + "}{"
                        + str(frac.denominator)
                        + "}"
                    )
                else:
                    latex += str(a[i, j]).replace(".", ",")
            if j != ncol - 1:
                latex += " & "
        if i != nrow - 1:
            latex += r"\\" + "\n"
    latex += "\n" + r"\end{pmatrix}"
    return latex


def read_from_notebook(
    path: str, tags: Optional[list[str]] = None, commit: str = "HEAD"
) -> list[list[str]]:
    subprocess.run(["git", "checkout", commit, path], check=True)
    sources = []
    with open(path, "r") as f:
        nb = nbformat.read(f, nbformat.NO_CONVERT)
        cells = nb["cells"]
        for cell in cells:
            if not tags:
                if cell["cell_type"] == "code":
                    sources.append(cell["source"].splitlines())
            else:
                metadata = cell["metadata"]
                if "tags" in metadata:
                    for tag in tags:
                        if tag in metadata["tags"]:
                            sources.append(cell["source"].splitlines())
                            break
    subprocess.run(["git", "checkout", "HEAD", path], check=True)
    return sources
