import argparse
import os
import time

import numpy as np
import scipy.sparse as ssp
from mpi4py import MPI

# Some options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
np.set_printoptions(precision=3)


# Initialize the matrix
def init_B_mpi(n, comm=MPI.COMM_WORLD):

    size = comm.size
    rank = comm.rank

    Bglob = ssp.lil_matrix((n * size, n * size), dtype=np.float64)
    Bglob.setdiag(np.ones(n * size - 1), -1)
    Bglob[1:, 0] = 1.0
    Bglob = Bglob + Bglob.T
    Bglob = Bglob * ssp.diags(1 / Bglob.sum(axis=0).A1)

    if rank == 0:
        B = ssp.lil_matrix((n, n * size), dtype=np.float64)
        B = Bglob[0:n, :]
        u = np.zeros(n * size)
        start = 0
        end = n
    elif rank == size - 1:
        B = ssp.lil_matrix((n, n + 2), dtype=np.float64)
        B[:, 0] = Bglob[n * rank : n * rank + n, 0]
        B[:, 1 : n + 2] = Bglob[n * rank : n * rank + n, n * rank - 1 : n * rank + n]
        u = np.zeros(n + 2)
        start = 2
        end = n + 2
    else:
        B = ssp.lil_matrix((n, n + 3), dtype=np.float64)
        B[:, 0] = Bglob[n * rank : n * rank + n, 0]
        B[:, 1 : n + 3] = Bglob[
            n * rank : n * rank + n, n * rank - 1 : n * rank + n + 1
        ]
        u = np.zeros(n + 3)
        start = 2
        end = n + 2

    return B, u, start, end


# Basic Linear Algebra operations


def norm2(u, comm=MPI.COMM_WORLD):
    norm = u.dot(u)
    result = np.empty(1)
    comm.Allreduce(norm, result, MPI.SUM)
    return np.sqrt(result)


def dot_product(u, v, comm=MPI.COMM_WORLD):
    product = u.dot(v)
    result = np.empty(1)
    comm.Allreduce(product, result, MPI.SUM)
    return result


# Matrix/Vector product
def matvec(H, x):
    return H @ x


# MPI All to All
def mpi_all_to_all(x, xd, comm=MPI.COMM_WORLD):
    rank = comm.rank
    size = comm.size
    n = np.shape(xd)[0]
    comm.Gather(xd, x, root=0)
    xbcast = np.empty(n * size) if rank != 0 else x
    comm.Bcast(xbcast, root=0)
    if rank == size - 1:
        x[0] = xbcast[0]
        x[1 : n + 2] = xbcast[n * rank - 1 : n * rank + n]
    elif rank != 0:
        x[0] = xbcast[0]
        x[1 : n + 3] = xbcast[n * rank - 1 : n * rank + n + 1]
    return x


# Power Iteration Method
def power_iteration(
    matvec,
    G,
    u,
    v,
    start,
    end,
    comm=MPI.COMM_WORLD,
    tol=1e-8,
    itmax=200,
    log=False,
    log_matrix=False,
):
    t = time.time()
    it = 0
    diff_norm = 2 * tol
    l = 0
    while it < itmax and diff_norm > tol:
        v = matvec(G, u)
        l = dot_product(u[start:end], v, comm)
        v = (1 / norm2(v, comm)) * v
        diff_norm = norm2(u[start:end] - v, comm)
        u = mpi_all_to_all(u, v, comm)
        it += 1
    if comm.rank == 0 and log:
        print("number of iterations = %3d" % it)
        print("residual = %7.2e" % diff_norm)
        print("eigenvalue = %11.6e" % l)
        if log_matrix:
            print(f"eigenvector = {u}")
        print(f"highest pagerank = {np.argmax(u)}")
        print(f"Computational time = {time.time() - t}")
    return l, v


#  Main
def pagerank_mpi_main(n=3, log=False, log_matrix=False):

    # MPI initialization
    comm = MPI.COMM_WORLD

    # Distribution over MPI processes along matrix rows
    B, u, start, end = init_B_mpi(n, comm)

    # Initialize vectors
    v = np.random.uniform(0, 1, n)

    # Call the power iteration method with Google matrix
    u = mpi_all_to_all(u, v, comm)
    l, u = power_iteration(
        matvec, B, u, v, start, end, comm, log=log, log_matrix=log_matrix
    )
    np.testing.assert_almost_equal(l, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("--log", action=argparse.BooleanOptionalAction)
    parser.add_argument("--matrix", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    pagerank_mpi_main(args.n, args.log, args.matrix)
