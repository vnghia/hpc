import argparse
import functools
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
np.random.seed(0)


# Initialize the matrix with external data
def init_H(p=1):
    if p == 1:
        return ssp.load_npz("random10.npz")
    elif p == 2:
        return ssp.load_npz("random100.npz")
    elif p == 3:
        return ssp.load_npz("random1000.npz")
    elif p == 4:
        return ssp.load_npz("random10000.npz")
    else:
        return ssp.load_npz("ucam2006.npz")


def init_B(n):
    B = ssp.lil_matrix((n, n), dtype=np.float64)
    B.setdiag(np.ones(n - 1), -1)
    B[1:, 0] = 1.0
    B = B + B.T
    B = B * ssp.diags(1 / B.sum(axis=0).A.ravel())
    return B


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


# MPI All to All Allgather version
def mpi_all_to_all_allgather(x, xd, comm=MPI.COMM_WORLD):
    comm.Allgather(xd, x)
    return x


# MPI All to All Allgatherv version
def mpi_all_to_all_allgatherv(x, xd, counts, disps, comm=MPI.COMM_WORLD):
    comm.Allgatherv([xd, np.size(xd), MPI.DOUBLE], [x, counts, disps, MPI.DOUBLE])
    return x


# Define a function that return the vector index of the dangling nodes
def dangling_nodes(H):
    return np.where(np.sum(H, axis=1).A1 == 0.0, 1.0, 0.0)


# Define the Google matrix in dense format
def google_matrix_dense(M, alpha=0.85):
    d = dangling_nodes(M)
    n = np.shape(M)[0]
    e = np.ones(n)
    return alpha * M.A + alpha / n * np.outer(d, e) + (1 - alpha) / n * np.outer(e, e)


# Power Iteration Method
def power_iteration(
    matvec,
    G,
    u,
    v,
    start,
    end,
    comm=MPI.COMM_WORLD,
    mpi_all_to_all=mpi_all_to_all_allgather,
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
        # Note that `dot_product` and `norm2` are MPI-enabled functions.
        # It means that we are calculating across processes.
        # Therefore, although `v` is local to one process,
        # `l`, `norm2(v)` and `diff_norm` are global,
        # which is needed for the algorithm to converge.
        v = matvec(G[start:end], u)
        l = dot_product(u[start:end], v, comm)
        # Notably this because if we use a local `norm2`,
        # the normalization of `v` will be wrong.
        v = (1 / norm2(v, comm)) * v
        diff_norm = norm2(u[start:end] - v, comm)
        u = mpi_all_to_all(u, v, comm=comm)
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
def pagerank_dense_main(i=1, B=False, log=False, log_matrix=False):

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Matrix and Size initialization
    H = init_H(i) if not B else init_B(i).T
    n = H.shape[0]

    # Distribution over MPI processes along matrix rows
    nd = int(np.ceil(n / size))
    start = rank * nd
    end = (rank + 1) * nd
    mpi_all_to_all = mpi_all_to_all_allgather
    if n % size:
        counts = [nd] * size
        counts[-1] = n - nd * (size - 1)
        disps = np.zeros(size, dtype=int)
        disps[1:] = np.cumsum(counts)[:-1]
        mpi_all_to_all = functools.partial(
            mpi_all_to_all_allgatherv, counts=counts, disps=disps
        )
    if rank == size - 1:
        end = n
        nd = end - start

    # Construct the dense Google matrix
    Gdense = google_matrix_dense(H) if not B else H

    # Initialize vectors
    u = np.zeros(n)
    v = np.random.uniform(0, 1, nd)

    # Call the power iteration method with Google matrix
    u = mpi_all_to_all(u, v, comm=comm)
    l, u = power_iteration(
        matvec,
        Gdense.T,
        u,
        v,
        start,
        end,
        comm,
        mpi_all_to_all,
        log=log,
        log_matrix=log_matrix,
    )
    np.testing.assert_almost_equal(l, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int)
    parser.add_argument("--B", action=argparse.BooleanOptionalAction)
    parser.add_argument("--log", action=argparse.BooleanOptionalAction)
    parser.add_argument("--matrix", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    pagerank_dense_main(args.i, args.B, args.log, args.matrix)
