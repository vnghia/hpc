import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssp
from mpi4py import MPI

# Some options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
np.set_printoptions(precision=3)


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


# Basic Linear Algebra operations


def norm1(u, comm=MPI.COMM_WORLD):
    return  # TO BE FINISHED


def norm2(u, comm=MPI.COMM_WORLD):
    return  # TO BE FINISHED


def dot_product(u, v, comm=MPI.COMM_WORLD):
    return  # TO BE FINISHED


# Matrix/Vector product
def matvec(H, x):
    return H @ x


# Mpi All to All
def mpi_all_to_all(u, v, comm=MPI.COMM_WORLD):
    # Gather version
    # TO BE FINISHED
    # AllGather version
    # TO BE FINISHED
    # ISend/IRecv version
    # TO BE FINISHED
    return u


# Define a function that return the vector index of the dangling nodes
def dangling_nodes(H):
    return np.where(np.sum(H, axis=1).A1 == 0.0, 1.0, 0.0)


# Define the Google matrix in dense format
def google_matrix_dense(M, alpha=0.85):
    # TO BE FINISHED
    return alpha * M.A  # + #TO BE FINISHED


# Power Iteration Method
def power_iteration(
    matvec, G, u, v, start, end, comm=MPI.COMM_WORLD, tol=1e-8, itmax=200
):
    t = time.time()
    it = 0
    diff_norm = 2 * tol
    while it < itmax and diff_norm > tol:
        # TO BE FINISHED
        it += 1
    if comm.rank == 0:
        print("\nnumber of iterations %3d" % it)
        print("residual = %7.2e" % diff_norm)
        print("eigenvalue = %11.6e" % l)
        print("eigenvector = ", u)
        print("Computational time = ", time.time() - t)
    return l, v


#  Main
def main():

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Matrix and Size initialization
    H = init_H(1)
    n = H.shape[0]

    # Distribution over MPI processes along matrix rows
    nd = int(n / size + 0.5)
    start = rank * nd
    end = (rank + 1) * nd
    if rank == size - 1:
        end = n
        nd = end - start

    # Construct the dense Google matrix
    Gdense = google_matrix_dense(H)

    # Initialize vectors
    u = np.zeros(n)
    v = np.random.uniform(0, 1, nd)

    # Call the power iteration method with Google matrix
    u = mpi_all_to_all(u, v)
    l, u = power_iteration(matvec, Gdense, u, v, start, end)


if __name__ == "__main__":
    main()
