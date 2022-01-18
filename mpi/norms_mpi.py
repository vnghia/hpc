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


# Basic Linear Algebra operations


def norm1(u):
    return np.sum(np.abs(u))


def norm2(u):
    return np.sqrt(u.dot(u))


def dot_product(u, v):
    return u.dot(v)


# Basic Linear Algebra operations distributed using MPI


def norm1_mpi(u, comm=MPI.COMM_WORLD):
    return  # TO BE FINISHED


def norm2_mpi(u, comm=MPI.COMM_WORLD):
    return  # TO BE FINISHED


def dot_product_mpi(u, v, comm=MPI.COMM_WORLD):
    return  # TO BE FINISHED


#  Main
def main():

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Size initialization
    n = 10

    # Creation of a vector x (same for all processes, created by rank 0 and send to all others)
    if rank == 0:
        x = np.random.uniform(0, 1, n)
        for k in range(1, size):
            comm.Send(x, dest=k, tag=k)
    else:
        x = np.zeros(n)
        comm.Recv(x, source=0, tag=rank)

    # Computation of norms/dot product
    if rank == 0:
        print("x           = ", x)
        print("1-norm      = ", norm1(x))
        print("2-norm      = ", norm2(x))
        print("dot product = ", dot_product(x, x))

    # Distribution over MPI processes along matrix rows
    nd = int(n / size + 0.5)
    start = rank * nd
    end = (rank + 1) * nd
    if rank == size - 1:
        end = n
        nd = end - start
    comm.Barrier()
    print("For rank %d, start = %d, end = %d" % (rank, start, end))

    # Creation of the distributed vector xd
    xd = x[start:end]

    # Computation of MPI norms/dot product
    print("Rank %d, xd          = " % rank, xd)
    print("Rank %d, 1-norm      = " % rank, norm1_mpi(xd))
    print("Rank %d, 2-norm      = " % rank, norm2_mpi(xd))
    print("Rank %d, dot product = " % rank, dot_product_mpi(xd, xd))


if __name__ == "__main__":
    main()
