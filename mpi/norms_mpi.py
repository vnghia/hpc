import os

import numpy as np
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
    norm = norm1(u)
    result = np.empty(1)
    comm.Allreduce(norm, result, MPI.SUM)
    return result


def norm2_mpi(u, comm=MPI.COMM_WORLD):
    norm = dot_product(u, u)
    result = np.empty(1)
    comm.Allreduce(norm, result, MPI.SUM)
    return np.sqrt(result)


def dot_product_mpi(u, v, comm=MPI.COMM_WORLD):
    product = dot_product(u, v)
    result = np.empty(1)
    comm.Allreduce(product, result, MPI.SUM)
    return result


#  Main
def norms_mpi_main(log=False):

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

    # Distribution over MPI processes along matrix rows
    nd = int(n / size + 0.5)
    start = rank * nd
    end = (rank + 1) * nd
    if rank == size - 1:
        end = n
        nd = end - start
    comm.Barrier()
    if log:
        print("For rank %d, start = %d, end = %d" % (rank, start, end))

    # Creation of the distributed vector xd
    xd = x[start:end]

    # Computation of MPI norms/dot product
    result_norm1_mpi = norm1_mpi(xd)
    result_norm2_mpi = norm2_mpi(xd)
    result_dot_product_mpi = dot_product_mpi(xd, xd)
    if log:
        print("Rank %d, xd          = " % rank, xd)
        print("Rank %d, 1-norm      = " % rank, result_norm1_mpi)
        print("Rank %d, 2-norm      = " % rank, result_norm2_mpi)
        print("Rank %d, dot product = " % rank, result_dot_product_mpi)

    # Computation of norms/dot product
    if rank == 0:
        result_norm1 = norm1(x)
        result_norm2 = norm2(x)
        result_dot_product = dot_product(x, x)
        np.testing.assert_almost_equal(result_norm1_mpi, result_norm1)
        np.testing.assert_almost_equal(result_norm2_mpi, result_norm2)
        np.testing.assert_almost_equal(result_dot_product_mpi, result_dot_product)
        if log:
            print("x           = ", x)
            print("1-norm      = ", result_norm1)
            print("2-norm      = ", result_norm2)
            print("dot product = ", result_dot_product)


if __name__ == "__main__":
    norms_mpi_main()
