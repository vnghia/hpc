#!/usr/bin/env python3

import numpy as np
import scipy.sparse as ssp
import matplotlib.pyplot as plt
from mpi4py import MPI
import time
import sys
import os

# Some options
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
np.set_printoptions(precision=3, linewidth=np.inf)

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


# Matrix/Vector product
def matvec(H, x):
    return H @ x


# Matrix/Vector product distributed using MPI
def matvec_mpi(H, x, n, comm=MPI.COMM_WORLD):
    xall = np.zeros(n)
    xall = mpi_all_to_all(xall, x)
    return H @ xall


# Mpi All to All
def mpi_all_to_all(x, xd, comm=MPI.COMM_WORLD):
    # Gather version
    # TO BE FINISHED
    # AllGather version
    # TO BE FINISHED
    # ISend/IRecv version
    # TO BE FINISHED
    return x


#  Main
def main():

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # MAtrix and Size initialization
    H = init_H(1)
    n = H.shape[0]

    if rank == 0:
        print(ssp.csr_matrix.todense(H))

    # Creation of a vector x (same for all processes)
    if rank == 0:
        x = np.random.uniform(0, 1, n)
        for k in range(1, size):
            comm.Send(x, dest=k, tag=k)
    else:
        x = np.zeros(n)
        comm.Recv(x, source=0, tag=rank)

    # Computation of matrix/vector product
    if rank == 0:
        print("x           = ", x)
        print("H.T.dot(x)  = ", matvec(H.T, x))

    # Distribution over MPI processes along matrix rows
    nd = int(n / size + 0.5)
    start = rank * nd
    end = (rank + 1) * nd
    if rank == size - 1:
        end = n
        nd = end - start
    comm.Barrier()
    print("For rank %d, start = %d, end = %d" % (rank, start, end))

    # Creation of the distributed vector xd and distributed matrix Hd (along rows)
    xd = x[start:end]
    Hd = H.T[start:end, :]

    # Computation of MPI norms/dot product
    print("Rank %d, xd           = " % rank, xd)
    print("Rank %d, Hd.T.dot(xd) = " % rank, matvec_mpi(Hd, xd, n))


if __name__ == "__main__":
    main()
