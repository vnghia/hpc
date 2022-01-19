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


def sum(u, comm=MPI.COMM_WORLD):
    s = np.sum(u)
    result = np.empty(1)
    comm.Allreduce(s, result, MPI.SUM)
    return result


def norm1(u, comm=MPI.COMM_WORLD):
    norm = np.sum(np.abs(u))
    result = np.empty(1)
    comm.Allreduce(norm, result, MPI.SUM)
    return result


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


# MPI All to All Gather version
def mpi_all_to_all_gather(x, xd, comm=MPI.COMM_WORLD):
    for i in range(comm.size):
        comm.Gather(xd, x, root=i)
    return x


# MPI All to All Allgather version
def mpi_all_to_all_allgather(x, xd, comm=MPI.COMM_WORLD):
    comm.Allgather(xd, x)
    return x


# MPI All to All ISend/IRecv version
def mpi_all_to_all_isend_irecv(x, xd, comm=MPI.COMM_WORLD):
    size = comm.size
    req_send = np.full(size, MPI.REQUEST_NULL)
    req_recv = np.full(size, MPI.REQUEST_NULL)

    for k in range(size):
        req_send[k] = comm.Isend(xd, dest=k, tag=comm.rank)

    shapes = comm.allgather(np.shape(xd)[0])
    start, end = 0, shapes[0]
    for k in range(size):
        req_recv[k] = comm.Irecv(x[start:end], source=k, tag=k)
        if k < size - 1:
            start = end
            end += shapes[k + 1]

    assert MPI.Request.Waitall(req_send)
    assert MPI.Request.Waitall(req_recv)
    return x


# Matrix/Vector product distributed using MPI
def matvec_mpi(H, x, n, comm=MPI.COMM_WORLD, mpi_all_to_all=mpi_all_to_all_gather):
    xall = np.zeros(n)
    xall = mpi_all_to_all(xall, x, comm)
    return H @ xall


# Define a function that return the vector index of the dangling nodes
def dangling_nodes(H):
    return np.where(np.sum(H, axis=1).A1 == 0.0, 1.0, 0.0)


# Define the Google matrix in spare format
def matvec_sparse(M, x, d, start, end, alpha=0.85):
    M = M[start:end]
    d = d[start:end]
    xd = x[start:end]
    m, n = np.shape(M)
    e = np.ones(m)
    return (
        alpha * M.dot(x)
        + alpha / n * sum(d.T.dot(xd))
        + (1 - alpha) / n * sum(e.T.dot(xd))
    )


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
):
    t = time.time()
    it = 0
    diff_norm = 2 * tol
    d = dangling_nodes(G)
    l = 0
    while it < itmax and diff_norm > tol:
        v = matvec(G.T, u, d, start, end)
        l = dot_product(v, u[start:end], comm)
        v = (1 / norm2(v, comm)) * v
        diff_norm = norm2(u[start:end] - v, comm)
        u = mpi_all_to_all(u, v, comm)
        it += 1
    if comm.rank == 0 and log:
        print("\nnumber of iterations %3d" % it)
        print("residual = %7.2e" % diff_norm)
        print("eigenvalue = %11.6e" % l)
        print("eigenvector = ", u)
        print("Computational time = ", time.time() - t)
    return l, v


#  Main
def pagerank_sparse_main(mpi_all_to_all=mpi_all_to_all_allgather, log=False):

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

    # Initialize vectors
    u = np.zeros(n)
    v = np.random.uniform(0, 1, nd)

    # Call the power iteration method with Google matrix
    u = mpi_all_to_all(u, v)
    ev, u = power_iteration(
        matvec_sparse, H, u, v, start, end, comm, mpi_all_to_all, log=log
    )
    np.testing.assert_almost_equal(ev, 1)


if __name__ == "__main__":
    pagerank_sparse_main()
