import os

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


#  Main
def matvec_mpi_main(i=1, mpi_all_to_all=mpi_all_to_all_gather, log=False):

    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    # Matrix and Size initialization
    H = init_H(i)
    n = H.shape[0]

    if rank == 0 and log:
        print(ssp.csr_matrix.todense(H))

    # Creation of a vector x (same for all processes)
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

    # Creation of the distributed vector xd and distributed matrix Hd (along rows)
    xd = x[start:end]
    Hd = H.T[start:end]

    # Computation of MPI norms/dot product
    rd = matvec_mpi(Hd, xd, n, comm, mpi_all_to_all)
    if log:
        print("Rank %d, xd           = " % rank, xd)
        print("Rank %d, Hd.T.dot(xd) = " % rank, rd)

    result_mpi = np.zeros(n) if rank == 0 else None
    comm.Gather(rd, result_mpi, root=0)

    # Computation of matrix/vector product
    if rank == 0:
        result = matvec(H.T, x)
        np.testing.assert_almost_equal(result_mpi, result)
        if log:
            print("x           = ", x)
            print("H.T.dot(x)  = ", result)
            print("matvec_mpi  = ", result_mpi)


if __name__ == "__main__":
    for mpi_all_to_all in [
        mpi_all_to_all_gather,
        mpi_all_to_all_allgather,
        mpi_all_to_all_isend_irecv,
    ]:
        matvec_mpi_main(1, mpi_all_to_all)
