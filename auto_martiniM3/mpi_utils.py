"""MPI helpers for AutoMartiniM3.

Provides a graceful fallback when mpi4py is unavailable or the script is
launched outside of mpirun (size == 1). All MPI-aware code paths should go
through these helpers so the serial path stays identical.
"""

_MPI_STATE = None


def get_mpi():
    """Return (comm, rank, size). If mpi4py is not installed, returns
    (None, 0, 1) and behaviour is purely serial."""
    global _MPI_STATE
    if _MPI_STATE is not None:
        return _MPI_STATE
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        _MPI_STATE = (comm, comm.Get_rank(), comm.Get_size())
    except Exception:
        _MPI_STATE = (None, 0, 1)
    return _MPI_STATE


def is_root():
    _, rank, _ = get_mpi()
    return rank == 0


def bcast(obj):
    """Broadcast a picklable object from rank 0 to all ranks."""
    comm, _, size = get_mpi()
    if comm is None or size == 1:
        return obj
    return comm.bcast(obj, root=0)


def barrier():
    comm, _, size = get_mpi()
    if comm is not None and size > 1:
        comm.Barrier()
