from __future__ import annotations

import contextlib
import enum
import io
import os
import subprocess
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence, Type, TypeVar, cast

import numpy as np
import pandas as pd
import scipy.sparse as ssp
import sqlalchemy
from sqlalchemy import Column, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.sql.type_api import TypeEngine

    T = TypeVar("T")

    class Enum(TypeEngine[T]):
        def __init__(self, enum: Type[T], **kwargs: Any) -> None:
            ...

else:
    from sqlalchemy import Enum

Base = declarative_base()
ROOT_PATH = os.environ["MPI_ROOT_PATH"]
PYTHON_BIN = os.environ["PYTHON_BIN"]
RAM_LIMIT = float(os.environ.get("RAM_LIMIT", 4))


class Matrix(enum.Enum):
    random10 = enum.auto()
    random100 = enum.auto()
    random1000 = enum.auto()
    random10000 = enum.auto()
    ucam2006 = enum.auto()
    B = enum.auto()


class AlgoMPI(enum.Enum):
    dense = enum.auto()
    sparse = enum.auto()
    B = enum.auto()


MatrixInfo: dict[tuple[Matrix, Optional[int]], tuple[int, int, int, float]] = {}


class BaseMPI(Base):
    __tablename__ = "mpi"

    matrix = Column(Enum(Matrix), nullable=False, primary_key=True)
    shape = Column(Integer, nullable=False, primary_key=True)
    algo = Column(Enum(AlgoMPI), nullable=False, primary_key=True)
    niters = Column(Integer, nullable=False)
    important = Column(Integer, nullable=False)
    time = Column(Float, nullable=False)
    nonzero = Column(Integer, nullable=False, primary_key=False)
    dangling = Column(Integer, nullable=False, primary_key=False)
    np = Column(Integer, nullable=False, primary_key=True)
    multiple_times = Column(Integer, nullable=False, primary_key=True)


def create_engine_and_table(base, path) -> sqlalchemy.engine.Engine:
    engine = sqlalchemy.create_engine(
        f"sqlite+pysqlite:///{path}", future=True, echo=False
    )
    base.metadata.create_all(engine)
    return engine


@dataclass(eq=True)
class BinaryMPI:
    Session: Optional[sessionmaker] = field(hash=False, compare=False, repr=False)

    matrix: Matrix = Matrix.random10
    shape: int = 10
    algo: AlgoMPI = AlgoMPI.dense
    np: int = 2
    nonzero: int = 0
    dangling: int = 0
    multiple_times: int = 100

    init: Optional[str] = None
    algo_source: str = ""
    raw_outputs: Optional[Sequence[str]] = field(
        default=None, hash=False, compare=False, repr=False
    )

    niters: Optional[int] = field(default=None, hash=False, compare=False)
    important: Optional[int] = field(default=None, hash=False, compare=False)
    time: Optional[float] = field(default=None, hash=False, compare=False)

    in_database: bool = field(default=False, hash=False, compare=False, repr=False)

    def __has_result(self: BinaryMPI):
        return self.niters and self.important and self.time is not None

    def clear_result(self: BinaryMPI):
        self.niters = None
        self.important = None
        self.time = None

    def __query(
        self: BinaryMPI, session: sqlalchemy.orm.Session
    ) -> sqlalchemy.orm.Query[BaseMPI]:
        query = (
            session.query(BaseMPI)
            .filter(BaseMPI.matrix == self.matrix)
            .filter(BaseMPI.shape == self.shape)
            .filter(BaseMPI.algo == self.algo)
            .filter(BaseMPI.np == self.np)
            .filter(BaseMPI.multiple_times == self.multiple_times)
        )
        return query

    def __post_init__(self: BinaryMPI):
        if self.matrix == Matrix.B:
            assert self.shape % self.np == 0
        if not self.Session:
            return

        session: sqlalchemy.orm.Session
        with self.Session() as session:
            query = self.__query(session)
            res = query.one_or_none()
            if res:
                self.in_database = True
                self.niters = res.niters
                self.important = res.important
                self.time = float(res.time)

    def __run_raw(self: BinaryMPI):
        if self.__has_result():
            return
        if self.np == 0:
            f = io.StringIO()
            np.random.seed(0)
            with contextlib.redirect_stdout(f):
                exec(self.algo_source)
            self.raw_outputs = f.getvalue().splitlines()
        else:
            args = [
                "mpirun",
                "-np",
                str(self.np),
                "--oversubscribe",
                PYTHON_BIN,
                self.algo_source,
            ]
            if self.matrix != Matrix.B:
                args.append(str(self.matrix.value))
            else:
                if self.algo != AlgoMPI.B:
                    args.append(str(self.shape))
                    args.append("--B")
                else:
                    args.append(str(int(self.shape / self.np)))
            args.append("--log")
            self.raw_outputs = (
                subprocess.run(
                    args, capture_output=True, env=dict(os.environ, TMPDIR="/tmp")
                )
                .stdout.decode("utf-8")
                .splitlines()
            )
        return self.raw_outputs

    def parse(self: BinaryMPI):
        if self.__has_result():
            return
        assert self.raw_outputs
        outputs = self.raw_outputs
        for output in outputs:
            if output.startswith("number of iterations = "):
                _, rhs = output.strip().split(" = ")
                self.niters = int(rhs)
            elif output.startswith("highest pagerank = "):
                _, rhs = output.strip().split(" = ")
                self.important = int(rhs) + 1
            elif output.startswith("Computational time = "):
                _, rhs = output.strip().split(" = ")
                self.time = float(rhs)

    def __to_sql(self: BinaryMPI) -> Optional[BaseMPI]:
        assert self.__has_result()
        assert self.important
        assert self.niters
        if not self.in_database:
            return BaseMPI(
                matrix=self.matrix,
                shape=self.shape,
                algo=self.algo,
                niters=self.niters,
                important=self.important,
                time=cast(Decimal, self.time),
                np=self.np,
                nonzero=self.nonzero,
                dangling=self.dangling,
                multiple_times=self.multiple_times,
            )
        else:
            return None

    def run(self: BinaryMPI):
        try:
            if self.__has_result():
                return
            niterss = np.empty(self.multiple_times)
            importants = np.empty_like(niterss)
            times = np.empty_like(niterss)
            if self.init and not self.np:
                exec(self.init, globals())
            for i in range(self.multiple_times):
                print(i)
                self.__run_raw()
                self.parse()
                assert self.__has_result()
                niterss[i] = self.niters
                importants[i] = self.important
                times[i] = self.time
                self.clear_result()
            assert np.all(importants == importants[0])
            self.niters = int(np.average(niterss))
            self.important = importants[0]
            self.time = np.average(times).astype(float)
        except BaseException as exception:
            print("Current binary options: ", self)
            raise exception

    def insert(
        self: BinaryMPI,
        upsert: bool = False,
    ):
        assert self.Session
        res = self.__to_sql()
        if res:
            session: sqlalchemy.orm.Session
            with self.Session() as session:
                if not self.in_database:
                    session.add(res)
                elif upsert:
                    query = self.__query(session)
                    query.update(
                        {
                            "niters": self.niters,
                            "important": self.important,
                            "time": self.time,
                        },
                        synchronize_session="fetch",
                    )
                session.commit()


KT = tuple[Matrix, int, AlgoMPI, int]


@dataclass
class DBMPI:
    matrices: Sequence[Matrix] = field(default_factory=lambda: [Matrix.random10])
    shapes: Sequence[Optional[int]] = field(default_factory=lambda: [10])
    algos: Sequence[AlgoMPI] = field(default_factory=lambda: [AlgoMPI.dense])
    nps: Sequence[int] = field(default_factory=lambda: [2])
    multiple_times: int = 100

    inits: Sequence[Optional[str]] = field(default_factory=list)
    algo_sources: Sequence[str] = field(default_factory=list)
    upsert: bool = False
    outputs: set[KT] = field(default_factory=set)
    Binaries: ClassVar[dict[KT, BinaryMPI]] = {}
    Outputs: ClassVar[dict[KT, tuple[int, int, float]]] = {}

    Engine: ClassVar[sqlalchemy.engine.Engine] = create_engine_and_table(
        Base, os.path.join(ROOT_PATH, "mpi.db")
    )
    Session: ClassVar[sessionmaker] = sessionmaker(bind=Engine)
    Colnames: ClassVar[Sequence[str]] = [
        "matrix",
        "shape",
        "algo",
        "niters",
        "important",
        "time",
        "np",
        "nonzero",
        "dangling",
        "density",
        "memory",
    ]
    MatrixDtype: ClassVar[pd.CategoricalDtype] = pd.CategoricalDtype(
        [matrix.name for matrix in Matrix], ordered=True
    )
    AlgoDtype: ClassVar[pd.CategoricalDtype] = pd.CategoricalDtype(
        [algo.name for algo in AlgoMPI], ordered=True
    )
    DFDtype: ClassVar[dict[str, type]] = {
        "matrix": MatrixDtype,
        "shape": int,
        "algo": AlgoDtype,
        "niters": int,
        "important": int,
        "time": float,
        "np": int,
        "nonzero": int,
        "dangling": int,
        "density": str,
        "memory": str,
    }

    df: pd.DataFrame = field(
        default_factory=lambda Colnames=Colnames: pd.DataFrame(columns=Colnames)  # type: ignore
    )

    def __cast_df(self: DBMPI, df: Optional[pd.DataFrame] = None):
        df = df if df is not None else self.df
        for k, v in self.DFDtype.items():
            df[k] = df[k].astype(v)
        return df

    def get_info_matrix(self: DBMPI, matrix: Matrix, shape: Optional[int]):
        if matrix == Matrix.B and shape:
            MatrixInfo[(matrix, shape)] = (
                (shape - 1) + 2 + (shape - 3) * 3 + 2,
                0,
                shape,
                shape * shape * np.dtype(np.float64).itemsize / 1e9,
            )
        if (matrix, shape) not in MatrixInfo:
            M = ssp.load_npz(os.path.join(ROOT_PATH, f"{matrix.name}.npz"))
            dangling = np.count_nonzero(np.where(np.sum(M, axis=1).A1 == 0, 1, 0))
            real_shape = M.shape[0]
            memory = real_shape * real_shape * M.dtype.itemsize / 1e9
            MatrixInfo[(matrix, shape)] = (M.size, dangling, real_shape, memory)
        return MatrixInfo[(matrix, shape)]

    def __post_init__(self: DBMPI):
        self.df = self.__cast_df()
        for (matrix, shape, algo, num_process, init, source) in zip(
            self.matrices,
            self.shapes,
            self.algos,
            self.nps,
            self.inits,
            self.algo_sources,
        ):
            nonzero, dangling, shape, memory = self.get_info_matrix(matrix, shape)
            key = (matrix, shape, algo, num_process)
            print(key)
            if key not in self.Outputs:
                binary = BinaryMPI(
                    Session=self.Session,
                    matrix=matrix,
                    shape=shape,
                    algo=algo,
                    np=num_process,
                    nonzero=nonzero,
                    dangling=dangling,
                    multiple_times=self.multiple_times,
                    init=init,
                    algo_source=source,
                )
                if algo == AlgoMPI.dense and memory > RAM_LIMIT:
                    binary.niters = -1
                    binary.important = -1
                    binary.time = np.inf
                else:
                    binary.run()
                binary.insert(self.upsert)
                assert binary.niters and binary.important and binary.time is not None
                self.Outputs[key] = (binary.niters, binary.important, binary.time)
                self.Binaries[key] = binary

            niters, important, time = self.Outputs[key]
            if key not in self.outputs:
                new_df = pd.DataFrame(
                    {
                        "matrix": matrix.name,
                        "shape": shape,
                        "algo": algo.name,
                        "niters": niters,
                        "important": important,
                        "time": time,
                        "np": num_process,
                        "nonzero": nonzero,
                        "dangling": dangling,
                        "density": f"{1000 * nonzero / (shape ** 2):.2f} \\textperthousand",
                        "memory": "%.2f" % memory if algo == AlgoMPI.dense else "",
                    },
                    index=[0],
                )
                new_df = self.__cast_df(new_df)
                self.df = self.df.append(new_df, ignore_index=True)
                self.outputs.add(key)

    def to_df(
        self: DBMPI, colnames: Optional[Sequence[str]] = None, table: bool = True
    ):
        colnames = colnames or self.Colnames
        df = self.df[colnames].reset_index(drop=True)
        if table:
            df = df.replace(0, "")
        return df
