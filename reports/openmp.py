from __future__ import annotations

import enum
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence, Type, TypeVar, cast

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import Boolean, Column, Float, Integer, String
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


class Algo(enum.Enum):
    naive = enum.auto()
    saxpy = enum.auto()
    tiled = enum.auto()
    blas = enum.auto()


class Schedule(enum.Enum):
    static = enum.auto()
    dynamic = enum.auto()
    guided = enum.auto()
    auto = enum.auto()


class Compiler(enum.Enum):
    gcc = enum.auto()
    clang = enum.auto()


class OpenMP(Base):
    __tablename__ = "openmp"

    algo = Column(Enum(Algo), primary_key=True)
    time = Column(Float, nullable=False)
    norm = Column(Float, nullable=False)
    gflops = Column(Float, nullable=False)
    M = Column(Integer, nullable=False, primary_key=True)
    K = Column(Integer, nullable=False, primary_key=True)
    N = Column(Integer, nullable=False, primary_key=True)
    block = Column(Integer, nullable=True, primary_key=True)
    omp = Column(Boolean, nullable=True, primary_key=True)
    num_threads = Column(Integer, nullable=True, primary_key=True)
    schedule = Column(Enum(Schedule), nullable=True, primary_key=True)
    chunk = Column(Integer, nullable=True, primary_key=True)
    compiler = Column(Enum(Compiler), nullable=False, primary_key=True)
    multiple_times = Column(Integer, nullable=False, primary_key=True)
    hash = Column(String(10), nullable=False)


def create_engine_and_table(base) -> sqlalchemy.engine.Engine:
    engine = sqlalchemy.create_engine(
        "sqlite+pysqlite:///openmp.db", future=True, echo=False
    )
    base.metadata.create_all(engine)
    return engine


@dataclass(eq=True)
class BinaryOpenMP:
    Session: Optional[sessionmaker] = field(hash=False, compare=False, repr=False)

    algo: Algo = Algo.naive
    M: int = 4
    K: int = 8
    N: int = 4
    block: Optional[int] = 4
    omp: Optional[bool] = False
    num_threads: Optional[int] = 4
    schedule: Optional[Schedule] = Schedule.static
    chunk: Optional[int] = 0
    compiler: Compiler = Compiler.clang if sys.platform == "darwin" else Compiler.gcc
    multiple_times: int = 100

    hash: Optional[str] = None

    source_path: str = field(
        default="../openmp/blas3.c", hash=False, compare=False, repr=False
    )
    executable_prefix: str = field(
        default="bin/openmp/", hash=False, compare=False, repr=False
    )

    exec_path: Optional[str] = field(
        default=None, hash=False, compare=False, repr=False
    )
    raw_outputs: Optional[Sequence[str]] = field(
        default=None, hash=False, compare=False, repr=False
    )

    time: Optional[float] = field(default=None, hash=False, compare=False)
    norm: Optional[float] = field(default=None, hash=False, compare=False)
    gflops: Optional[float] = field(default=None, hash=False, compare=False)

    in_database: bool = field(default=False, hash=False, compare=False, repr=False)

    commit: str = field(default="HEAD", hash=False, compare=False, repr=False)

    def __has_result(self: BinaryOpenMP):
        return self.time is not None and self.norm and self.gflops is not None

    def clear_result(self: BinaryOpenMP):
        self.time = None
        self.norm = None
        self.gflops = None

    def __query(
        self: BinaryOpenMP, session: sqlalchemy.orm.Session
    ) -> sqlalchemy.orm.Query[OpenMP]:
        query = (
            session.query(OpenMP)
            .filter(OpenMP.algo == self.algo)
            .filter(OpenMP.M == self.M)
            .filter(OpenMP.K == self.K)
            .filter(OpenMP.N == self.N)
            .filter(OpenMP.compiler == self.compiler)
            .filter(OpenMP.multiple_times == self.multiple_times)
        )
        if self.algo == Algo.tiled:
            query = query.filter(OpenMP.block == self.block)
        if self.algo != Algo.blas:
            query = query.filter(OpenMP.omp == self.omp)
            if self.omp:
                query = (
                    query.filter(OpenMP.num_threads == self.num_threads)
                    .filter(OpenMP.schedule == self.schedule)
                    .filter(OpenMP.chunk == self.chunk)
                )
        return query

    def __post_init__(self: BinaryOpenMP):
        if self.algo == Algo.blas:
            self.omp = None
        if not self.omp:
            self.num_threads = None
            self.schedule = None
            self.chunk = None
        if self.algo != Algo.tiled:
            self.block = None
        if not self.Session:
            return

        session: sqlalchemy.orm.Session
        with self.Session() as session:
            query = self.__query(session)
            res = query.one_or_none()
            if res:
                self.in_database = True
                self.time = float(res.time)
                self.norm = float(res.norm)
                self.gflops = float(res.gflops)
                self.hash = res.hash

    @classmethod
    def compile_and_run(
        cls,
        Session: Optional[sessionmaker] = None,
        algo: Algo = Algo.naive,
        M: int = 4,
        K: int = 8,
        N: int = 4,
        block: Optional[int] = 4,
        omp: Optional[bool] = False,
        num_threads: Optional[int] = 4,
        schedule: Optional[Schedule] = Schedule.static,
        chunk: Optional[int] = 0,
        print_array: bool = False,
        force_recompile: bool = False,
        custom_path: Optional[str] = None,
        save_source: Optional[str] = None,
        save_output: Optional[str] = None,
        compiler: Optional[Compiler] = None,
        check_divisible: bool = True,
        debug: bool = False,
        commit: str = "HEAD",
    ):
        compiler = (
            compiler or Compiler.clang if sys.platform == "darwin" else Compiler.gcc
        )
        bin = cls(
            Session,
            algo,
            M,
            K,
            N,
            block,
            omp,
            num_threads,
            schedule,
            chunk,
            compiler,
            1,
            commit=commit,
        )
        command = ""
        if save_output and os.path.exists(save_output):
            with open(save_output, "r") as f:
                outputs = f.read().splitlines()
                bin.raw_outputs = outputs[:-1]
                command = outputs[-1]
        else:
            command = bin.__compile(
                print_array,
                force_recompile,
                custom_path,
                save_source,
                check_divisible,
                debug,
            )
            bin.__run_raw()
            assert bin.raw_outputs
            if save_output:
                with open(save_output, "w") as f:
                    f.write("\n".join(bin.raw_outputs) + "\n" + command)
        bin.__parse()
        return bin, command

    def __compile(
        self: BinaryOpenMP,
        print_array: bool = False,
        force_recompile: bool = False,
        custom_path: Optional[str] = None,
        save_source: Optional[str] = None,
        check_divisible: bool = True,
        debug: bool = False,
    ) -> str:
        if not self.hash:
            identity = f"{self.algo.name}.{self.M}.{self.N}.{self.K}"
            if self.algo == Algo.tiled:
                identity += f".{self.block}"
            if self.algo != Algo.blas:
                identity += f".{self.omp}"
            if self.omp:
                identity += f".{self.num_threads}.{self.schedule}.{self.chunk}"
            if self.commit != "HEAD":
                identity += f".{self.commit}"
            identity += f".{self.compiler}"
            self.hash = sha256(identity.encode()).hexdigest()[:10]

        executable = f"blas3_{self.hash}"
        exec_path = custom_path or os.path.join(self.executable_prefix, executable)
        command = ""

        if (
            not os.path.exists(exec_path) and not self.__has_result()
        ) or force_recompile:
            if force_recompile:
                self.time = None
                self.norm = None
                self.gflops = None
            if self.commit != "HEAD":
                subprocess.run(
                    ["git", "checkout", self.commit, self.source_path], check=True
                )
            args = [
                self.compiler.name,
                "-o",
                exec_path,
                "-x",
                "c",
                ("-" if not save_source else save_source),
                "-lblas",
                "-fopenmp",
                "-lm",
            ]

            if not debug:
                args.append("-O3")
            else:
                args += ["-g3", "-O0"]

            codes: str = ""
            with open(self.source_path) as f:
                codes = f.read()

            args.append(f"-D M_HPC={self.M}")
            args.append(f"-D K_HPC={self.K}")
            args.append(f"-D N_HPC={self.N}")
            if self.algo == Algo.tiled and self.block:
                args.append(f"-D BLOCK_HPC={self.block}")
                if check_divisible and (
                    self.M % self.block or self.K % self.block or self.N % self.block
                ):
                    args.append("-D NOT_DIVISIBLE_BY_BLOCK")

            if not self.omp:
                args.append("-D NO_OMP")
                codes = codes.replace("num_threads(NUM_THREADS_HPC)", "")
                codes = codes.replace("schedule(SCHEDULE_HPC)", "")
            else:
                if self.num_threads:
                    codes = codes.replace("NUM_THREADS_HPC", str(self.num_threads))
                schedule = None
                if self.schedule:
                    schedule = self.schedule.name
                    if self.chunk:
                        schedule += f", {self.chunk}"
                if schedule:
                    codes = codes.replace("SCHEDULE_HPC", schedule)

            if not print_array:
                args.append("-D NO_PRINT_ARRAY")

            for algo in Algo:
                if algo != self.algo:
                    args.append(f"-D NO_{algo.name.upper()}_DOT")

            input = None
            if save_source:
                with open(save_source, "w") as f:
                    f.write(codes)
            else:
                input = codes.encode()
            command = " ".join(args)
            subprocess.run([command], input=input, shell=True, check=True)
            if self.commit != "HEAD":
                subprocess.run(
                    ["git", "checkout", "HEAD", self.source_path], check=True
                )

            assert os.path.exists(exec_path)
        self.exec_path = exec_path
        return command

    def __run_raw(self: BinaryOpenMP):
        assert self.exec_path
        if self.__has_result():
            return
        self.raw_outputs = (
            subprocess.run(self.exec_path, capture_output=True, check=True)
            .stdout.decode("utf-8")
            .splitlines()
        )
        assert self.raw_outputs

    def __parse(self: BinaryOpenMP):
        if self.__has_result():
            return
        assert self.raw_outputs
        outputs = self.raw_outputs
        i = 0
        while i < len(outputs):
            output = outputs[i].strip()
            if output.startswith("Total time "):
                lhs, rhs = output.split(" = ")
                name = lhs[len("Total time ") :].strip().lower()
                assert name == self.algo.name
                time = float(rhs)
                _, rhs = outputs[i - 1].strip().split(" = ")
                norm = float(rhs)
                _, rhs = outputs[i + 1].strip().split(" = ")
                gflops = float(rhs)
                self.time = time
                self.norm = norm
                self.gflops = gflops
                i = i + 1
            i = i + 1

    def __to_sql(self: BinaryOpenMP) -> Optional[OpenMP]:
        assert self.__has_result()
        assert self.hash
        if not self.in_database:
            return OpenMP(
                algo=self.algo,
                time=cast(Decimal, self.time),
                norm=cast(Decimal, self.norm),
                gflops=cast(Decimal, self.gflops),
                M=self.M,
                K=self.K,
                N=self.N,
                block=self.block,
                omp=self.omp,
                num_threads=self.num_threads,
                schedule=self.schedule,
                chunk=self.chunk,
                compiler=self.compiler,
                multiple_times=self.multiple_times,
                hash=self.hash,
            )
        else:
            return None

    def run(
        self: BinaryOpenMP,
        print_array: bool = False,
        force_recompile: bool = False,
    ):
        try:
            if self.__has_result():
                return
            times = np.empty(self.multiple_times)
            norms = np.empty_like(times)
            gflopss = np.empty_like(times)
            self.__compile(print_array, force_recompile)
            for i in range(self.multiple_times):
                print(i)
                self.__run_raw()
                self.__parse()
                assert self.__has_result()
                times[i] = self.time
                norms[i] = self.norm
                gflopss[i] = self.gflops
                self.clear_result()
            assert np.all(norms == norms[0])
            self.time = np.average(times).astype(float)
            self.norm = norms[0]
            self.gflops = np.average(gflopss).astype(float)
        except BaseException as exec:
            print("Current binary options: ", self)
            raise exec

    def insert(
        self: BinaryOpenMP,
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
                        {"time": self.time, "norm": self.norm, "gflops": self.gflops},
                        synchronize_session="fetch",
                    )
                session.commit()


KT = tuple[
    Algo,
    int,
    int,
    int,
    Optional[int],
    Optional[bool],
    Optional[int],
    Optional[Schedule],
    Optional[int],
    Compiler,
]


@dataclass
class DBOpenMP:
    algos: Sequence[Algo] = field(
        default_factory=lambda: [Algo.naive, Algo.saxpy, Algo.tiled, Algo.blas]
    )

    Ms: Sequence[int] = field(default_factory=lambda: [4])
    Ks: Sequence[int] = field(default_factory=lambda: [8])
    Ns: Sequence[int] = field(default_factory=lambda: [4])

    blocks: Sequence[Optional[int]] = field(default_factory=lambda: [4])

    omps: Sequence[Optional[bool]] = field(default_factory=lambda: [False])

    num_threadss: Sequence[Optional[int]] = field(default_factory=lambda: [4])
    schedules: Sequence[Optional[Schedule]] = field(
        default_factory=lambda: [Schedule.static]
    )
    chunks: Sequence[Optional[int]] = field(default_factory=lambda: [0])
    compilers: Sequence[Compiler] = field(
        default_factory=lambda: [
            Compiler.clang if sys.platform == "darwin" else Compiler.gcc
        ]
    )
    multiple_times: int = 100

    source_path: str = "../openmp/blas3.c"
    executable_prefix: str = "bin/openmp/"
    print_array: bool = False
    force_recompile: bool = False
    upsert: bool = False

    outputs: set[KT] = field(default_factory=set)

    Binaries: ClassVar[
        dict[
            KT,
            BinaryOpenMP,
        ]
    ] = {}
    Outputs: ClassVar[
        dict[
            KT,
            tuple[float, float, float],
        ]
    ] = {}

    Engine: ClassVar[sqlalchemy.engine.Engine] = create_engine_and_table(Base)
    Session: ClassVar[sessionmaker] = sessionmaker(bind=Engine)
    Colnames: ClassVar[Sequence[str]] = [
        "algo",
        "time",
        "norm",
        "gflops",
        "M",
        "K",
        "N",
        "block",
        "omp",
        "num_threads",
        "schedule",
        "chunk",
        "compiler",
    ]
    AlgoDtype: ClassVar[pd.CategoricalDtype] = pd.CategoricalDtype(
        [algo.name for algo in Algo], ordered=True
    )
    ScheduleDType: ClassVar[pd.CategoricalDtype] = pd.CategoricalDtype(
        [schedule.name for schedule in Schedule], ordered=True
    )
    CompilerDType: ClassVar[pd.CategoricalDtype] = pd.CategoricalDtype(
        [compiler.name for compiler in Compiler], ordered=True
    )
    DFDtype: ClassVar[dict[str, type]] = {
        "algo": AlgoDtype,
        "time": float,
        "norm": float,
        "gflops": float,
        "M": int,
        "K": int,
        "N": int,
        "block": pd.Int64Dtype(),
        "omp": bool,
        "num_threads": pd.Int64Dtype(),
        "schedule": ScheduleDType,
        "chunk": pd.Int64Dtype(),
        "compiler": CompilerDType,
    }

    df: pd.DataFrame = field(
        default_factory=lambda Colnames=Colnames: pd.DataFrame(columns=Colnames)  # type: ignore
    )

    def __cast_df(self: DBOpenMP, df: Optional[pd.DataFrame] = None):
        df = df if df is not None else self.df
        for k, v in self.DFDtype.items():
            df[k] = df[k].astype(v)
        return df

    def __post_init__(self: DBOpenMP):
        self.df = self.__cast_df()
        for (
            algo,
            (M, K, N),
            block,
            omp,
            (num_threads, schedule, chunk),
            compiler,
        ) in itertools.product(
            self.algos,
            zip(self.Ms, self.Ks, self.Ns),
            self.blocks,
            self.omps,
            zip(self.num_threadss, self.schedules, self.chunks),
            self.compilers,
        ):
            if algo != Algo.tiled:
                block = None
            if algo == Algo.blas:
                omp = None
            if not omp:
                num_threads = None
                schedule = None
                chunk = None
            key = (algo, M, K, N, block, omp, num_threads, schedule, chunk, compiler)
            print(key)
            if key not in self.Outputs:
                binary = BinaryOpenMP(
                    Session=self.Session,
                    algo=algo,
                    M=M,
                    K=K,
                    N=N,
                    block=block,
                    omp=omp,
                    num_threads=num_threads,
                    schedule=schedule,
                    chunk=chunk,
                    compiler=compiler,
                    multiple_times=self.multiple_times,
                )
                binary.run(self.print_array, self.force_recompile)
                binary.insert(self.upsert)
                assert (
                    binary.time is not None
                    and binary.norm is not None
                    and binary.gflops is not None
                )
                self.Outputs[key] = (binary.time, binary.norm, binary.gflops)
                self.Binaries[key] = binary

            time, norm, gflops = self.Outputs[key]
            if key not in self.outputs:
                new_df = pd.DataFrame(
                    {
                        "algo": algo.name,
                        "time": time,
                        "norm": norm,
                        "gflops": gflops,
                        "M": M,
                        "K": K,
                        "N": N,
                        "block": block,
                        "omp": omp,
                        "num_threads": num_threads,
                        "schedule": schedule.name if schedule else pd.NA,
                        "chunk": chunk,
                        "compiler": compiler.name,
                    },
                    index=[0],
                )
                new_df = self.__cast_df(new_df)
                self.df = self.df.append(new_df, ignore_index=True)
                self.outputs.add(key)

    def to_df(
        self: DBOpenMP, colnames: Optional[Sequence[str]] = None, table: bool = True
    ):
        colnames = colnames or self.Colnames
        df = self.df[colnames].reset_index(drop=True)
        if table:
            df = df.replace(False, "").replace(True, "x")
            if "schedule" in colnames:
                df["schedule"] = df["schedule"].str.lower().fillna("")
        for col in ["block", "num_threads", "chunk"]:
            if col in colnames:
                df[col] = (
                    df[col].astype(object).apply(lambda x: int(x) if pd.notna(x) else x)
                )
        return df
