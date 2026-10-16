from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class GSBSConfig:
    kmax: int
    statewise_detection: bool = True
    finetune: int = 1
    finetune_order: bool = True
    blocksize: int = 50
    dmin: int = 1
    dtype: str = "float64"
    q_strategy: str = "full"
    tile_size: int = 128
    n_jobs: int = 1
    blas_threads: int = 1
    numba_threads: int = 1
    memory_fraction: float = 0.55
    trace: bool = False
    mode: str = "latency"
    tie_tolerance: float = 1e-11
    candidate_backend: str = "streaming"
    candidate_chunk_size: int = 256
    td_backend: str = "prefix2d"

    def __post_init__(self):
        if self.kmax < 2 or self.dmin < 1 or self.blocksize < 0:
            raise ValueError("Require kmax >= 2, dmin >= 1, blocksize >= 0")
        if self.dtype not in ("float64", "float32"):
            raise ValueError("Only explicit float64/float32 supported; TF32 is disabled")
        if self.q_strategy not in ("full", "lazy", "streaming"):
            raise ValueError("q_strategy must be full, lazy or streaming")
        if self.candidate_backend not in ("streaming", "materialized") or self.candidate_chunk_size < 1:
            raise ValueError("Invalid candidate backend or chunk size")
        if self.td_backend not in ("prefix2d", "feature-moments"):
            raise ValueError("td_backend must be prefix2d or feature-moments; auto is not calibrated")
        if min(self.tile_size, self.n_jobs, self.blas_threads, self.numba_threads) < 1:
            raise ValueError("Thread counts and tile size must be positive")
        if not 0 < self.memory_fraction < 1 or self.mode not in ("latency", "throughput"):
            raise ValueError("Invalid memory_fraction or mode")

    def algorithm(self):
        return {k: asdict(self)[k] for k in ("kmax", "statewise_detection", "finetune", "finetune_order", "blocksize", "dmin")}
