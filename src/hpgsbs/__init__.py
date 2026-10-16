"""HP-GSBS; optional GPU modules are imported only on demand."""
__version__ = "0.2.0"
from .config import GSBSConfig
from .api import fit
from .result import GSBSResult
from .batch.scheduler import fit_many
