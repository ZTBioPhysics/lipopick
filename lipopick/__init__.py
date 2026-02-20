"""
lipopick — Multi-scale DoG particle picker for cryo-EM lipoprotein micrographs.

Quick start:
    from lipopick import PickerConfig, pick_micrograph, process_micrograph, process_batch
    from lipopick.io import read_micrograph

    cfg = PickerConfig(dmin=150, dmax=500, refine=True)
    image = read_micrograph("micrograph.mrc")
    picks = pick_micrograph(image, cfg)          # ndarray (N, 5)
    # or: process one file end-to-end
    result = process_micrograph("micrograph.mrc", outdir="outputs/", cfg=cfg)
"""

__version__ = "0.1.0"

from .config import PickerConfig
from .pipeline import pick_micrograph, process_micrograph, process_batch

__all__ = [
    "PickerConfig",
    "pick_micrograph",
    "process_micrograph",
    "process_batch",
    "__version__",
]

try:
    from .mpi import mpi_process_batch
    __all__.append("mpi_process_batch")
except ImportError:
    pass
