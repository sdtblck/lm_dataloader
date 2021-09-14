from .lm_dataset import LMDataset
from .encode import encode
from .indexed_dataset import (
    make_indexed_dataset,
    MMapIndexedDataset,
    dataset_exists,
    inspect_dataset,
    merge_datasets,
)
from .utils import compile_helpers
from .global_vars import set_mpu

try:
    from .helpers import *
except ModuleNotFoundError:
    try:
        compile_helpers()
        from .helpers import *
    except (ModuleNotFoundError, SystemError):
        print(
            "Not able to compile C++ helpers. Some functions may be slower, or not work at all (blendable_dataset)"
        )
