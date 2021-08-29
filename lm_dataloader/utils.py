import torch
import os
from pathlib import Path
import requests
from typing import Tuple
import tqdm


def get_cache_dir(cache_dir=None):
    return cache_dir or os.getenv("LMD_CACHE_DIR", Path.home() / "lm_datasets")


def compile_helpers():
    """Compile helper functions at runtime.
    Make sure this is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)


def is_main():
    """
    returns True if the current process is the main process.
    If torch distributed is not initialized, returns True (since there is assumed to only be the one process)
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def download(args: Tuple[str, str]):
    """Downloads a file from the given url and saves it to the given path."""
    url, out_path = args
    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(out_path))
        with open(out_path, "wb") as f:
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=1024 * 8),
                desc=f"Downloading {url} to {out_path}",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        raise r.raise_for_status()