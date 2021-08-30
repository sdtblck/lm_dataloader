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


def get_rank():
    """Returns the rank of the current process.
    If torch distributed is not initialized, returns 0 (since there is assumed to only be the one process)
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


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


def weights_by_num_docs(l, alpha=0.3):
    """
    Builds weights from a multinomial distribution over groups of data according to the number of
    samples in each group.
    We sample from a group according to the probability p(L) ∝ |L| ** α,
    where p(L) is the probability of sampling from a given group,
          |L| is the number of examples in that datapoint,
          and α is a coefficient that acts to upsample data from underrepresented groups
    Hence α (`alpha`) allows us to control how much to 'boost' the probability of training on low-resource groups.
    See https://arxiv.org/abs/1911.02116 for more details
    """
    total_n_docs = sum(l)
    unbiased_sample_probs = [i / total_n_docs for i in l]

    probs = [i ** alpha for i in unbiased_sample_probs]

    # normalize
    total = sum(probs)
    probs = [i / total for i in probs]

    # weights should be the inverse of the number of samples
    unbiased_sample_probs_inverse = [1 - p for p in unbiased_sample_probs]
    weights = [p * p2 for p, p2 in zip(probs, unbiased_sample_probs_inverse)]

    # normalize
    total = sum(weights)
    weights = [i / total for i in weights]

    return weights