import torch
import os
from pathlib import Path
import requests
from typing import Tuple
import tqdm
import sys 
import subprocess 
import time
from .global_vars import S3_CLIENT


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

def human_size(bytes, units=[' bytes','KB','MB','GB','TB', 'PB', 'EB']):
    """ Returns a human readable string representation of bytes """
    return str(bytes) + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def validate_s3():
    import boto3 
    sts = boto3.client('sts')
    try:
        sts.get_caller_identity()
    except boto3.exceptions.ClientError as e:
        print("AWS credentials are not valid.")
        raise e

def s3_download(s3_filename, local_filename):
    validate_s3()
    assert s3_filename.lower().startswith('s3://'), "must be an s3 path"
    path = s3_filename.replace('s3://', '').replace('S3://', '')
    bucket = path.split('/')[0]
    object_key = '/'.join(path.split('/')[1:])
    _s3_download(local_filename, bucket, object_key)

def _s3_download(local_file_name, s3_bucket, s3_object_key):

    global S3_CLIENT
    if S3_CLIENT is None:
        import boto3
        S3_CLIENT = boto3.client('s3')

    meta_data = S3_CLIENT.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    t = human_size(total_length)
    start = time.time()

    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        pct = downloaded / total_length
        duration = time.time() - start
        eta = duration / pct - duration
        done = int(50 * downloaded / total_length)
        progress_str = f"\r[{'=' * done}{' ' * (50-done)}] {human_size(downloaded)}/{t} | {eta:.2f}s"
        if len(progress_str) < 80:
            progress_str += " " * (80 - len(progress_str))
        sys.stdout.write(progress_str)
        sys.stdout.flush()

    print(f'Downloading {s3_object_key} to {local_file_name}')
    with open(local_file_name, 'wb') as f:
        S3_CLIENT.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)

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