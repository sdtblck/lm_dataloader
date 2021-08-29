import os
import sys
from pathlib import Path

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from lm_dataloader.indexed_dataset import merge_datasets
from lm_dataloader import LMDataset, encode, MMapIndexedDataset
from lm_dataloader.encode import tokenize_char_level
import requests
import shutil

DUMMY_URL = "http://eaidata.bmk.sh/data/dummy.lmd"
DUMMY_BIN_URL = "http://eaidata.bmk.sh/data/dummy.lmd/dset.bin"
DUMMY_IDX_URL = "http://eaidata.bmk.sh/data/dummy.lmd/dset.idx"
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "test_data"))
CACHE_DIR = DATA_DIR / ".cache"


def url_exists(url):
    """Test if a URL exists"""
    r = requests.head(url)
    return r.ok


def test_encode(input_paths=None):
    """Test encoding a JSONL file to .lmd format"""
    if input_paths is None:
        input_paths = list(
            Path(os.path.join(os.path.dirname(__file__), "test_data")).glob("*.jsonl")
        )
    output_prefix = Path(os.path.join(os.path.dirname(__file__), "test_data")) / "test"
    output_path = str(output_prefix) + ".lmd"
    if os.path.isfile(output_path):
        # delete the output file
        os.remove(output_path)

    dataset = encode(
        input_paths,
        tokenize_fn=tokenize_char_level,
        tokenizer_vocab_size=256,
        output_prefix=output_prefix,
        eod_token=0,
        log_interval=100,
    )

    assert len(dataset) == 18


def test_merge_datasets():
    """Test merging two datasets"""
    input_path_1 = DATA_DIR / "dummy.jsonl"
    output_prefix_1 = DATA_DIR / "test_1"
    input_path_2 = DATA_DIR / "dummy_2.jsonl"
    output_prefix_2 = DATA_DIR / "test_2"
    dataset_1 = encode(
        [input_path_1],
        tokenize_fn=tokenize_char_level,
        tokenizer_vocab_size=256,
        output_prefix=output_prefix_1,
        eod_token=0,
    )

    dataset_2 = encode(
        [input_path_2],
        tokenize_fn=tokenize_char_level,
        tokenizer_vocab_size=256,
        output_prefix=output_prefix_2,
        eod_token=0,
    )

    merged = merge_datasets(
        [dataset_1._path, dataset_2._path], str(DATA_DIR / "test_merged")
    )
    assert len(merged) == len(dataset_1) + len(dataset_2)


def test_lmd_from_url():
    """Test loading a dataset from a URL"""

    assert url_exists(DUMMY_URL), "Dummy URL does not exist"

    # reset cache dir
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    dataset = LMDataset(DUMMY_URL, seq_length=1024, cache_dir=CACHE_DIR)
    assert len(dataset) > 0


def test_mmap_from_url():
    """Test loading a mmap indexed dataset from a URL"""
    assert url_exists(DUMMY_URL), "Dummy URL does not exist"

    # reset cache dir
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    dataset = MMapIndexedDataset.from_url(DUMMY_URL, cache_dir=CACHE_DIR)
    assert len(dataset) > 0


def test_mmap_from_urls():
    """Test loading a mmap indexed dataset from multiple URLs"""
    assert url_exists(DUMMY_BIN_URL), "Dummy .bin URL does not exist"
    assert url_exists(DUMMY_IDX_URL), "Dummy .idx URL does not exist"

    # reset cache dir
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    dataset = MMapIndexedDataset.from_urls(
        DUMMY_BIN_URL, DUMMY_IDX_URL, cache_dir=CACHE_DIR
    )

    assert len(dataset) > 0


def test_inspect_dataset():
    """Test inspecting a dataset"""
    input_path = DATA_DIR / "dummy.jsonl"
    output_prefix = DATA_DIR / "test"
    output_path = str(output_prefix) + ".lmd"

    _ = encode(
        [input_path],
        tokenize_fn=tokenize_char_level,
        tokenizer_vocab_size=256,
        output_prefix=output_prefix,
        eod_token=0,
    )

    from lm_dataloader.cmd_line import main
    import sys

    sys.argv.append(output_path)
    sys.argv = ["inspect", output_path]
    main()


# test_encode()
test_merge_datasets()
# test_lmd_from_url()
# test_mmap_from_urls()
# test_mmap_from_url()
# test_inspect_dataset()