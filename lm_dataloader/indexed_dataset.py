# coding=utf-8
# Copyright (c) 2021, Sid Black
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate

import numpy as np
import torch

try:
    from utils import print_rank_0, get_cache_dir, download
except ImportError:
    from .utils import print_rank_0, get_cache_dir, download

from pathlib import Path
import multiprocessing
from typing import Union, List


def __best_fitting_dtype(vocab_size: int = None):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32


def make_builder(out_file: Union[Path, str], vocab_size=None):
    return MMapIndexedDatasetBuilder(out_file, dtype=__best_fitting_dtype(vocab_size))


def make_indexed_dataset(
    data_prefix: str, cache_dir: Union[Path, str] = None, skip_warmup: bool = False
):
    if isinstance(data_prefix, Path):
        data_prefix = str(data_prefix)
    if not MMapIndexedDataset.exists(data_prefix) and not data_prefix.startswith(
        "http://"
    ):
        error_msg = f"Dataset does not exist: {data_prefix}"
        error_msg += "\nPath should be a basename that both .idx and .bin can be appended to get full filenames."
        raise FileNotFoundError(error_msg)
    elif data_prefix.startswith("http://"):
        return MMapIndexedDataset.from_url(
            data_prefix, cache_dir=cache_dir, skip_warmup=skip_warmup
        )
    else:
        return MMapIndexedDataset(data_prefix, skip_warmup)


def dataset_exists(data_prefix):
    return MMapIndexedDataset.exists(data_prefix)


dtypes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16,
}


def _code(dtype):
    for k in dtypes.keys():
        if dtypes[k] == dtype:
            return k
    raise ValueError(dtype)


def index_file_path(prefix_path):
    return str(Path(prefix_path) / "dset.idx")


def data_file_path(prefix_path):
    return str(Path(prefix_path) / "dset.bin")


def create_doc_idx(sizes):
    doc_idx = [0]
    for i, s in enumerate(sizes):
        if s == 0:
            doc_idx.append(i + 1)
    return doc_idx


def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index(object):
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, "wb")

                    # Write Magic string so we can check the file format then opening it again.
                    self._file.write(cls._HDR_MAGIC)
                    # Write version number
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", 1))
                    # Little endian unsigned 8 Bit integer
                    self._file.write(struct.pack("<B", _code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)

                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(sizes)))
                    # Little endian unsigned 64 Bit integer
                    self._file.write(struct.pack("<Q", len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order="C"))

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert (
                    self._HDR_MAGIC == magic_test
                ), "Index file doesn't match expected format. "
                # Little endian unsigned 64 Bit integer
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                # Little endian unsigned 8 Bit integer
                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                self._doc_count = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._doc_count,
                offset=offset + self._sizes.nbytes + self._pointers.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    @classmethod
    def from_url(cls, url, cache_dir=None, skip_warmup=False):
        url = Path(url)
        dataset_name = url.stem
        bin_url = str(data_file_path(url)).replace("http:/", "http://")
        idx_url = str(index_file_path(url)).replace("http:/", "http://")
        return cls.from_urls(bin_url, idx_url, dataset_name, cache_dir, skip_warmup)

    @classmethod
    def from_urls(
        cls, bin_url, idx_url, dataset_name=None, cache_dir=None, skip_warmup=False
    ):

        cache_dir = get_cache_dir(cache_dir)
        dataset_name = dataset_name or Path(bin_url).parent.stem
        prefix_path = cache_dir / f"{dataset_name}.lmd"

        os.makedirs(prefix_path, exist_ok=True)
        if os.path.exists(prefix_path / "dset.bin") and os.path.exists(
            prefix_path / "dset.idx"
        ):
            print_rank_0(f"Using cached dataset at {prefix_path}")
            return cls(prefix_path, skip_warmup)
        else:
            bin_out_path = data_file_path(prefix_path)
            idx_out_path = index_file_path(prefix_path)
            with multiprocessing.Pool(processes=2) as p:
                p.map(download, [(bin_url, bin_out_path), (idx_url, idx_out_path)])

            return cls(prefix_path, skip_warmup)

    def _do_init(self, path, skip_warmup):
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # get pointer to location in .bin file from .idx file
            ptr, size = self._index[idx]
            # read data from .bin file
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
            )
            return np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(
                self._bin_buffer, dtype=self._index.dtype, count=total_size, offset=ptr
            )
            sents = np.split(np_array, offsets[:-1])
            return sents

    def get(self, idx, offset=0, length=None):
        """Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=length, offset=ptr
        )
        return np_array

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and os.path.exists(
            data_file_path(path)
        )


class MMapIndexedDatasetBuilder(object):
    def __init__(self, file_prefix, dtype=np.int64, open_mode="wb"):
        if not file_prefix.endswith(".lmd"):
            file_prefix += ".lmd"
        self.file_prefix = file_prefix
        os.makedirs(self.file_prefix, exist_ok=True)
        self.bin_file = data_file_path(self.file_prefix)
        self.idx_file = index_file_path(self.file_prefix)
        self._data_file = open(self.bin_file, open_mode)
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, other):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(other))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(other), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def __add__(self, other):
        self.merge_file_(other)
        return self

    def finalize(self, index_file=None):
        self._data_file.close()
        if index_file is None:
            index_file = self.idx_file
        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes, self._doc_idx)
        # return finalized dataset
        return MMapIndexedDataset(self.file_prefix, skip_warmup=True)


def merge_datasets(datasets: List[Union[Path, str]], out_file: Union[Path, str]):
    """
    Merges a list of datasets into a single dataset at out_file.
    """
    if isinstance(datasets, str):
        # assume it's a comma-separated list of paths
        datasets = [d.strip() for d in datasets.split(",")]

    dtype = MMapIndexedDataset.Index(index_file_path(datasets[0])).dtype
    builder = MMapIndexedDatasetBuilder(out_file, dtype=dtype)
    for dataset in datasets:
        builder += dataset
    return builder.finalize()


def inspect_dataset(path):
    import code

    dataset = MMapIndexedDataset(path)

    code.interact(
        banner="\nEntering interactive shell. You can access the dataset contents through the local variable 'dataset', and index into it with dataset[i] where i is an integer. \
                \nEach index is a (tokenized) single document.",
        local={"dataset": dataset, "torch": torch, "np": np},
    )
