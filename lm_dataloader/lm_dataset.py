# coding=utf-8
# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT2 style dataset."""

import os
import time

import numpy as np
import torch

try:
    from .utils import print_rank_0, is_main, get_cache_dir
    from .indexed_dataset import make_indexed_dataset, MMapIndexedDataset
except ImportError:
    print("got an import error")
    from utils import print_rank_0, is_main, get_cache_dir
    from indexed_dataset import make_indexed_dataset, MMapIndexedDataset

from pathlib import Path
from typing import Optional, Union, List, Any
from tqdm import tqdm


class LMDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_prefix: Union[str, Path],
        seq_length: int,
        num_samples: Optional[int] = None,
        indexed_dataset: Optional[MMapIndexedDataset] = None,
        documents: Optional[np.ndarray] = None,
        seed: int = 0,
        build_index_mappings: bool = True,
        mpu=None,
        cache_dir: Optional[Union[str, Path]] = None,
        skip_warmup: bool = False,
        mode: str = "pack",
        pad_token: int = None,
    ):
        print(f"Beginning init of {self}")
        self.cache_dir = get_cache_dir(
            cache_dir
        )  # set cache_dir to cache_dir if it exists

        # If no indexed dataset is provided, build one. Otherwise, just use the one provided
        if indexed_dataset is None:
            self.indexed_dataset = make_indexed_dataset(
                data_prefix, cache_dir=self.cache_dir, skip_warmup=skip_warmup
            )
        else:
            self.indexed_dataset = indexed_dataset

        self.mpu = mpu  # optional mpu object from megatron
        self.data_prefix = (
            self.indexed_dataset._path
        )  # get actual prefix from indexed_dataset, data_prefix may be url or local path
        self.name = Path(self.data_prefix).stem
        self.seq_length = seq_length
        self.seed = seed
        self.skip_warmup = skip_warmup
        self.sizes = self.indexed_dataset.sizes
        self.mode = mode.lower()
        assert self.mode in [
            "pack",
            "pad",
        ], f"mode must be 'pack' or 'pad', got {self.mode}"
        self.pad_token = pad_token
        if self.mode == "pad":
            assert self.pad_token is not None, "pad_token must be set if mode == pad"

        # build document index
        if documents is None:
            total_num_of_documents = self.indexed_dataset.sizes.shape[0]
            # TODO: cleaner printing
            print_rank_0("    {}:".format(self.name))
            print_rank_0("     no. of documents:{}".format(total_num_of_documents))
            self.documents = np.arange(
                start=0, stop=total_num_of_documents, step=1, dtype=np.int32
            )
        else:
            self.documents = documents

        self.num_samples = num_samples or self.tokens_per_epoch() // self.seq_length

        if build_index_mappings:
            # Build index mappings.
            (
                self.doc_idx,
                self.sample_idx,
                self.shuffle_idx,
            ) = self._build_index_mappings()
            self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
            self.sample_idx_len = self.sample_idx.shape[0] - 1

    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx) -> np.ndarray:
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]
            if self.mode == "pad":
                # if we are in pad mode, we only ever sample one document
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f], offset=offset_f
                )

                # if the sample is shorter than the seq_length, pad it
                if len(sample) < self.seq_length + 1:
                    sample = np.pad(
                        sample,
                        (0, self.seq_length + 1 - len(sample)),
                        mode="constant",
                        constant_values=self.pad_token,
                    )
                else:
                    # otherwise, truncate it.
                    # TODO: we'll lose some data here. How to avoid? (self.sizes gives us the size of each document, could use that)
                    sample = sample[: self.seq_length + 1]
            elif doc_index_f == doc_index_l:
                # If we are within the same document, just extract the chunk.
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f],
                    offset=offset_f,
                    length=offset_l - offset_f + 1,
                )
            else:
                # Otherwise, get the rest of the initial document.
                sample_list = [
                    self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                ]
                # Loop over all in between documents and add the entire document.
                for i in range(doc_index_f + 1, doc_index_l):
                    sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                # And finally add the relevant portion of last document.
                sample_list.append(
                    self.indexed_dataset.get(
                        self.doc_idx[doc_index_l], length=offset_l + 1
                    )
                )
                sample = np.concatenate(sample_list)

            return np.array(sample, dtype=np.int64)
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]

    def tokens_per_epoch(self):
        """Total number of tokens (per epoch) in the dataset."""
        return np.sum(self.sizes[self.documents])

    def _build_index_mappings(self):
        """Build doc-idx, sample-idx, and shuffle-idx.
        doc-idx: is an array (ordered) of documents to be used in training.
        sample-idx: is the start document index and document offset for each
        training sample.
        shuffle-idx: maps the sample index into a random index into sample-idx.
        """
        # Number of tokens in each epoch and number of required epochs.
        tokens_per_epoch = self.tokens_per_epoch()
        num_epochs = _num_epochs(tokens_per_epoch, self.seq_length, self.num_samples)

        # rng state
        np_rng = np.random.RandomState(seed=self.seed)

        # Filename of the index mappings.
        _filename = str(Path(self.data_prefix) / ".dset")
        _filename += "_{}_indexmap".format(self.name)
        _filename += "_{}ns".format(self.num_samples)
        _filename += "_{}sl".format(self.seq_length)
        _filename += "_{}s".format(self.seed)
        doc_idx_filename = _filename + "_doc_idx.npy"
        sample_idx_filename = _filename + f"_{self.mode}_sample_idx.npy"
        shuffle_idx_filename = _filename + "_shuffle_idx.npy"

        # Build the indexed mapping if not exist.
        if is_main():
            if (
                (not os.path.isfile(doc_idx_filename))
                or (not os.path.isfile(sample_idx_filename))
                or (not os.path.isfile(shuffle_idx_filename))
            ):
                print_rank_0(
                    " > WARNING: could not find index map files, building "
                    "the indices on rank 0 ..."
                )
                # doc-idx.
                start_time = time.time()
                doc_idx = _build_doc_idx(self.documents, num_epochs, np_rng)
                np.save(doc_idx_filename, doc_idx, allow_pickle=True)
                print_rank_0(
                    " > elasped time to build and save doc-idx mapping "
                    "(seconds): {:4f}".format(time.time() - start_time)
                )
                # sample-idx.
                start_time = time.time()
                # Use C++ implementation for speed.

                assert doc_idx.dtype == np.int32
                assert self.sizes.dtype == np.int32
                if self.mode == "pad":
                    sample_idx = np.concatenate(
                        (
                            np.expand_dims(np.arange(len(doc_idx)), 1),
                            np.expand_dims(np.zeros(len(doc_idx)), 1),
                        ),
                        axis=1,
                    ).astype(
                        np.int32
                    )  # map each sample straight to each document. I.e no packing. Each sample is padded to the sequence length.
                else:
                    try:
                        from .helpers import build_sample_idx

                        sample_idx = build_sample_idx(
                            self.sizes,
                            doc_idx,
                            self.seq_length,
                            num_epochs,
                            tokens_per_epoch,
                        )
                    except ImportError:
                        error_msg = "Could not find C++ helpers module - please make sure to run lm_dataloader.compile_helpers()."
                        error_msg += "\n Falling back to the python implementation (Will be much slower)"
                        sample_idx = _build_sample_idx(
                            self.sizes,
                            doc_idx,
                            self.seq_length,
                            num_epochs,
                            tokens_per_epoch,
                        )

                np.save(sample_idx_filename, sample_idx, allow_pickle=True)
                print_rank_0(
                    " > elapsed time to build and save sample-idx mapping "
                    "(seconds): {:4f}".format(time.time() - start_time)
                )
                # shuffle-idx.
                start_time = time.time()
                # -1 is due to data structure used to retieve the index:
                #    sample i --> [sample_idx[i], sample_idx[i+1])
                shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
                np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
                print_rank_0(
                    " > elapsed time to build and save shuffle-idx mapping"
                    " (seconds): {:4f}".format(time.time() - start_time)
                )

        if torch.distributed.is_initialized() and self.mpu is not None:
            # This should be a barrier but nccl barrier assumes
            # device_index=rank which is not the case for model
            # parallel case
            counts = torch.cuda.LongTensor([1])
            torch.distributed.all_reduce(counts, group=self.mpu.get_io_parallel_group())
            assert counts[0].item() == torch.distributed.get_world_size(
                group=self.mpu.get_io_parallel_group()
            )

        # Load mappings.
        start_time = time.time()
        print_rank_0(f" > loading doc-idx mapping from {doc_idx_filename}")
        doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
        print_rank_0(f" > loading sample-idx mapping from {sample_idx_filename}")
        sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
        print_rank_0(f" > loading shuffle-idx mapping from {shuffle_idx_filename}")
        shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
        print_rank_0(
            f"    loaded indexed file in {time.time() - start_time:3.3f} seconds"
        )
        print_rank_0(f"    total number of samples: {sample_idx.shape[0]}")
        print_rank_0(f"    total number of epochs: {num_epochs}")

        return doc_idx, sample_idx, shuffle_idx


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    assert tokens_per_epoch > 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx


def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)
    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def from_splits(
    indexed_dataset: MMapIndexedDataset,
    splits: Union[List[float], str],
    num_samples: List[int],
    seq_length: int,
    seed: int = 0,
    mpu: Optional[Any] = None,
) -> List[LMDataset]:
    """
    Build an `LMDataset` from an indexed dataset and a list of splits.

    :param indexed_dataset:
        The indexed dataset to build splits from.
    :param splits:
        A list of floats or a string. If a string, it should be a comma-separated
        list of floats between 0.0 and 1.0.
    :param num_samples:
        A list of integers representing the number of samples in each split.
    :param seq_length:
        The sequence length of the model
    :param seed:
        The random seed to use for shuffling.
    :param mpu:
        If using multiprocessing, provide an mpu.
    """
    if isinstance(splits, str):
        splits = [float(x) for x in splits.split(",")]

    # get rid of zeros and negative values
    assert len(splits) == len(num_samples)
    _splits, _num_samples = [], []
    for i in range(len(splits)):
        if splits[i] > 0.0:
            _splits.append(splits[i])
            _num_samples.append(num_samples[i])
    splits, num_samples = _splits, _num_samples

    # normalize
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]

    # upweight to size of the dataset
    total_docs = len(indexed_dataset)
    splits_num_docs = [int(round(s * total_docs)) for s in splits]
    assert sum(splits_num_docs) == total_docs

    # build the datasets
    datasets = []

    for idx, n in enumerate(
        tqdm(splits_num_docs, desc=f"Building datasets from splits: {splits_num_docs}")
    ):
        if idx == 0:
            # first split, start at 0
            start = 0
        else:
            # otherwise, the start is the end of the previous split
            start = splits_num_docs[idx - 1]
        end = start + n
        datasets.append(
            LMDataset(
                data_prefix="",  # not needed, as we're providing indexed_dataset,
                seq_length=seq_length,
                num_samples=num_samples[idx],
                indexed_dataset=indexed_dataset,
                documents=np.arange(start=start, stop=end, dtype=np.int32),
                seed=seed,
                mpu=mpu,
            )
        )

    return datasets
