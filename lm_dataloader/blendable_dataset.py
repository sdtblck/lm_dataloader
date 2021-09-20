# coding=utf-8
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

"""Blendable dataset."""

import time

import numpy as np
import torch

try:
    from .utils import print_rank_0, is_main, get_rank
    from .lm_dataset import LMDataset
    from .global_vars import MPU
except ImportError:
    from lm_dataset import LMDataset
    from utils import print_rank_0, is_main, get_rank
    from global_vars import MPU

from typing import List, Tuple, Optional
import math


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


def get_normalized_weights_and_num_samples(
    weights: List[float], num_samples: int
) -> Tuple[List[float], List[int]]:
    """
    Distributes the total number of samples (`num_samples`) among the groups in `weights`
    i.e with weights = [0.3, 0.2, 0.5] and num_samples = 10,
    The outputs should be [0.3, 0.2, 0.5] and [3, 2, 5] respectively
    """
    # Normalize weights
    weight_sum = sum(weights)
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    # Add 0.5% (the 1.005 factor) so in case the blending dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    weighted_num_samples = []
    for weight in weights:
        weighted_num_samples.append(int(math.ceil(num_samples * weight * 1.005)))
    return weights, weighted_num_samples


class BlendableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[LMDataset],
        weights: Optional[np.ndarray] = None,
        weight_by_num_docs=False,
        weighted_sampler_alpha=0.3,
        mpu=None,
    ):
        self.datasets = datasets
        num_datasets = len(datasets)

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset)

        self.weight_by_num_docs = weight_by_num_docs
        self.weighted_sampler_alpha = weighted_sampler_alpha
        if weights is None:
            assert (
                self.weight_by_num_docs
            ), "Must provide weights if not weighting by number of docs"
        if self.weight_by_num_docs:
            num_docs = [len(dataset) for dataset in self.datasets]
            weights = weights_by_num_docs(num_docs, alpha=self.weighted_sampler_alpha)
            total_num_samples = sum([d.num_samples for d in self.datasets])
            weights, weighted_num_samples = get_normalized_weights_and_num_samples(
                weights, total_num_samples
            )
            for i in range(num_datasets):
                # rebuild the datasets with the new num samples
                og_ds = self.datasets[i]
                self.datasets[i] = og_ds.__class__(
                    data_prefix=og_ds.data_prefix,
                    seq_length=og_ds.seq_length,
                    num_samples=weighted_num_samples[i],
                    indexed_dataset=og_ds.indexed_dataset,
                    documents=og_ds.documents,
                    seed=og_ds.seed,
                    mpu=og_ds.mpu,
                    cache_dir=og_ds.cache_dir,
                    skip_warmup=og_ds.skip_warmup,
                    mode=og_ds.mode,
                    pad_token=og_ds.pad_token,
                )

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights
        self.weights = weights

        assert num_datasets == len(self.weights)

        # Build indices.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8)
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64)

        from .helpers import build_blending_indices

        build_blending_indices(
            self.dataset_index,
            self.dataset_sample_index,
            self.weights,
            num_datasets,
            self.size,
            is_main(),
        )

        print(
            f"> RANK {get_rank()} elapsed time for building blendable dataset indices: \n\t{time.time() - start_time:.2f} (sec)"
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        try:
            dataset_idx = self.dataset_index[idx]
            sample_idx = self.dataset_sample_index[idx]
            return self.datasets[dataset_idx][sample_idx]
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]