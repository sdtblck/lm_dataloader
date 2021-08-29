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

"""Processing data for pretraining."""

import multiprocessing

import lm_dataformat as lmd

import time
import tqdm
import torch
import ftfy

import lm_dataloader.indexed_dataset as indexed_dataset
from threading import Semaphore
from typing import List, Callable, Optional, Union
import numpy as np
from pathlib import Path


class Encoder(object):
    def __init__(
        self,
        tokenize_fn: Callable[[str], List[int]],
        json_key: str = "text",
        use_ftfy: bool = False,
        append_eod: bool = False,
        eod_token: Optional[int] = None,
    ):
        self.tokenize_fn = tokenize_fn
        self.use_ftfy = use_ftfy
        self.append_eod = append_eod
        if self.append_eod:
            assert eod_token is not None, "EOD token must be specified if appending"
        self.eod_token = eod_token
        self.json_key = json_key

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenize_fn = self.tokenize_fn

    def encode(self, text):
        if self.use_ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        doc_ids = []
        text_ids = Encoder.tokenize_fn(text)
        if len(text_ids) > 0:
            doc_ids.append(text_ids)
        if self.append_eod:
            doc_ids[-1].append(self.eod_token)
        ids[self.json_key] = doc_ids
        return ids, len(text)


def yield_from_files(fnames: list, semaphore: Semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.
    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        if isinstance(fname, Path):
            fname = str(fname)
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def tokenize_char_level(text: str, vocab_size: int = 256) -> List[str]:
    return list(np.clip(np.fromstring(text, dtype=np.uint8), 0, vocab_size))


def encode(
    paths: Union[List[str], str, Path, List[Path]],
    tokenize_fn: Callable[[str], List[int]],
    tokenizer_vocab_size: int,
    output_prefix: str,
    eod_token: int,
    log_interval: int = 100,
    workers=None,
    use_ftfy=False,
    append_eod=True,
    json_key="text",
    num_docs=None,
):
    """
    Encode a list of files into an .lmd format dataset for pretraining.

    :param paths:
        Either a list of paths to files, or a single path. (in str or pathlib.Path format)
    :param tokenize_fn:
        Function to tokenize a string into a list of ints
    :param tokenizer_vocab_size:
        The size of the vocabulary of the tokenizer
    :param output_prefix:
        The prefix of the output files. E.G an output prefix of "my_data" will create an .lmd file at "my_data.lmd"
    :param eod_token:
        The integer to use for the end of document token
    :param log_interval:
        How often to log progress (default: 100)
    :param workers:
        Number of workers to use for parallel processing (default: cpu count)
    :param use_ftfy:
        Whether to use ftfy to fix unicode text. (default: False)
    :param append_eod:
        Whether to append an end of document token to each document. (default: True)
    :param json_key:
        The key in the .jsonl input where the text is stored. (default: "text")
    :param num_docs:
        The number of documents in the input file. Only used with tqdm progress bar to report progress. (default: None)
    """
    if workers is None:
        workers = multiprocessing.cpu_count()

    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, Path):
        paths = [str(paths)]

    encoder = Encoder(
        tokenize_fn,
        json_key=json_key,
        use_ftfy=use_ftfy,
        append_eod=append_eod,
        eod_token=eod_token,
    )

    print(f"Vocab size: {tokenizer_vocab_size}")
    print(f"Output prefix: {output_prefix}")
    print()
    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(paths, semaphore)

    if workers > 1:
        pool = multiprocessing.Pool(workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(
            encoder.encode, fin, chunksize=25
        )  # TODO why 25? are different numbers faster?
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    # make a dataset builder
    builder = indexed_dataset.make_builder(
        output_prefix,
        vocab_size=tokenizer_vocab_size,
    )

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builder.add_item(torch.IntTensor(sentence))
            # separate with eos token
            builder.end_document()

        # log progress
        if i % log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if num_docs is None else '/' + str(num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(log_interval)

    # save output file + return finalized dataset
    return builder.finalize()
