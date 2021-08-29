# Installation:

```bash
pip install lm_dataloader
```

# Design Philosophy

- A library to unify lm dataloading at large scale
- Simple interface, any tokenizer can be integrated
- Minimal changes needed from small -> large scale (many multiple GPU nodes)

- follows fairseq / megatron's 'mmap' dataformat, but with improvements. Those being:
    - [ ] Datasets should easily be able to be combined (blendable dataset does this I guess)
    - [x] unified into a single 'file' (which is actually a directory containing a .bin / .idx file)
    - [ ] index files that are built on the fly are hidden files, no longer leaving less mess in the directory.
    - [ ] More straightforward interface, better documentation.
    - [ ] Inspectable with a command line tool
    - [ ] Can load from urls
    - [ ] Can load from S3 buckets
    - [ ] Can load from GCS buckets
    - [ ] Can tokenize *on the fly* instead of preprocessing

# Example usage

To tokenize a dataset contained in a .jsonl file (where the text to be tokenized can be accessed under the 'text' key):

```python
import lm_dataloader as lmdl
from transformers import GPT2TokenizerFast 

jsonl_path = "test.jsonl"
output = "my_dataset.lmd"
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

lmdl.encode(
    jsonl_path,
    tokenize_fn=tokenizer.encode,
    tokenizer_vocab_size=len(tokenizer),
    output_prefix=output,
    eod_token=tokenizer.eos_token_id,
)
```

This will create a dataset at "my_dataset.lmd" which can be loaded as an indexed torch dataset like so:

```python
from lm_dataloader import LMDataset

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
seq_length = tokenizer.model_max_length # or whatever the sequence length of your model is

dataset = LMDataset("my_dataset.lmd", seq_length=seq_length)

# peek at 0th index
print(dataset[0])
```

# Command line utilities

There are also command line utilities provided to inspect / merge datasets, e.g:

```bash
lm-dataloader inspect my_dataset.lmd
```

Launches an interactive terminal to inspect the data in my_dataset.lmd

And:

```bash
lm-dataloader merge my_dataset.lmd,my_dataset_2.lmd new_dataset.lmd
```

Merges the datasets at "my_dataset.lmd" and "my_dataset_2.lmd" into a new file at "new_dataset.lmd".