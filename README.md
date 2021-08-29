# Installation:

```bash
pip install lm_dataloader
```

# Design Philosophy

- A library to unify lm dataloading at large scale
- Simple interface, any tokenizer can be integrated
- Minimal changes needed from small -> large scale (many multiple GPU nodes)

- follows fairseq / megatron's 'mmap' dataformat, but with improvements. Those being:
    - [x] Easily combine multiple datasets
    - [ ] Easily split a dataset into train / val / test splits
    - [x] Easily build a weighted dataset out of a list of existing ones along with weights.
    - [x] unified into a single 'file' (which is actually a directory containing a .bin / .idx file)
    - [x] index files that are built on the fly are hidden files, leaving less mess in the directory.
    - [ ] More straightforward interface, better documentation.
    - [x] Inspectable with a command line tool
    - [x] Can load from urls
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
    output_prefix=output,
    tokenize_fn=tokenizer.encode,
    tokenizer_vocab_size=len(tokenizer),
    eod_token=tokenizer.eos_token_id,
)
```

This will create a dataset at "my_dataset.lmd" which can be loaded as an indexed torch dataset like so:

```python
from lm_dataloader import LMDataset
from transformers import GPT2TokenizerFast 

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