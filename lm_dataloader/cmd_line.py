from .indexed_dataset import inspect_dataset, merge_datasets
import sys


def main():
    """Entry point for the inspect_dataset command"""
    sub_command = sys.argv[1]
    args = sys.argv[2:]
    if sub_command == "inspect":
        inspect_dataset(args[0])
    elif sub_command == "merge":
        merge_datasets(*args)


if __name__ == "__main__":
    main()