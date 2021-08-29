from .indexed_dataset import inspect_dataset, merge_datasets
import sys


def main():
    """Entry point for the inspect_dataset command"""
    sub_command = sys.argv[1]
    args = sys.argv[2:]
    if sub_command == "inspect":
        inspect_dataset(sys.argv[1])
    elif sub_command == "merge":
        merge_datasets(sys.argv[1:])


if __name__ == "__main__":
    main()