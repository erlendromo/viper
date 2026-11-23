"""MegaLoc on the scan (database) with the IHM photographs (queries).

    python main.py megaloc --db-imgs data/latest/imgs --db-poses data/latest/poses \\
        --queries data/IHM-dataset/mapped --json data/IHM-dataset/IHM_full_with_queries-fixed_v2.json \\
        --out runs/megaloc

    add --weights runs/train_head_s0/last.pth --out runs/megaloc_ft_s0 to score a fine-tuned model

Steps
    1. database     one descriptor per scan image      (cached: db_desc.npy)
    2. queries      one descriptor per IHM photograph
    3. evaluation   cosine similarity, metrics and result files (see viper.utils.evaluation)

The model is downloaded by torch.hub on first use. The database descriptors are reused when the command
is run again with the same --out, so an interrupted run resumes where it stopped.
"""
from viper.utils.cli import add_data_arguments, add_output_arguments
from viper.utils.retrieval import add_model_arguments, run_retrieval

from .model import INPUT_SIZE

HELP = "MegaLoc: pretrained place recognition model (8448-d), optionally fine-tuned"


def add_arguments(parser):
    add_data_arguments(parser)
    add_output_arguments(parser)
    add_model_arguments(parser, default_size=INPUT_SIZE)


def run(args):
    from .model import load_model

    run_retrieval(args, method="megaloc", load_model=load_model)
