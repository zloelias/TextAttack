from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import datetime
import os

from textattack.commands import TextAttackCommand


class TrainModelCommand(TextAttackCommand):
    """
    The TextAttack train module:
    
        A command line parser to train a model from user specifications.
    """

    def run(self, args):

        date_now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        outputs_dir = os.path.join(
            current_dir, os.pardir, os.pardir, os.pardir, "outputs", "training"
        )
        outputs_dir = os.path.normpath(outputs_dir)

        args.output_dir = os.path.join(
            outputs_dir, f"{args.model}-{args.dataset}-{date_now}/"
        )

        from .run_training import train_model

        train_model(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        from transformers import HfArgumentParser, TrainingArguments
        hf_parser = HfArgumentParser((TrainingArguments,))
        parser = main_parser.add_parser(
            "train",
            help="train a model for sequence classification",
            formatter_class=ArgumentDefaultsHelpFormatter,
            parents=[hf_train_parser]
        )
        parser.add_argument(
            "--model", type=str, required=True, help="directory of model to train",
        )
        parser.add_argument(
            "--dataset",
            type=str,
            required=True,
            default="yelp",
            help="dataset for training; will be loaded from "
            "`nlp` library. if dataset has a subset, separate with a colon. "
            " ex: `glue:sst2` or `rotten_tomatoes`",
        )
        parser.add_argument(
            "--dataset-split",
            type=str,
            default="",
            help="dataset split, if non-standard "
            "(can automatically detect 'dev', 'validation', 'eval')",
        )
        parser.add_argument(
            "--allowed-labels",
            type=int,
            nargs="*",
            default=[],
            help="Labels allowed for training (examples with other labels will be discarded)",
        )

        parser.set_defaults(func=TrainModelCommand())
