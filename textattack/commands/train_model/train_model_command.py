from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
import datetime
import json
import os
import pickle
import textattack
import torch
import yaml
import re

from textattack.commands import TextAttackCommand

from .run_training import train_model

class HuggingFaceArgs(Namespace):
    """ A special namespace that makes args compatible with the HuggingFace API. """
    device = textattack.shared.utils.device
    
    def __init__(self, args):
        MAGIC_METHOD = re.compile('_\S*')
        for key in dir(args):
            # Don't override magic methods.
            if re.match(MAGIC_METHOD, key): 
                continue
        
            setattr(self, key, getattr(args, key))
    
    @classmethod
    def to_json_string(self):
        # @TODO: support this
        return ''
    
    @classmethod
    def to_sanitized_dict(self):
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        # @TODO: support this
        return {}
    
class TrainModelCommand(TextAttackCommand):
    """
    The TextAttack train module:
    
        A command line parser to train a model from user specifications.
    """
    
    def _process_args(self, args):
        """ Process args to align with HuggingFace ``TrainingArguments``
        """
        args.n_gpu = torch.cuda.device_count()
        
        args.train_batch_size = args.per_gpu_train_batch_size or args.per_device_train_batch_size
        args.train_batch_size *= max(1, torch.cuda.device_count())
        
        args.eval_batch_size = args.per_gpu_eval_batch_size or args.per_device_eval_batch_size
        args.eval_batch_size *= max(1, torch.cuda.device_count())
        
        delattr(args, 'func')
        
        return HuggingFaceArgs(args)


    def run(self, args):
        args = self._process_args(args)

        train_model(args)

    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        # The HFArgumentParser and TrainingArguments 
        from transformers import HfArgumentParser, TrainingArguments
        hf_train_parser = HfArgumentParser((TrainingArguments,), add_help=False)
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
            "--dataset_split",
            type=str,
            default="",
            help="dataset split, if non-standard "
            "(can automatically detect 'dev', 'validation', 'eval')",
        )
        parser.add_argument(
            "--allowed_labels",
            type=int,
            nargs="*",
            default=[],
            help="Labels allowed for training (examples with other labels will be discarded)",
        )
        parser.add_argument(
            "--max_length",
            type=int,
            default=512,
            help="Maximum length of a sequence (anything beyond this will "
            "be truncated)",
        )

        parser.set_defaults(func=TrainModelCommand())
