import os
import time

import textattack
import transformers

from .train_args_helpers import data_from_args, model_from_args, create_dataset

logger = textattack.shared.logger


def make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def train_model(args):
    args.device = textattack.shared.utils.device
    args.local_rank = -1 # Don't support distributed training (yet)
    args.logging_dir = args.output_dir # Write logs to same directory model gets saved to (TODO is this the best choice?)

    start_time = time.time()
    make_directories(args.output_dir)

    # Start Tensorboard and log hyperparams.
    # from tensorboardX import SummaryWriter

    # tb_writer = SummaryWriter(args.output_dir)
    # args_dict = vars(args)
    # del args_dict["func"]
    # tb_writer.add_hparams(args_dict, {})

    # Use Weights & Biases, if enabled.
    # if args.enable_wandb:
        # wandb.init(sync_tensorboard=True)

    # Get list of text and list of label (integers) from disk.
    (train_text, train_labels), (eval_text, eval_labels), num_labels = data_from_args(args)
    
    # Get model 
    model, tokenizer = model_from_args(args, num_labels)
    
    # Create dataset
    train_dataset = create_dataset(tokenizer, train_text, train_labels)
    eval_dataset  = create_dataset(tokenizer, eval_text,  eval_labels)
    
    load_time = time.time()
    logger.info(f"Loaded data and tokenized in {load_time-start_time}s")
    
    logger.info('Training data with HuggingFace Trainer.')
    
    #  create HF Trainer
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    trainer.train(model_path=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
