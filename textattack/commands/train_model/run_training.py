import logging
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
    start_time = time.time()
    make_directories(args.output_dir)

    # Get list of text and list of label (integers) from disk.
    (train_text, train_labels), (eval_text, eval_labels), num_labels = data_from_args(args)
    
    # Get model 
    model, tokenizer = model_from_args(args, num_labels)
    
    # Create dataset
    train_dataset = create_dataset(tokenizer, train_text, train_labels)
    eval_dataset  = create_dataset(tokenizer, eval_text,  eval_labels)
    
    load_time = time.time()
    logger.info(f"Loaded data and tokenized in {load_time-start_time}s")
    
    logger.info('Training model with HuggingFace Trainer.')
    
    # Turn logging level up so that we can see output of transformers training
    # scripts.
    logging.basicConfig(level=logging.INFO)
    
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
