import torch
import textattack
logger = textattack.shared.logger

def prepare_dataset_for_training(nlp_dataset):
    """ Changes an `nlp` dataset into the proper format for tokenization. """

    def prepare_example_dict(ex):
        """ Returns the values in order corresponding to the data.
        
            ex:
                'Some text input'
            or in the case of multi-sequence inputs:
                ('The premise', 'the hypothesis',)
            etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return values

    return zip(*((prepare_example_dict(x[0]), x[1]) for x in nlp_dataset))

def encode_batch(tokenizer, text_list):
    try:
        return tokenizer.encode_batch(text_list)
    except AttributeError:
        return [tokenizer.encode(text) for text in text_list]

def data_from_args(args):
    """ Returns a tuple of ``HuggingFaceNLPDataset`` for the train and test
        datasets for ``args.dataset``.
    """
    dataset_args = args.dataset.split(":")
    # TODO `HuggingFaceNLPDataset` -> `HuggingFaceDataset`
    try:
        train_dataset = textattack.datasets.HuggingFaceNLPDataset(
            *dataset_args, split="train"
        )
    except KeyError:
        raise KeyError(f"Error: no `train` split found in `{args.dataset}` dataset")
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    
    if args.dataset_split:
        eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
            *dataset_args, split=args.dataset_split
        )
    else:
        # try common dev split names
        try:
            eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                *dataset_args, split="dev"
            )
        except KeyError:
            try:
                eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                    *dataset_args, split="eval"
                )
            except KeyError:
                try:
                    eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                        *dataset_args, split="validation"
                    )
                except KeyError:
                    raise KeyError(
                        f"Could not find `dev` or `test` split in dataset {args.dataset}."
                    )
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset)
    
    # Filter labels
    if args.allowed_labels:
        logger.info(f"Filtering samples with labels outside of {args.allowed_labels}.")
        final_train_text, final_train_labels = [], []
        for text, label in zip(train_text, train_labels):
            if label in args.allowed_labels:
                final_train_text.append(text)
                final_train_labels.append(label)
        logger.info(f"Filtered {len(train_text)} train samples to {len(final_train_text)} points.")
        train_text, train_labels = final_train_text, final_train_labels
        final_eval_text, final_eval_labels = [], []
        for text, label in zip(eval_text, eval_labels):
            if label in args.allowed_labels:
                final_eval_text.append(text)
                final_eval_labels.append(label)
        logger.info(f"Filtered {len(eval_text)} dev samples to {len(final_eval_text)} points.")
        eval_text, eval_labels = final_eval_text, final_eval_labels
    
    label_id_len = len(train_labels)
    label_set = set(train_labels)
    num_labels = len(label_set)
    logger.info(f"Loaded dataset. Found: {num_labels} labels: ({sorted(label_set)})")

    train_examples_len = len(train_text)

    if len(train_labels) != train_examples_len:
        raise ValueError(
            f"Number of train examples ({train_examples_len}) does not match number of labels ({len(train_labels)})"
        )
    if len(eval_labels) != len(eval_text):
        raise ValueError(
            f"Number of test examples ({len(eval_text)}) does not match number of labels ({len(eval_labels)})"
        )
    
    return (train_text, train_labels), (eval_text, eval_labels), num_labels

def model_from_args(args, num_labels):
    if args.model == "lstm":
        textattack.shared.logger.info("Loading textattack model: LSTMForClassification")
        model = textattack.models.helpers.LSTMForClassification(
            max_seq_length=args.max_length, num_labels=num_labels
        )
    elif args.model == "cnn":
        textattack.shared.logger.info(
            "Loading textattack model: WordCNNForClassification"
        )
        model = textattack.models.helpers.WordCNNForClassification(
            max_seq_length=args.max_length, num_labels=num_labels
        )
    else:
        import transformers

        textattack.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {args.model}"
        )
        config = transformers.AutoConfig.from_pretrained(
            args.model,
            num_labels=num_labels,
            finetuning_task=args.dataset
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model,
            config=config,
        )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(
            args.model, 
            use_fast=False, 
            max_length=args.max_length,
        )
        setattr(model, "tokenizer", tokenizer)

    model = model.to(textattack.shared.utils.device)
    tokenizer = model.tokenizer

    return model, tokenizer

def create_dataset(tokenizer, text, labels):
    """ Takes a list of text and labels, tokenizes, and creates dataset.
    
        Returns ``torch.utils.Dataset``
    """
    # Create datasets

    logger.info(f"Tokenizing data (len: {len(text)})")
    encoded_text = encode_batch(tokenizer, text)
    data = [{**encoding, 'label_ids': [label]} for encoding, label in zip(encoded_text, labels)]
    # for dict_obj in data:
        # dict_obj['__dict__'] = dict_obj
    return data
    # return torch.utils.data.dataset.Dataset(data)