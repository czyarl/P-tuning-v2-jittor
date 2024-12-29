import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
# from tasks.superglue.dataset import SuperGlueDataset
from training.trainer_base_2 import BaseTrainer

from datasets.load import load_metric

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from tasks.alpaca import utils
from torch.utils.data import Dataset
from transformers import Trainer

from model.glm2b.configuration_glm import GLMConfig
from model.glm2b.tokenization_glm import GLMTokenizer

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    gen_tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    gen_num_new_tokens = gen_tokenizer.add_special_tokens(special_tokens_dict)
    assert num_new_tokens == gen_num_new_tokens
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def remove_eos_token(input_ids_list: list, eos_token_id: int) -> list:
    """
    Removes the eos_token_id and any subsequent tokens from each sequence in the list of input_ids.

    Args:
        input_ids_list (list): A list of 1D tensors containing token IDs.
        eos_token_id (int): The ID of the EOS token to remove along with all subsequent tokens.

    Returns:
        list: A list where each tensor in the list is truncated at the first occurrence of eos_token_id.
    """
    # Initialize a list to store truncated sequences
    truncated_input_ids_list = []

    # Iterate over the list of input_ids
    for input_ids in input_ids_list:
        eos_mask = (input_ids == eos_token_id)  # Create a mask to find EOS tokens
        eos_pos = eos_mask.nonzero(as_tuple=False)  # Find positions of EOS token

        if eos_pos.size(0) > 0:
            first_eos_pos = eos_pos[0, 0].item()  # Get the position of the first EOS token
            truncated_input_ids_list.append(input_ids[:first_eos_pos])  # Append truncated sequence
        else:
            truncated_input_ids_list.append(input_ids)  # If no EOS token, keep the original sequence

    return truncated_input_ids_list

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    used_for_training: bool, 
) -> Dict:
    """Preprocess the data by tokenizing."""
    if used_for_training: 
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        # print(input_ids[0])
    else: 
        examples = [s for s, t in zip(sources, targets)]
        answer = [t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized, answer_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources, answer)]
        input_ids = examples_tokenized["input_ids"]
        input_ids = remove_eos_token(input_ids, tokenizer.eos_token_id)
        labels = answer_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, used_for_training = True):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        # for i in sources[:5]:
        #     print(len(i))
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer, used_for_training)

        self.input_ids = data_dict["input_ids"]
        print(len(self.input_ids))
        self.labels = data_dict["labels"]
        # print(self.labels[:10])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    padding_side: str = "right"

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_lengths = [len(seq) for seq in input_ids]
        label_lengths = [len(seq) for seq in labels]

        max_input_length = max(input_lengths)
        max_label_length = max(label_lengths)

        if self.padding_side == "right":
            # print(f"input_lengths = {input_lengths}")
            # print(f"label_lengths = {label_lengths}")
            padded_input_ids = [
                torch.cat([seq, torch.full((max_input_length - len(seq),), self.tokenizer.pad_token_id)])
                for seq in input_ids
            ]
            padded_labels = [
                torch.cat([seq, torch.full((max_label_length - len(seq),), IGNORE_INDEX)])
                for seq in labels
            ]
        elif self.padding_side == "left":
            padded_input_ids = [
                torch.cat([torch.full((max_input_length - len(seq),), self.tokenizer.pad_token_id), seq])
                for seq in input_ids
            ]
            padded_labels = [
                torch.cat([torch.full((max_label_length - len(seq),), IGNORE_INDEX), seq])
                for seq in labels
            ]
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        input_ids = torch.stack(padded_input_ids)
        labels = torch.stack(padded_labels)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, gen_tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.train_data_path)
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.eval_data_path)
    gen_dataset = SupervisedDataset(tokenizer=gen_tokenizer, data_path=data_args.eval_data_path, used_for_training=False)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    gen_data_collator = DataCollatorForSupervisedDataset(tokenizer=gen_tokenizer, padding_side="left")
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, gen_dataset=gen_dataset, data_collator=data_collator, gen_data_collator=gen_data_collator)


logger = logging.getLogger(__name__)

import numpy as np

class RougeEvaluator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        """
        Initialize the RougeEvaluator with a tokenizer.
        :param tokenizer: A Hugging Face tokenizer to decode token indices to text.
        """
        self.tokenizer = tokenizer
        self.rouge_metric = load_metric("rouge")
    
    def compute_rouge(self, references, predictions) -> Dict[str, float]:
        result = self.rouge_metric.compute(predictions=predictions, references=references)

        return {
            "rouge1": result["rouge1"].mid.fmeasure,
            "rouge2": result["rouge2"].mid.fmeasure,
            "rougeL": result["rougeL"].mid.fmeasure,
        }
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Compute ROUGE scores for a batch of predictions and labels.
        :param eval_preds: A tuple containing two numpy arrays:
                            - predictions: (batch_size, seq_len, vocab)
                            - labels: (batch_size, seq_len)
        :return: A dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.
        """
        predictions, labels = eval_preds

        # Step 1: Convert predictions from logits to token indices
        # Take argmax across the second dimension (num_heads) to get the best token index
        # predictions = np.argmax(predictions, axis=2)  # shape: (batch_size, seq_len)

        # Step 2: Mask positions where labels are -100
        mask = labels != -100  # Create a mask where valid labels are non-negative (i.e., not -100)
        
        # Step 3: Apply the mask to predictions and labels to exclude the -100 positions
        predictions = np.where(mask, predictions, 0)  # Replace invalid positions with a dummy value (e.g., 0)
        labels = np.where(mask, labels, 0)  # Replace invalid positions with 0

        # Step 4: Decode token indices into strings for the entire batch
        decoded_preds = self.tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)

        # Step 5: Remove leading/trailing whitespaces for each decoded text
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        # print(decoded_labels)

        # Step 6: Compute ROUGE scores
        result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        # Step 7: Return ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        return {
            "rouge1": result["rouge1"].mid.fmeasure,
            "rouge2": result["rouge2"].mid.fmeasure,
            "rougeL": result["rougeL"].mid.fmeasure,
        }

def get_trainer(args):
    model_args, data_args, training_args, _ = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    if model_args.model_name_or_path == "model/glm2b":
        config = GLMConfig.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
    else: 
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
    config.is_decoder = True
    if config.model_type == "gpt2": 
        config.pad_token_id = config.eos_token_id
    # print(config.max_position_embeddings)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=data_args.max_seq_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    gen_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=data_args.max_seq_length,
        padding_side="left",
        use_fast=False,
        trust_remote_code=True
    )

    model = get_model(model_args, TaskType.CAUSAL_LM, config)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        if config.model_type == "gpt2": 
            special_tokens_dict["pad_token"] = tokenizer.eos_token
        else: 
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        
    print(special_tokens_dict)

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        gen_tokenizer=gen_tokenizer, 
        model=model,
    )

    print(gen_tokenizer.pad_token_id)
    print(gen_tokenizer.pad_token)
    data_module = make_supervised_data_module(tokenizer=tokenizer, gen_tokenizer=gen_tokenizer, data_args=data_args)
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    evaluater = RougeEvaluator(tokenizer)

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"] if training_args.do_train else None,
        eval_dataset=data_module["eval_dataset"] if training_args.do_eval else None,
        test_key="rougeL", 
        gen_metric=evaluater.compute_rouge, 
        gen_metric_key="rougeL", 
        gen_tokenizer=gen_tokenizer,
        gen_dataset=data_module["gen_dataset"] if training_args.do_eval else None, 
        gen_data_collator=data_module["gen_data_collator"], 
        compute_metrics=None,
        tokenizer=tokenizer,
        data_collator=data_module["data_collator"],
    )


    return trainer, None
