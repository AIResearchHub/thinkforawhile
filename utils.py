

import torch
from transformers import BertTokenizerFast, BartTokenizerFast

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)


def tokenize(texts):
    """
    Args:
        texts (List[string]): A list of strings, each string represents a book etc.

    Returns:
        output (List[List[int]]): A list of list of ints, each list of ints represent a tokenized book
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    return [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in texts]


def partition(ids, max_len):
    """
    partition id in ids into blocks of max_len,
    remove last block to make sure every block is the same size
    """
    return [torch.tensor([id[i:i+max_len] for i in range(0, len(id), max_len)][:-1], dtype=torch.int32)
            for id in ids]


def filter_empty(data, min_len=1):
    """
    Filter out all the empty tensors so there's no index error
    """
    return [x for x in data if x.size(0) >= min_len]



