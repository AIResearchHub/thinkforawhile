

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast

import random

import logging
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)


def tokenize(texts, type="word"):
    """
    Args:
        texts (List[string]): A list of strings, each string represents a book etc.
        type (string=char): What tokenizer to use, character level (char) or word level (word)

    Returns:
        output (List[List[int]]): A list of list of ints, each list of ints represent a tokenized book
    """
    if type == "word":
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    if type == "char":
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


def join(strings):
    """
    Simple function to join list of token strings into readable and coherent original strings
    """
    output = ""
    for string in strings:
        if string[:2] == "##":
            output += string[2:]
        elif string == "i" or string == "a":
            output += ' ' + string
        elif len(string) == 1:
            output += string
        else:
            output += ' ' + string

    return output


def remove_padding(dataset, sep_token=102, start_token=50, end_token=51):
    """
    Args:
        dataset (tensor):
        sep_token (int): Id associated with [SEP]
        start_token (int): Id associated with [START]
        end_token (int): Id associated with [END]
    """



@torch.inference_mode()
def generate_samples(model, prompt_ids=[50, 102],
                     B=8, T=500, temperature=0.5, device="cuda"):
    """
    Note:
        [START] = 50
        [SEP] = 102
        [END] = 51


    Args:
        model (nn.Module): trained model to generate samples with
        prompt_ids (List[int]): A list of ids of the starting prompt
        B (int): batch size
        T (int): generate timesteps
        temperature (float): Controls the "creativity" of the text generated always between 0 and 1
                             higher (e.g. 0.7) results in more diverse and creative outputs
                             lower (e.g. 0.2) makes the output more deterministic and focused
        device (string): which device to run on, cpu or cuda

    Returns:
        x (tensor): Generated samples in tensor
    """

    x = torch.tensor(prompt_ids, dtype=torch.int64, device=device).repeat(B, 1)

    model.module.reset()
    for _ in range(T):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.concat((x, next_id), dim=-1)

    return x


@torch.inference_mode()
def generate_samples_with_sep(model, seq_len, w, prompt_ids=[50, 102],
                              B=8, T=500, temperature=0.5, device="cuda"):
    """
    Generate samples with [SEP] padding to allow model to think-for-a-while
    before deciding what to generate

    Note:
        [START] = 50
        [SEP] = 102
        [END] = 51

    Args:
        model (nn.Module): trained model to generate samples with
        prompt_ids (List[int]): A list of ids of the starting prompt
        B (int): batch size
        T (int): generate timesteps
        temperature (float): Controls the "creativity" of the text generated always between 0 and 1
                             higher (e.g. 0.7) results in more diverse and creative outputs
                             lower (e.g. 0.2) makes the output more deterministic and focused
        device (string): which device to run on, cpu or cuda

    Returns:
        x (tensor): Generated samples in tensor
    """

    x = torch.tensor(prompt_ids, dtype=torch.int64, device=device).repeat(B, 1)

    model.module.reset()
    for _ in range(T):
        logits = model(x)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.concat((x, next_id), dim=-1)

        if x.size(-1) % w == 0:
            length = random.randrange(1, (seq_len // w) - 1)
            pad = torch.full((w * length,))
            pad[0] = 50
            pad[-1] = 51

            pad = pad.unsqueeze(0).repeat(B, 1)
            x = torch.concat((x, pad), dim=-1)

    return x

