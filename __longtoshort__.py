

import torch
from transformers import BertTokenizerFast

from utils import generate_samples, generate_samples_with_sep, join
from eval import test_perplexity


def generate_target_dist(model, dataset):
    """
    Args:
        model (nn.Module): model to be loaded
        dataset (tensor): tensor of shape (B, T)
    """



    return


def train(
    model, dataset
):
    """
    Args:
        model (nn.Module): model to be loaded
        dataset (tensor): tensor of shape (B, T)
    """

    return


def main(
    device="cuda",
):

    model = torch.load("saved/final").to(device)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # returns tensor with shape (B, T)
    # samples = generate_samples(model)
    samples = generate_samples(model, temperature=1.0, B=64)
    generate_target_dist

    print(samples.shape)
    print(type(samples))
    print(samples)

    for sample in samples:
        print("SAMPLE:")
        print()
        sentence = tokenizer.batch_decode(sample)
        print(join(sentence))

    train(model)


if __name__ == "__main__":
    main()

