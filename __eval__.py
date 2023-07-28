

import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from eval import test_perplexity


def main(
    cache_dir="./cache/datasets",
    device="cuda:0"
):

    model = torch.load("saved/final").to(device)

    dataloader = DataLoader(
        TextDataset(
            name="scientific_papers",
            cache_dir=cache_dir,
            split="test",
            seq_len=512,
            block_len=5,
            device=device,
            sep_padding=False,
        ),
        batch_size=8,
    )

    ppl = test_perplexity(model, dataloader, device)
    print("No [SEP] Perplexity: ", ppl)

    dataloader = DataLoader(
        TextDataset(
            name="scientific_papers",
            cache_dir=cache_dir,
            split="test",
            seq_len=512,
            block_len=5,
            device=device,
            sep_padding=True,
            sep_padding_prob=0.7
        ),
        batch_size=8,
    )
    ppl = test_perplexity(model, dataloader, device)
    print("[SEP] Perplexity: ", ppl)


if __name__ == "__main__":
    main()

