

from torch.utils.data import DataLoader

from transformer import *
from trainer import AutoregressiveTrainer
from dataset import TextDataset


"""
Supported datasets:

pg19
scientific_papers

"""


def main(
    dataset="scientific_papers",
    seq_len=512,
    vocab_size=30522,
    n_layers=4,
    d_model=768,
    n_head=8,
    p=0.1,
    lr=6e-5,
    batch_size=8,
    accum=4,
    burnin=0,
    rollout=5,
    device="cuda",
    cache_dir="/media/yh04/New Volume/datasets"
):

    lm = AutoregressiveLM(
        cls=RecurrentMemoryTransformer,
        vocab_size=vocab_size,
        max_len=seq_len,
        n_layers=n_layers,
        d_model=d_model,
        n_head=n_head,
        p=p,
        device=device,
        w=128,
        bsz=batch_size,
        topk=1,
        num_tokens=128,
        mem_tokens=64,
    )
    lm.load_pretrained()

    dataloader = DataLoader(
        TextDataset(
            name=dataset,
            cache_dir=cache_dir,
            split="train[:40000]",
            seq_len=seq_len,
            block_len=rollout,
            device=device,
            sep_padding=True,
            sep_padding_prob=0.7
        ),
        batch_size=batch_size,
    )

    trainer = AutoregressiveTrainer(
        model=lm,
        dataloader=dataloader,
        lr=lr,
        batch_size=batch_size,
        accum=accum,
        seqlen=seq_len,
        burnin=burnin,
        rollout=rollout,
        device=device
    )
    print("Starting training run...")

    epochs = 1000
    for epoch in range(epochs):
        trainer.run_epoch(epoch)


if __name__ == "__main__":
    main()

