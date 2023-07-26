

import torch
import torch.nn as nn
from torch.optim import AdamW

import time
import datetime


class AutoregressiveTrainer:
    """
    Trains a Large Language Model to predict the next word using causal mask.
    """

    def __init__(self,
                 model,
                 dataloader,
                 lr,
                 batch_size,
                 seqlen,
                 burnin,
                 rollout,
                 device="cuda",
                 ):

        self.model = nn.DataParallel(model).to(device)
        self.opt = AdamW(self.model.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.dataloader = dataloader
        self.batch_size = batch_size

        self.seqlen = seqlen
        self.burnin = burnin
        self.rollout = rollout
        self.length = burnin + rollout

        self.dt = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
        self.log = open(f"logs/{self.dt}", "w")
        self.start = time.time()
        self.updates = 0

    def run_epoch(self, epoch):
        """
        Main training loop for one epoch.
        Logs and prints time and loss.

        Args:
            epoch (int): The epoch number

        """

        for i, batch in enumerate(self.dataloader):
            if batch.size(0) != self.batch_size:
                continue

            # batch (bsz, block_len, seq_len)
            loss = self.step(batch)

            self.log.write(f"{time.time() - self.start}, {loss}\n")
            self.log.flush()

            self.updates += 1

            if i % 5 == 0:
                print(f"Epoch: {epoch} \t "
                      f"Time: {time.time() - self.start} \t "
                      f"Loss: {loss} \t "
                      f"Sec/Update: {(time.time() - self.start) / self.updates}")

            if i % 1000 == 0:
                self.model.module.reset()
                torch.save(self.model, "saved/arxivrecsep120000ppl23")

    def step(self, batch):
        """
        TODO:
            Accumulate gradients

        A training step that does backpropagation at each rollout timestep.
        To train long sequence transformer models such as Transformer XL.

        Args:
            batch (B, T, S+1): batch to be trained on

        Returns:
            loss (float): Total loss normalized by T and S

        """
        total_loss = 0
        inputs, targets = batch[:, :, :-1], batch[:, :, 1:]

        self.model.module.reset()
        for t in range(self.rollout):
            self.opt.zero_grad()
            expected = self.model(inputs[:, t, :])
            loss = self.cross_entropy_loss(expected, targets[:, t, :])
            loss.backward()

            total_loss += loss.item()

        for x in self.model.parameters():
            if x.grad is not None:
                x.grad.data.mul_(1/self.rollout)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.opt.step()

        return total_loss / (self.rollout * self.seqlen)

    def cross_entropy_loss(self, expected, target):
        """
        cross entropy loss

        Args:
            expected (batch_size, max_len, vocab_size)
            target (batch_size, max_len)

        Returns:
            loss (,)
        """
        assert target.shape == (target.size(0), target.size(1))
        assert expected.shape == (target.size(0), target.size(1), expected.size(2))

        loss = self.criterion(expected.reshape(-1, expected.size(2)), target.reshape(-1))
        return loss.mean()
