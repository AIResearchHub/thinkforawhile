

import torch
from torcheval.metrics.text import Perplexity


def test_perplexity(
    model,
    dataloader,
    device,
    sep_token=102,
    start_token=50,
    end_token=51,
    vocab_size=30522
):
    """
    Simple function that returns the perplexity on the dataset
    of the dataloader, excluding the [SEP] token, [START] token,
    and the [END] token.

    Args:
        model (nn.Module): Model to be evaluated
        dataloader (Dataloader): dataloader that has sep padding
        device (String): "cpu" or "cuda" or "cuda:n"
        sep_token (int=102): id for [SEP]
        start_token (int=50): id for [START]
        end_token (int=51): id for [END]
        vocab_size (int=30522): vocab size default to bert

    Returns:
        ppl (float): Perplexity

    """
    metric = Perplexity(device=device)

    for i, batch in enumerate(dataloader):
        bsz, timesteps, seqlen = batch.shape
        inputs, targets = batch[:, :, :-1], batch[:, :, 1:]
        seqlen -= 1

        with torch.no_grad():
            model.module.reset()
            for t in range(timesteps):
                expected = model(inputs[:, t, :])
                # expected = (bsz, seq_len, vocab_size)

                expected = expected.reshape(-1, vocab_size)
                targets_ = targets[:, t, :].reshape(-1)

                mask = (inputs[:, t, :].reshape(-1) != sep_token) & \
                       (inputs[:, t, :].reshape(-1) != start_token) & \
                       (inputs[:, t, :].reshape(-1) != end_token)

                assert targets_.shape == (bsz * seqlen,)
                assert expected.shape == (bsz * seqlen, vocab_size)

                targets_ = targets_[mask]
                expected = expected[mask]

                metric.update(expected.unsqueeze(0), targets_.unsqueeze(0))

    ppl = metric.compute()
    return ppl.item()

