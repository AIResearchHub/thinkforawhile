

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, XLAttentionLayer
from transformers import TransfoXLModel


class TransformerXL(nn.Module):
    """
    Transformer XL is an upgrade to Transformer that caches
    the previous keys and values to be concatenated with
    current keys and values for Extra Long sequence lengths.

    Args:
        vocab_size (int): Vocabulary size
        max_len (int): Max length
        n_layers (int): Number of layers
        d_model (int): Dimension of transformer
        n_head (int): Number of attention heads
        p (int): Dropout probability

    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 device="cuda:0",
                 **kwargs
                 ):

        super(TransformerXL, self).__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.device = device

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device)

        self.layers = nn.ModuleList([XLAttentionLayer(d_model=d_model,
                                                      ffn_hidden=4 * d_model,
                                                      n_head=n_head,
                                                      p=p)
                                    for _ in range(n_layers)])

        self.reset()

    def reset(self):
        self.state = None

    def set_state(self, state):
        self.state = state

    def get_state(self):
        return self.state

    def forward(self, ids, is_causal):
        """
        Computes transformer xl output
        Layer takes in (length, batch_size, d_model) so transpose before and after layers

        Parameters:
        ids (Tensor[batch_size, length]): tokens
        state (Tensor[batch_size, state_len, d_model]): recurrent state

        Returns:
        x (Tensor[batch_size, length, d_model]): output
        state (Tensor[batch_size, length, d_model]): next recurrent state

        """
        bsz = ids.size(0)

        if self.state is None:
            self.state = torch.zeros(self.n_layers, bsz, self.max_len, self.d_model, device=self.device)

        x = self.embedding(ids)

        next_state = []
        for layer, s in zip(self.layers, self.state):
            next_state.append(x.detach())
            x = layer(x, s, is_causal=is_causal)

        self.state = torch.stack(next_state)

        return x

