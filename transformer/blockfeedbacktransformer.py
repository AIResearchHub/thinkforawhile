

import torch
import torch.nn as nn

from .layer import TransformerEmbedding, XLCrossLayer, RecurrentLayer


class BlockFeedbackTransformer(nn.Module):
    """
    Block Feedback Transformer is an upgrade
    from Block Recurrent Transformer where the recurrent state
    is broadcasted to all layers to do cross attention
    after self attention and gradients are backpropagated
    by each layer. The improvements depend on the dataset
    (I assume datasets that require reasoning is improved
    and datasets that require memory is degraded)

    Args:
        w (int): window size to iterate over per sequence
        statelen (int): Length of the recurrent state e.g. (bsz, statelen, dim)
    """

    def __init__(self,
                 vocab_size,
                 max_len=512,
                 n_layers=4,
                 d_model=512,
                 n_head=8,
                 p=0.1,
                 w=512,
                 device="cuda:0",
                 statelen=32,
                 **kwargs
                 ):
        super(BlockFeedbackTransformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.w = w
        self.device = device

        # learnable init state
        self.init_state = nn.Parameter(torch.randn(statelen, d_model))

        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_len,
                                              device=device
                                              )
        self.layers1 = nn.ModuleList([XLCrossLayer(d_model=d_model,
                                                   ffn_hidden=4 * d_model,
                                                   n_head=n_head,
                                                   p=p
                                                   )
                                      for _ in range(n_layers // 2)])
        self.recurrent = RecurrentLayer(d_model=d_model,
                                        ffn_hidden=4 * d_model,
                                        n_head=n_head,
                                        p=p
                                        )
        self.layers2 = nn.ModuleList([XLCrossLayer(d_model=d_model,
                                                   ffn_hidden=4 * d_model,
                                                   n_head=n_head,
                                                   p=p
                                                   )
                                      for _ in range(n_layers - (n_layers // 2))])

        self.reset()

    def reset(self):
        self.xlmems = []
        self.state = None

    def set_state(self, state=None, xlmems=None):
        if state is not None:
            self.state = state
        if xlmems is not None:
            self.xlmems = xlmems

    def get_state(self):
        return self.state, self.xlmems

    # @torch.autocast("cuda", dtype=torch.float16)
    def forward(self, ids, is_causal):
        x = self.embedding(ids)

        bsz, seqlen, dim = x.shape

        if self.state is None:
            self.state = self.init_state.unsqueeze(0).repeat(bsz, 1, 1)

        out = []
        xs = x.split(self.w, dim=-2)

        for x in xs:
            self.xlmems = iter(self.xlmems)
            nextxlmems = []

            for layer in self.layers1:
                nextxlmems.append(x.detach())
                x = layer(x, mem=next(self.xlmems, None), state=self.state, is_causal=is_causal)

            x, next_state = self.recurrent(x, self.state, is_causal=is_causal)
            # x = self.recurrent(x, is_causal=is_causal)

            for layer in self.layers2:
                nextxlmems.append(x.detach())
                x = layer(x, mem=next(self.xlmems, None), state=self.state, is_causal=is_causal)

            out.append(x)
            self.xlmems = nextxlmems
            self.state = next_state

        out = torch.concat(out, dim=1)
        assert out.shape == (bsz, seqlen, dim)

        self.state = self.state.detach()
        return out

