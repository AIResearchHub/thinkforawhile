

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class XLAttention(nn.Module):
    """
    XL Attention that incorporates previous cached keys and values.
    Main difference from standard attention is this
            attn_score = torch.einsum('bhid,bojd->bhij', (q, k))
    to adjust for different length queries and key/value pair.
    """

    def __init__(self, d_model, n_head, device='cuda:0'):
        super(XLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.device = device

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_concat = nn.Linear(d_model, d_model, bias=True)

        self.to(device)  # Move the entire model to the device

    def forward(self, q, kv, mem=None, mask=None, is_causal=False):
        """
        Parameters:
        q:     [batch_size, length, d_model]
        kv:    [batch_size, length, d_model]
        mems:  [batch_size, mem_length, d_model]

        Returns:
        out:   [batch_size, length, d_model]
        """
        # batch_size, length, d_model = q.shape
        q = q.to(self.device)  # ensure q is on the correct device
        print(f"q device: {q.device}")

        kv = kv.to(self.device)  # ensure kv is on the correct device
        print(f"kv device: {kv.device}")

        if mem is not None:
            mem = mem.to(self.device)  # ensure mem is on the correct device
            print(f"mem device: {mem.device}")

            c = torch.cat([mem, kv], dim=1)
            mem_length = c.size(1) - q.size(1)
        else:
            c = kv

        # q  [batch_size, length, d_model]
        # kv [batch_size, length+mem_length, d_model]
        q, k, v = self.w_q(q), self.w_k(c), self.w_v(c)
        print(f"w_q weight device: {self.w_q.weight.device}")
        print(f"w_k weight device: {self.w_k.weight.device}")
        print(f"w_v weight device: {self.w_v.weight.device}")

        q, k, v = self.split(q), self.split(k), self.split(v)

        if mem is not None and mask is not None:
            mask = mask.to(self.device)  # ensure mask is on the correct device
            print(f"mask device: {mask.device}")

            mask = F.pad(mask, (mem_length, 0, 0, 0), value=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.w_concat(out)
        print(f"w_concat weight device: {self.w_concat.weight.device}")

        return out

    def split(self, tensor):
        tensor = tensor.view(tensor.size(0), tensor.size(1), self.n_head, self.d_head)
        tensor = tensor.transpose(1, 2)

        return tensor

