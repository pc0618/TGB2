"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch_geometric.utils import scatter


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        if msg.size(0) == 0:
            return msg.new_zeros((dim_size, msg.size(-1)))
        # Avoid hard dependency on `torch_scatter` by using PyG's `scatter` (which
        # can fall back to torch.scatter_reduce) plus a stable tie-breaker.
        #
        # Goal: for each group in `index`, select the message corresponding to the
        # maximum timestamp `t`. If there are ties, pick the *latest row*.
        max_t = scatter(t, index, dim=0, dim_size=dim_size, reduce="max")  # [dim_size]
        is_max = t == max_t[index]  # [E]
        pos = torch.arange(t.size(0), device=t.device, dtype=torch.long)
        pos = torch.where(is_max, pos, pos.new_full(pos.shape, -1))
        argmax = scatter(pos, index, dim=0, dim_size=dim_size, reduce="max")  # [dim_size]

        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax >= 0  # groups with at least one entry
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")


class SumAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="sum")


class MaxAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="max")
