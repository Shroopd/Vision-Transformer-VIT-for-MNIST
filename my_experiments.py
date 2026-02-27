# from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
import sys

from typing import Callable


EPSILON = sys.float_info.epsilon
"""A number that is very small, but not 0"""

CLOSE_ENOUGH = 1e-4
"""Small but not vanishingly so value, to determine if two values are almost equal"""

NOT_EPSILON = 1
"""A number that isn't very small, and definitely not 0"""


def swishmax(input: torch.Tensor, dim=0):
    xexp = input * torch.exp(input - torch.max(input, dim=dim, keepdim=True).values)
    out = torch.div(
        xexp, (torch.sum(torch.abs(xexp), dim=dim, keepdim=True) + NOT_EPSILON)
    )
    return out


def near_zero(input: torch.Tensor):
    return CLOSE_ENOUGH > torch.max(torch.abs(input))


def near_equal(a: torch.Tensor, b: torch.Tensor):
    return near_zero(a - b)


def make_weight(*dims: int):
    return Parameter(torch.randn(dims))


nn.functional.linear


class AttentionMeta(nn.Module):
    def __init__(
        self,
        key_transform: nn.Module,
        query_transform: nn.Module,
    ) -> None:
        super().__init__()
        self.make_keys = key_transform
        self.make_query = query_transform


class AttentionMono(nn.Module):
    def __init__(
        self,
        token_size: int,
        attention_size: int,
        mask: nn.Module = nn.Identity(),
        *,
        key_bias=False,
        query_bias=False,
        attention_bias=False,
        value_down_bias=False,
        value_up_bias=False,
    ) -> None:
        super().__init__()

        self.key_down = nn.Linear(token_size, attention_size, key_bias)
        self.query_down = nn.Linear(token_size, attention_size, query_bias)

        self.attention_over = nn.Linear(attention_size, attention_size, attention_bias)

        self.value_down = nn.Linear(token_size, attention_size, value_down_bias)
        self.value_up = nn.Linear(attention_size, token_size, value_up_bias)

        self.mask = mask

    def forward(
        self,
        key_tokens: torch.Tensor,
        query_tokens: torch.Tensor | None = None,
    ):
        """
        number of queries:  Q
        length of queries:  QT
        number of keys:     K
        length of keys:     KT

        query_tokens.shape  [..., Q, T]
        key_tokens.shape    [..., K, T]

        batching and broadcasting for dimensions prior to last two
        """

        if query_tokens is None:
            query_tokens = key_tokens

        key = self.key_down(key_tokens)
        # [..., K, A]
        query = self.query_down(query_tokens)
        # [..., Q, A]

        attention_raw = key.unsqueeze(-2) * query.unsqueeze(-3)
        # [..., K, Q, A] = [..., K, 1, A] * [..., 1, Q, A]
        attention_logits = self.mask(self.attention_over(attention_raw))
        # [..., K, Q, A]
        attention_scale = swishmax(attention_logits, dim=-2)
        # [..., K, Q, A]

        value = self.value_down(key_tokens)
        # [..., K, A]
        values_scaled = value.unsqueeze(-2) * attention_scale
        # [..., K, Q, A] = [..., K, 1, A] * [..., K, Q, A]
        value_sum = values_scaled.sum(-3)
        # [..., Q, A]

        value_tokens = self.value_up(value_sum)
        # [..., Q, T]

        return value_tokens


class AttentionZP(nn.Module):
    # """One query vector per operation"""

    def __init__(
        self,
        key_size: int,
        query_size: int,
        heads: int,
        attention_size: int,
        compress_size: int,
        bias_query: bool,
        return_attention: bool = False,
        mask: Callable[[torch.Tensor], torch.Tensor] | None = None,
        dropout = 0.0
    ) -> None:
        super().__init__()

        # should preserve 0s
        self.key_down = make_weight(heads, key_size, attention_size)

        # should preserve 0s
        self.query_down = make_weight(heads, query_size, attention_size)
        # should NOT preserve 0s
        self.bias_query = bias_query
        if bias_query:
            self.query_down_bias = make_weight(heads, 1, attention_size)

        # should preserve 0s
        self.compress = query_size > compress_size * 2
        if self.compress:
            self.value_down = make_weight(heads, query_size, compress_size)
            self.value_up = make_weight(heads, compress_size, query_size)
        else:
            self.value_over = make_weight(heads, query_size, query_size)

        self.return_attention = return_attention
        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        key_tokens: torch.Tensor,
        query_tokens: torch.Tensor | None = None,
    ):
        """
        number of queries:  Q
        length of queries:  QT
        number of keys:     K
        length of keys:     KT
        number of heads:    H

        query_tokens.shape  [..., Q, T]
        key_tokens.shape    [..., K, T]

        batching and broadcasting for dimensions prior to last two
        """

        if query_tokens is None:
            query_tokens = key_tokens

        key_tokens = key_tokens.unsqueeze(-3)
        # [..., 1, K, KT]

        query_tokens = query_tokens.unsqueeze(-3)
        # [..., 1, Q, QT]

        key = key_tokens @ self.key_down
        # [..., H, K, A]

        query = query_tokens @ self.query_down
        if self.bias_query:
            query = query + self.query_down_bias
        # [..., H, Q, A]

        attention_logits = key @ query.transpose(-2, -1)
        # [..., H, K, Q]

        attention_dist_keys = swishmax(attention_logits, dim=-2)
        # [..., H, K, Q]

        values_scaled = key_tokens.unsqueeze(-2) * attention_dist_keys.unsqueeze(-1)
        # [..., H, K, Q, T] = [..., 1, K, 1, KT] * [..., H, K, Q, 1]

        value_shift = values_scaled.sum(-3)
        # [..., H, Q, T]

        if self.compress:
            value_shift @= self.value_down
            # [..., H, Q, A]
            value_shift @= self.value_up
            # [..., H, Q, T]
        else:
            value_shift @= self.value_over
            # [..., H, Q, T]

        value_shift = value_shift.sum(-3)
        #  [..., Q, T]

        # if self.return_attention:
        #     return value_shift, attention_logits
        # else:
        return self.dropout(value_shift)


class NaiveUnaryZP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.centered_module = module
        self.output_zero = None
        self.h = self.centered_module.register_full_backward_hook(self.reset)

    def reset(self, *_):
        self.output_zero = None

    def forward(self, x):
        if self.output_zero is None:
            self.output_zero = self.centered_module.forward(torch.zeros_like(x))
        return self.centered_module(x) - self.output_zero


class LinearActivateZP(nn.Linear):
    """
    Linear with bias but it's a zero preserving unary function
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        # bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features, out_features, bias=True, device=device, dtype=dtype
        )
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.activation(super().forward(input)) - self.activation(self.bias)


class PosEncode(torch.nn.Module):
    def __init__(
        self,
        dim_pairs,
        scale_divisor,
        exp_factor=1.0,
    ) -> None:
        super().__init__()
        self.D = dim_pairs
        self.M = scale_divisor / (math.pi * 2)
        self.I = torch.nn.Buffer(torch.arange(0, self.D) + 1)
        self.I = self.I * (exp_factor ** (self.I / self.D))

    def forward(self, x) -> torch.Tensor:
        x = x.unsqueeze(-1)
        out_shape = list(x.size())
        out_shape[-1] = self.D * 2
        out = torch.zeros(out_shape)
        out[..., 0::2] = torch.sin((x / self.M) * self.I)
        out[..., 1::2] = torch.cos((x / self.M) * self.I)
        return out


class Conv2DAttention(nn.Module):
    # def __init__(self, *args, ) -> None:
    #     super().__init__(*args, )
    def __init__(self, attention_block: AttentionZP) -> None:
        super().__init__()
        self.attention = attention_block
        self.pad = nn.CircularPad2d(3 // 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        width = 3

        image = self.pad(image)

        sections = image.unfold(-2, width, 1).unfold(-2, width, 1)
        sections = sections.reshape(list(sections.shape[:-2]) + [-1])
        middle = sections.shape[-1] // 2

        # print(middle)

        # print(sections.shape)

        keys = torch.cat((sections[..., :middle], sections[..., middle + 1 :]), dim=-1)
        # print(keys.shape, "K")

        query = sections[..., middle : middle + 1]
        # print(query.shape, "Q")

        keys = keys.movedim(1, -1)
        query = query.movedim(1, -1)

        # print(keys,query)

        outs, _ = self.attention.forward(keys, query)

        return outs.movedim(-1, 1)
