# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""PyTorch TELECHAT model."""

import warnings
from typing import Optional, Tuple, Union, List, Dict
from threading import Thread

import torch
import math
import copy
from torch import nn
import torch.utils.checkpoint
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MoeModelOutputWithPast
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput
from transformers import GenerationConfig

from .configuration_telechat2 import Telechat2Config
from dataclasses import dataclass

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "telechat"
_CONFIG_FOR_DOC = "Telechat2Config"

TELECHAT_PRETRAINED_MODEL_ARCHIVE_LIST = []

try:
    from einops import rearrange
except ImportError:
    rearrange = None

use_flash_attn = True
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None


@dataclass
class MoECausalLMOutputWithCrossAttentions(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None

@dataclass
class HWMoECausalLMOutputWithCrossAttentions(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    lm_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    hidden_state_loss: Optional[torch.FloatTensor] = None
    kl_loss: Optional[torch.FloatTensor] = None
    router_bucket_status: Optional[torch.FloatTensor] = None




class RotaryEmbedding(torch.nn.Module):
    # Extracted from: https://github.com/EleutherAI/gpt-neox
    def __init__(self, dim, config, base=10000, precision=torch.bfloat16):
        super().__init__()
        self.config = config
        self.dim = dim
        self.base = base
        self.inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float().half() / dim)).cuda()
        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def get_mscale(self, scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / self.config.base_seqlen, 2) + 1
        # ntk_alpha = 2 ** context_value - 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(self, x, seq_dim=0, seq_len=None):
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            if seq_len is None:
                seq_len = x.shape[seq_dim]
            seq_len = max(seq_len, self.config.training_seqlen)
            ntk_alpha = self.get_ntk_alpha(seq_len)
            self.mscale = float(self.get_mscale(seq_len / self.config.training_seqlen))
            if True:
                base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
                self.inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
                self.max_seq_len_cached = seq_len
                t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
                freqs = torch.einsum('i,j->ij', t, self.inv_freq)
                # Different from paper, but it uses a different permutation in order to obtain the same calculation
                emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
                if self.precision == torch.bfloat16:
                    emb = emb.float()
                # [sx, 1 (b * np), hn]
                self.cos_cached = self.mscale * emb.cos()[:, None, :]# .hatlf()
                self.sin_cached = self.mscale * emb.sin()[:, None, :]# .hatlf()
                if self.precision == torch.bfloat16:
                    self.cos_cached = self.cos_cached.bfloat16()
                    self.sin_cached = self.sin_cached.bfloat16()
                else:
                    raise NotImplementedError
            return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


# rotary pos emb helpers:
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):  # jitting fails with bf16
    cos, sin = cos[offset:q.shape[0] + offset, ...], sin[offset:q.shape[0] + offset, ...]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MixedFusedRMSNorm(nn.Module):
    # Extracted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FlashSelfAttention(torch.nn.Module):
    # Extracted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/model/transformer.py
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        q = q.bfloat16()
        k = k.bfloat16()
        v = v.bfloat16()
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)
        # self.training = True
        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        # output = output.float()

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)

        return output


def _make_causal_mask(
        input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty((target_length, target_length + past_key_values_length), dtype=torch.bool, device=device)
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def telechat_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def telechat_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return telechat_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = telechat_gelu_back(grad_output, input)
        return tmp


class TelechatGelu(nn.Module):
    """
    TelechatBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return telechat_gelu_forward(x)


class TelechatAttention(nn.Module):
    def __init__(self, config: Telechat2Config, layer_idx):
        super().__init__()
        self.kv_cache = None
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.config = config

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
        self.kv_projection_size = self.head_dim * self.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key_value = nn.Linear(self.hidden_size, self.kv_projection_size * 2, bias=False)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        if self.config.flash_attn:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config=config, precision=torch.bfloat16)
        else:
            self.rotary_emb = RotaryEmbedding(self.head_dim, config=config)

        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=config.attention_dropout
        )

        self.last_key_layer = None
        # logn_list = [math.log(i, 4096) if i > 4096 else 1 for i in range(1, 32768)]
        # self.logn_tensor = torch.tensor(logn_list)[None, :, None, None].half().cuda()

    def repeat_kv(self, hidden_states, n_rep):
        slen, batch, num_key_value_heads_per_partition, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, None, :].expand(slen, batch, num_key_value_heads_per_partition, n_rep,
                                                               head_dim)
        return hidden_states.reshape(slen, batch, num_key_value_heads_per_partition * n_rep, head_dim)

    def split_tensor_along_last_dim(self,
                                    tensor: torch.Tensor,
                                    num_partitions: int,
                                    contiguous_split_chunks: bool = False,
                                    ):

        # Get the size and dimension.
        last_dim = tensor.dim() - 1
        last_dim_size = tensor.size()[last_dim] // num_partitions
        # Split.
        tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
        # Note: torch.split does not create contiguous tensors by default.
        if contiguous_split_chunks:
            return tuple(chunk.contiguous() for chunk in tensor_list)

        return tensor_list

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            residual: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
    ):
        hidden_states = hidden_states.transpose(1, 0)
        query_layer = self.query(hidden_states)
        new_tensor_shape = query_layer.size()[:-1] + \
                           (self.num_heads,
                            self.head_dim)
        query_layer = query_layer.view(*new_tensor_shape)

        mixed_kv_layer = self.key_value(hidden_states)
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                           (self.num_key_value_heads,
                            2 * self.head_dim)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)
        (key_layer, value_layer) = self.split_tensor_along_last_dim(mixed_kv_layer, 2)

        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0),
                       key_layer.size(2)
                       )

        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[4], -1)

        apply_rotary_fn = apply_rotary_pos_emb_torch

        seq_len = key_layer.shape[0]
        offset = 0

        if use_cache and layer_past != None:
            past_key, past_value = layer_past
            offset = past_key.shape[0]
            seq_len += offset

        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)



        query_layer, key_layer = apply_rotary_fn(query_layer, key_layer, cos, sin, offset=offset)

        if use_cache:
            if layer_past != None:
                past_key, past_value = layer_past
                key_layer = torch.cat((past_key, key_layer[-1, ...].unsqueeze(0)), dim=0)
                value_layer = torch.cat((past_value, value_layer[-1, ...].unsqueeze(0)), dim=0)
            layer_past = key_layer, value_layer

        s_value, bz, kv_head, dim = value_layer.shape
        s_key = key_layer.shape[0]
        s_query = query_layer.shape[0]
        q_head = output_size[1]

        query_layer = query_layer.reshape((s_query, bz, q_head, dim))
        key_layer = key_layer.reshape((s_key, bz, kv_head, dim))

        key_layer = self.repeat_kv(key_layer, self.num_key_value_groups)
        value_layer = self.repeat_kv(value_layer, self.num_key_value_groups)

        if self.config.flash_attn:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous() for x in
                       (query_layer, key_layer, value_layer)]

            context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> b s (h d)').contiguous()
        else:
            assert 1==0
            ##[sq, b, np, hn] -> [sq, b * np, hn]
            query_layer = query_layer.reshape(s_query, bz * self.num_heads, dim)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.reshape(s_key, bz * self.num_heads, dim)
            matmul_result = self.inv_norm_factor * torch.einsum('bik,bkj->bij', query_layer.transpose(0, 1),
                                                                key_layer.transpose(0, 1).transpose(1, 2))

            attention_scores = matmul_result.view(bz, self.num_heads, s_query, s_key)

            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16:
                attention_scores = attention_scores.to(torch.float)
            attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
            attention_probs = F.softmax(attn_weights, dim=-1).to(input_dtype)  ##dtype = torch.float32
            attention_probs = self.attention_dropout(attention_probs)
            attention_probs_reshaped = attention_probs.view(bz * self.num_heads, s_query, s_key)

            value_layer = value_layer.reshape(s_key, bz * self.num_heads, dim)
            context_layer = torch.bmm(attention_probs_reshaped, value_layer.transpose(0, 1))
            context_layer = self._merge_heads(context_layer)

        # print(self.dense.weight.dtype, context_layer.dtype)
        # assert 1==0

        output_tensor = self.dense(context_layer)



        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        present = None
        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return output_tensor, layer_past


class TelechatMLP(nn.Module):
    def __init__(self, config: Telechat2Config):
        super().__init__()
        hidden_size = config.hidden_size
        self.gate_proj = nn.Linear(hidden_size, config.ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, config.ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_size, hidden_size, bias=True)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_output = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return intermediate_output

# Copied from transformers.models.mixtral.modeling_mixtral.load_balancing_loss_func
def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class SwitchMLP(nn.Module):
    def __init__(self, config: Telechat2Config):
        super().__init__()
        self.config = config
        self.num_expert = config.num_moe_experts
        self.hidden_size = config.hidden_size
        self.top_k = config.expert_chosen
        # self.router = nn.Linear(self.hidden_size, self.num_expert, bias=False, dtype=torch.float32)
        self.router = nn.Linear(self.hidden_size, self.num_expert, bias=False)
        self.local_experts = nn.ModuleList()
        for i in range(self.num_expert):
            self.local_experts.append(TelechatMLP(config))
        self.training = False
        self.hidden_dropout = 0.0

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        # if self.config.moe_input_jitter_eps is not None:
        #     eps = self.config.moe_input_jitter_eps
        #     if self.input_jitter is None:
        #         self.input_jitter = torch.distributions.uniform.Uniform(
        #             torch.tensor(1.0 - eps, device=input.device),
        #             torch.tensor(1.0 + eps, device=input.device),
        #         ).rsample
        #     return input * self.input_jitter(input.shape)
        # else:
        #     return input
        return input

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        # if self.config.moe_z_loss_coeff is not None:
        #     assert 1 == 0
        #     z_loss = z_loss_func(logits, self.config.moe_z_loss_coeff)
        #     logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        #     save_to_aux_losses_tracker(
        #         "z_loss",
        #         z_loss / self.config.moe_z_loss_coeff,
        #         self.layer_number,
        #         self.config.num_layers,
        #     )
        return logits

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The scores and the indices tensor after applying load balancing.
        """

        if self.config.expert_dropout:
            masked_logits = self.drop_expert(logits, drop_prob=self.config.expert_dropout_prob)
            top_logits, indices = torch.topk(masked_logits, k=self.topk, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(masked_logits)
        else:
            top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)

        # Apply load balancing loss
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        scores = self.apply_load_balancing_loss(probs, indices, activation=scores, )
        return scores, indices  ###[2048,2], [2048,2]

    def apply_load_balancing_loss(
            self,
            probs: torch.Tensor,
            indices: torch.Tensor,
            activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            loss_func (callable): The loss function to be used.
            probs (torch.Tensor): The probabilities output by the MoE layer.
            indices (torch.Tensor): The indices of the selected experts.
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        mask = torch.nn.functional.one_hot(indices, num_classes=self.num_experts).sum(dim=1)
        aux_loss = switch_load_balancing_loss_func(
            probs, mask,
            self.config.moe_aux_loss_coeff)
        save_to_aux_losses_tracker(
            "load_balancing_loss",
            aux_loss / self.config.moe_aux_loss_coeff,
            self.layer_number,
            self.config.num_layers,
        )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation


    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
        """
        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        # if (
        #         self.config.tensor_model_parallel_size > 1
        #         and self.config.moe_token_dispatcher_type == "alltoall"
        # ):
        #     # Gather the logits from the TP region
        #     logits = gather_from_sequence_parallel_region(logits)
        if self.routing_type == "sinkhorn":
            assert 1==0
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            if self.aux_loss_free:
                logits = logits + self.expert_bias.expand(logits.shape[0], -1).to(logits.device) ### add logits by expert bias

            top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
            if self.use_sigmoid:
                scores = torch.sigmoid(top_logits)
            else:
                scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")

        return scores, indices


    def forward(self, hidden_states, residual, expert_limit):
        b = hidden_states.size(0)
        s = hidden_states.size(1)
        h = hidden_states.size(2)
        hidden_states = self.apply_input_jitter(hidden_states)

        # print(hidden_states.shape)  # torch.Size([batch size, seq length 8193, hidden size 2048])
        router_logits = self.router(hidden_states)
        # print(route.shape)  # torch.Size([batch size, seq length 8193, num experts 16])
        if expert_limit is None:
            router_logits = router_logits.view(-1, self.config.num_moe_experts)  # logits = logits.view(-1, self.config.num_moe_experts)
        else:
            expert_mask = torch.arange(self.num_expert) < expert_limit
            expert_mask = expert_mask.to(router_logits.device)
            router_logits = router_logits.masked_fill(~expert_mask, float('-inf')).view(-1, self.config.num_moe_experts)

        # print(router_logits.shape)

        topk_weights, topk_ind = torch.topk(router_logits, self.top_k, dim=-1)  ##[33,2]. max_ind:[[7,3],[7,2],...]; topk_weight: [[0.4,0.6],[0.5,0.5],...]
        topk_weights = torch.softmax(topk_weights, dim=-1, dtype=torch.float32).type_as(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.size(2))

        # print(f"topk weights {topk_weights.shape}, {topk_weights}")
        # print(f"topk index,  {topk_ind.shape} {topk_ind}")

        # topk_weights = topk_weights.view(-1, topk_weights.size(2))
        # topk_ind = topk_ind.view(-1, topk_ind.size(2))
        output_total = torch.zeros_like(hidden_states).to(hidden_states)
        for expert_num, expert in enumerate(self.local_experts):
            # print(expert_num)
            sample_ind, expert_ind = torch.where(topk_ind == expert_num)
            hidden = hidden_states[sample_ind.unsqueeze(1), :]  ###[chosen_seqlen,1,3072]
            expert_output = expert(hidden)
            output_total[sample_ind] += torch.mul(expert_output.squeeze(1),
                                                  topk_weights[sample_ind, expert_ind].unsqueeze(1))
        output_total = output_total.view(b, s, h)
        output = dropout_add(output_total, residual, self.hidden_dropout, self.training)

        return output, router_logits


class TelechatBlock(nn.Module):
    def __init__(self, config: Telechat2Config, layer_idx):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = MixedFusedRMSNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.layer_idx = layer_idx
        self.self_attention = TelechatAttention(config, layer_idx)
        self.post_attention_layernorm = MixedFusedRMSNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = SwitchMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_router_logits: Optional[bool] = False,
            expert_limit: Optional[int] = None,
    ):
        layernorm_output = self.input_layernorm(hidden_states)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        layernorm_output = self.post_attention_layernorm(attention_output)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output
        output = self.mlp(layernorm_output, residual, expert_limit)
        if isinstance(output, tuple):
            output, router_logits = output
        else:
            router_logits = None

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class TelechatPreTrainedModel(PreTrainedModel):
    config_class = Telechat2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TelechatBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False):
        if isinstance(module, TelechatModel):
            module.gradient_checkpointing = value


class TelechatModel(TelechatPreTrainedModel):
    def __init__(self, config: Telechat2Config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        if self.config.embed_layernorm:
            self.word_embeddings_layernorm = MixedFusedRMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.h = nn.ModuleList([TelechatBlock(config, _) for _ in range(config.num_hidden_layers)])
        self.ln_f = MixedFusedRMSNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
            self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_hidden_states_layer: Optional[int] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            expert_limit: Optional[int] = None,
            multi_forward_expert_list: List = [],
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        # input_ids = torch.load("Megatron-LM-0624-3B/tensors/input_ids.pt").to(input_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds
        # print(f"[INFO_Telechat]: inputs_embeds={inputs_embeds}")
        if self.config.embed_layernorm:
            hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        all_router_logits = () if output_router_logits else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )
        if expert_limit is None:
            # print(f"[INFO_Telechat]: word_embeddings_layernorm={hidden_states}")
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
                if output_hidden_states:
                    if output_hidden_states_layer is None:
                        all_hidden_states = all_hidden_states + (hidden_states,)
                    else:
                        if i in output_hidden_states_layer:
                            all_hidden_states = all_hidden_states + (hidden_states,)
                        else:
                            all_hidden_states = all_hidden_states + (torch.Tensor([0]),)

                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, use_cache=use_cache, output_attentions=output_attentions, output_router_logits=output_router_logits)

                        return custom_forward

                    outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        causal_mask,
                        layer_past,
                        use_reentrant=False,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=causal_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                    )

                # print(f"[INFO_Telechat]: outputs{i}={outputs}")
                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

                if output_router_logits and outputs[-1] is not None:
                    all_router_logits += (outputs[-1],)
        else:
            # print(f"[INFO_Telechat]: word_embeddings_layernorm={hidden_states}")
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, use_cache=use_cache, output_attentions=output_attentions,
                                          output_router_logits=output_router_logits, expert_limit=expert_limit)

                        return custom_forward

                    outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        causal_mask,
                        layer_past,
                        use_reentrant=False,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=causal_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        expert_limit=expert_limit,
                    )

                # print(f"[INFO_Telechat]: outputs{i}={outputs}")
                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

                if output_router_logits and outputs[-1] is not None:
                    all_router_logits += (outputs[-1],)

        hidden_states = self.ln_f(hidden_states)
        # print(f"[INFO_Telechat]: hidden_states={hidden_states}")
        # ref = torch.load("Megatron-LM-0624-3B/tensors/final_layernorm.pt")
        # print(hidden_states.squeeze()[2048:])
        # print(ref.squeeze())
        # print(torch.max(hidden_states.squeeze()[2048:] - ref.squeeze().to(hidden_states.device)))
        # exit()
        # print(ref.shape,hidden_states.shape)
        # print(hidden_states)
        # exit()
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_router_logits] if v is not None)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            router_logits=all_router_logits,
        )


class Telechat2ForCausalLM(TelechatPreTrainedModel):
    # _tied_weights_keys = ["lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: Telechat2Config):
        super().__init__(config)
        self.config = config
        self.transformer = TelechatModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.moe_aux_loss_coeff = config.moe_aux_loss_coeff
        self.multi_forward_expert_list = config.multi_forward_expert_list

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> dict:
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], HWMoECausalLMOutputWithCrossAttentions]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )


        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )
            # print(f"loss is {loss}")

        lm_loss = loss.clone().detach() if loss is not None else None

        aux_loss = None
        if output_router_logits and self.training:
            aux_loss = load_balancing_loss_func(
                transformer_outputs.router_logits if return_dict else transformer_outputs[-1],
                self.config.num_moe_experts,
                self.config.expert_chosen,
                attention_mask,
            )
            # print(f"aux loss is {aux_loss}")

            if labels is not None:
                loss += self.moe_aux_loss_coeff * aux_loss.to(loss.device)  # make sure to reside in the same device
                # print(f"After add, loss is {loss}")


        if not return_dict:
            assert 1==0
            output = (lm_logits,) + transformer_outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return ((loss,) + output) if loss is not None else output

        # assert 1==0

        # print(f"forward return loss {loss}")


        return HWMoECausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            last_hidden_states = hidden_states,
            attentions=transformer_outputs.attentions,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
        )

    def single_forward_with_expert_limit(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            expert_limit: Optional[int] = None,
            output_router_bucket_status: Optional[bool] = None,
            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], HWMoECausalLMOutputWithCrossAttentions]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )


        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            expert_limit=expert_limit,
        )
        hidden_states = transformer_outputs.last_hidden_state
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )
            # print(f"loss is {loss}")
        lm_loss = loss.clone().detach() if loss is not None else None

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                transformer_outputs.router_logits if return_dict else transformer_outputs[-1],
                self.config.num_moe_experts,
                self.config.expert_chosen,
                attention_mask,
            )
            # print(f"aux loss is {aux_loss}")

            if labels is not None:
                loss += self.moe_aux_loss_coeff * aux_loss.to(loss.device)  # make sure to reside in the same device
                # print(f"After add, loss is {loss}")


        if not return_dict:
            assert 1==0
            output = (lm_logits,) + transformer_outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return ((loss,) + output) if loss is not None else output

        # assert 1==0

        # print(f"forward return loss {loss}")

        router_bucket_status = None
        if output_router_bucket_status:
            # 初始化一个列表来存储每层的激活专家信息
            router_bucket_status = []

            # 遍历每一层的router_logits
            for layer_idx, router_logits in enumerate(transformer_outputs.router_logits):

                router_logits = router_logits[:-1, :]  # last one in each sequence is not used
                # 获取top-2专家的索引
                topk_indices = router_logits.topk(2, dim=-1).indices  # (batch_size*seq_len, 2)

                # # 统计每个专家被选中的次数
                # expert_counts = torch.zeros(router_logits.size(-1), device=router_logits.device)
                # for expert_idx in topk_indices.view(-1):
                #     expert_counts[expert_idx] += 1

                # 使用scatter_add_加速统计每个专家被选中的次数
                expert_counts = torch.zeros(router_logits.size(-1), device=router_logits.device)
                # 将topk_indices展平并创建值为1的相同形状张量
                flat_indices = topk_indices.view(-1)
                ones = torch.ones_like(flat_indices, dtype=torch.float)
                # 使用scatter_add_进行统计
                expert_counts.scatter_add_(0, flat_indices, ones)

                # 记录该层的激活信息
                layer_status = {
                    "layer_index": layer_idx,
                    "topk_indices": topk_indices,
                    "expert_counts": expert_counts,
                    "most_used_experts": expert_counts.topk(2).indices.tolist()  # 最常使用的两个专家
                }
                router_bucket_status.append(layer_status)

            # # 打印简要信息
            # print(f"Router bucket status for {len(router_bucket_status)} layers:")
            # for status in router_bucket_status:
            #     print(f"Layer {status['layer_index']}: Top experts {status['most_used_experts']}")


        return HWMoECausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            last_hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            router_bucket_status=router_bucket_status,
        )

    def kd_loss_function(self, output, target_output, temperature):
        """Compute kd loss"""
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """

        output = output / temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def multi_forward_with_expert_list(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], HWMoECausalLMOutputWithCrossAttentions]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        multi_forward_result = {}

        for expert_limit in self.multi_forward_expert_list:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True and self.config.distill_all_hidden_states,
                output_hidden_states_layer=self.config.output_hidden_states_layer,
                output_router_logits=output_router_logits,
                return_dict=True,
                expert_limit=expert_limit,
            )
            multi_forward_result[expert_limit] = transformer_outputs


        # For every combination of experts, we calculate the language model loss
        final_loss = 0
        lm_loss = {}
        hidden_state_losses = {}
        kl_losses = {}

        # Get the full expert model outputs as teacher
        teacher_expert_limit = self.config.num_moe_experts
        teacher_all_results = multi_forward_result[teacher_expert_limit]
        teacher_last_hidden_state = teacher_all_results.last_hidden_state
        teacher_middle_hidden_states = teacher_all_results.hidden_states[:-1] if teacher_all_results.hidden_states is not None else None
        teacher_logits = None

        for expert_limit in self.multi_forward_expert_list[::-1]:

            last_hidden_states = multi_forward_result[expert_limit].last_hidden_state
            all_hidden_states = multi_forward_result[expert_limit].hidden_states[:-1] if multi_forward_result[expert_limit].hidden_states is not None else None  # Note: the last one equals to transformer_outputs.last_hidden_state
            lm_logits = self.lm_head(last_hidden_states)
            if expert_limit == teacher_expert_limit:
                teacher_logits = lm_logits
                shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
            # print(f"expert_limit {expert_limit} teacher_logits is {teacher_logits}")

            if labels is not None:
                labels = labels.to(lm_logits.device)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                batch_size, seq_length, vocab_size = shift_logits.shape
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
                )
                lm_loss[f"lm_loss_{expert_limit}"] = loss.clone().detach().cpu()
                final_loss += loss

            # 2. Hidden state loss (feature distillation)
            if expert_limit != teacher_expert_limit:
                # Distill all hidden states or just the last one
                if self.config.distill_all_hidden_states:
                    assert 1==0
                    assert len(all_hidden_states) == len(teacher_middle_hidden_states)
                    hidden_loss = 0
                    for stu_hid, tea_hid in zip(all_hidden_states, teacher_middle_hidden_states):
                        hidden_loss += F.mse_loss(stu_hid, tea_hid.detach())
                else:
                    hidden_loss = F.mse_loss(last_hidden_states, teacher_last_hidden_state)

                hidden_state_losses[f"hidden_loss_{expert_limit}"] = hidden_loss.clone().detach().cpu()
                final_loss += self.config.hidden_loss_weight * hidden_loss

            del all_hidden_states, last_hidden_states

            # 3. KL divergence loss (output distillation)
            if expert_limit != teacher_expert_limit:
                # Shift tokens for teacher and student logits
                # shift_student_logits = shift_logits

                # Temperature scaling
                teacher_probs = F.softmax(shift_teacher_logits / self.config.temperature, dim=-1)

                # print(f"expert limit {expert_limit}")
                kl_loss = self.kd_loss_function(shift_logits, teacher_probs.detach(), temperature=self.config.temperature) * (self.config.temperature ** 2)

                kl_losses[f"kl_loss_{expert_limit}"] = kl_loss.clone().detach().cpu()
                final_loss += self.config.kl_loss_weight * kl_loss


        del teacher_all_results, teacher_last_hidden_state
        # print("lm loss", lm_loss)
        # print("hidden_state_loss", hidden_state_losses)
        # print("kl_losses", kl_losses)



        # hidden state loss

        # KL penalty

        # Router logits only for the full router case
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                multi_forward_result[self.config.num_moe_experts].router_logits, # if return_dict else transformer_outputs[-1],
                self.config.num_moe_experts,
                self.config.expert_chosen,
                attention_mask,
            )
            # print(f"aux loss is {aux_loss}")

            if labels is not None:
                final_loss += self.moe_aux_loss_coeff * aux_loss.to(final_loss.device)  # make sure to reside in the same device
                # print(f"After add, loss is {loss}")


        if not return_dict:
            assert 1==0
            # output = (lm_logits,) + transformer_outputs[1:]
            # if output_router_logits:
            #     output = (aux_loss,) + output
            # return ((loss,) + output) if loss is not None else output

        # assert 1==0

        # print(f"forward return loss {loss}")


        return HWMoECausalLMOutputWithCrossAttentions(
            loss=final_loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            hidden_state_loss=hidden_state_losses,
            kl_loss=kl_losses
        )
