# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import json
import transformers
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.nn import functional as F
from torch.cuda.amp import autocast
from transformers.utils import is_remote_url
from transformers import BloomModel, BloomConfig
from transformers.models.bloom.modeling_bloom import (
    dropout_add,
    BaseModelOutputWithPastAndCrossAttentions,
    LayerNorm,
    BloomMLP,
    PreTrainedModel,
    _prepare_4d_causal_attention_mask,
)


class BloomAttention(transformers.models.bloom.modeling_bloom.BloomAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        fused_qkv = self.query_key_value(
            hidden_states
        )  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim
        )
        key_layer = key_layer.permute(0, 2, 3, 1).reshape(
            batch_size * self.num_heads, self.head_dim, q_length
        )
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size * self.num_heads, q_length, self.head_dim
        )
        prefix_key_layer, prefix_value_layer = None, None
        if layer_past is not None:
            past_key, past_value, prefix_key_layer, prefix_value_layer = layer_past
            # concatenate along seq_length dimension:
            #  - key: [batch_size * self.num_heads, head_dim, kv_length]
            #  - value: [batch_size * self.num_heads, kv_length, head_dim]
            if past_key is not None:
                key_layer = torch.cat((past_key, key_layer), dim=2)

            if past_value is not None:
                value_layer = torch.cat((past_value, value_layer), dim=1)

        _, _, kv_length = key_layer.shape

        if use_cache is True:
            present = [key_layer, value_layer, prefix_key_layer, prefix_value_layer]
        else:
            present = None

        # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11

        prefix_matmul_result, prefix_kv_length = None, None
        if prefix_key_layer is not None:
            prefix_batch_size, _, prefix_kv_length = prefix_key_layer.shape
            prefix_batch_size = prefix_batch_size // self.num_heads

            prefix_matmul_result = torch.einsum(
                "bxhtd,bhds->bxhts",
                query_layer.reshape(
                    prefix_batch_size, -1, self.num_heads, q_length, self.head_dim
                ),
                prefix_key_layer.reshape(
                    prefix_batch_size, self.num_heads, self.head_dim, prefix_kv_length
                ),
            )
            prefix_matmul_result = prefix_matmul_result.reshape(
                -1, q_length, prefix_kv_length
            )

        matmul_result = torch.matmul(query_layer, key_layer)

        if prefix_matmul_result is not None:
            matmul_result = torch.cat((prefix_matmul_result, matmul_result), dim=-1)

        matmul_result = alibi * self.beta + matmul_result * self.inv_norm_factor

        # matmul_result = alibi.baddbmm(
        #     batch1=query_layer,
        #     batch2=key_layer,
        #     beta=self.beta,
        #     alpha=self.inv_norm_factor,
        # )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, -1)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16:
            attention_scores = attention_scores.to(torch.float)
        attn_weights = torch.masked_fill(
            attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min
        )
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            input_dtype
        )

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads, q_length, -1
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        if prefix_value_layer is not None:
            prefix_attention_probs = attention_probs_reshaped[:, :, :prefix_kv_length]
            attention_probs_reshaped = attention_probs_reshaped[:, :, prefix_kv_length:]
            prefix_context_layer = torch.einsum(
                "bxhtd,bhds->bxhts",
                prefix_attention_probs.view(
                    prefix_batch_size, -1, self.num_heads, q_length, prefix_kv_length
                ),
                prefix_value_layer.view(
                    prefix_batch_size, self.num_heads, prefix_kv_length, self.head_dim
                ),
            ).reshape(-1, q_length, self.head_dim)

        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        if prefix_value_layer is not None:
            context_layer = prefix_context_layer + context_layer

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(
            output_tensor, residual, self.hidden_dropout, self.training
        )

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomBlock(transformers.models.bloom.modeling_bloom.BloomBlock):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttention(config)
        self.post_attention_layernorm = LayerNorm(
            hidden_size, eps=config.layer_norm_epsilon
        )

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.hidden_dropout = config.hidden_dropout


class BloomModel(transformers.models.bloom.modeling_bloom.BloomModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(
            self.embed_dim, eps=config.layer_norm_epsilon
        )

        # Transformer blocks
        self.h = nn.ModuleList(
            [BloomBlock(config) for _ in range(config.num_hidden_layers)]
        )

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        past_key, _, past_prefix_key, _ = past_key_value[0]
        if past_prefix_key is not None:
            (
                batch_size_times_num_heads,
                head_dim,
                prefix_seq_length,
            ) = past_prefix_key.shape
        if past_key is not None:
            batch_size_times_num_heads, head_dim, seq_length = past_key.shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return [
            [
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length)
                if layer_past[0] is not None
                else None,
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim)
                if layer_past[1] is not None
                else None,
                layer_past[2].view(-1, num_heads, head_dim, prefix_seq_length)
                if layer_past[2] is not None
                else None,
                layer_past[3].view(-1, num_heads, prefix_seq_length, head_dim)
                if layer_past[3] is not None
                else None,
            ]
            for layer_past in past_key_value
        ]

    @staticmethod
    def _convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        past_key, _, past_prefix_key, _ = past_key_value[0]
        if past_prefix_key is not None:
            batch_size, num_heads, head_dim, prefix_seq_length = past_prefix_key.shape
        if past_key is not None:
            batch_size, num_heads, head_dim, seq_length = past_key.shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return [
            [
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length)
                if layer_past[0] is not None
                else None,
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim)
                if layer_past[1] is not None
                else None,
                layer_past[2].view(-1, head_dim, prefix_seq_length)
                if layer_past[2] is not None
                else None,
                layer_past[3].view(-1, prefix_seq_length, head_dim)
                if layer_past[3] is not None
                else None,
            ]
            for layer_past in past_key_value
        ]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            if past_key_values[0][0] is not None:
                past_key_values_length += past_key_values[0][0].shape[2]
            if past_key_values[0][2] is not None:
                past_key_values_length += past_key_values[0][2].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), device=hidden_states.device
            )
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(
            attention_mask, self.num_heads, dtype=hidden_states.dtype
        )

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        causal_mask = causal_mask.bool()

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(
                            *inputs,
                            use_cache=use_cache,
                            output_attentions=output_attentions,
                        )

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    layer_past,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class BloomForCausalLM(transformers.models.bloom.modeling_bloom.BloomForCausalLM):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _convert_to_standard_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]], batch_size: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standardizes the format of the cache so as to match most implementations, i.e. to tuple(tuple([batch_size,
        num_heads, ...]))
        """
        past_key, _, past_prefix_key, _ = past_key_value[0]
        if past_prefix_key is not None:
            (
                batch_size_times_num_heads,
                head_dim,
                prefix_seq_length,
            ) = past_prefix_key.shape
        if past_key is not None:
            batch_size_times_num_heads, head_dim, seq_length = past_key.shape
        num_heads = batch_size_times_num_heads // batch_size
        # key: [batch_size * num_heads, head_dim, seq_length] -> [batch_size, num_heads, head_dim, seq_length]
        # value: [batch_size * num_heads, seq_length, head_dim] -> [batch_size, num_heads, seq_length, head_dim]
        return [
            [
                layer_past[0].view(batch_size, num_heads, head_dim, seq_length)
                if layer_past[0] is not None
                else None,
                layer_past[1].view(batch_size, num_heads, seq_length, head_dim)
                if layer_past[1] is not None
                else None,
                layer_past[2].view(-1, num_heads, head_dim, prefix_seq_length)
                if layer_past[2] is not None
                else None,
                layer_past[3].view(-1, num_heads, prefix_seq_length, head_dim)
                if layer_past[3] is not None
                else None,
            ]
            for layer_past in past_key_value
        ]

    @staticmethod
    def _convert_to_bloom_cache(
        past_key_value: Tuple[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the cache to the format expected by Bloom, i.e. to tuple(tuple([batch_size * num_heads, ...]))
        """
        past_key, _, past_prefix_key, _ = past_key_value[0]
        if past_prefix_key is not None:
            batch_size, num_heads, head_dim, prefix_seq_length = past_prefix_key.shape
        if past_key is not None:
            batch_size, num_heads, head_dim, seq_length = past_key.shape
        batch_size_times_num_heads = batch_size * num_heads
        # key:  [batch_size, num_heads, head_dim, seq_length] -> [batch_size * num_heads, head_dim, seq_length]
        # value: [batch_size, num_heads, seq_length, head_dim] -> [batch_size * num_heads, seq_length, head_dim]
        return [
            [
                layer_past[0].view(batch_size_times_num_heads, head_dim, seq_length)
                if layer_past[0] is not None
                else None,
                layer_past[1].view(batch_size_times_num_heads, seq_length, head_dim)
                if layer_past[1] is not None
                else None,
                layer_past[2].view(-1, head_dim, prefix_seq_length)
                if layer_past[2] is not None
                else None,
                layer_past[3].view(-1, prefix_seq_length, head_dim)
                if layer_past[3] is not None
                else None,
            ]
            for layer_past in past_key_value
        ]

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # only last token for input_ids if past is not None
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            if past_key_values[0][0].shape[0] == input_ids.shape[0]:
                past_key_values = self._convert_to_bloom_cache(past_key_values)

            past_key, _, past_prefix_key, _ = past_key_values[0]
            if past_prefix_key is None:
                num_beams = self.num_beams
                past_key_values = [
                    (
                        None,
                        None,
                        layer_past[0]
                        .view(
                            -1, num_beams, self.config.n_head, *layer_past[0].shape[1:]
                        )[:, 0]
                        .squeeze(1)
                        .reshape(-1, *layer_past[0].shape[1:]),
                        layer_past[1]
                        .view(
                            -1, num_beams, self.config.n_head, *layer_past[1].shape[1:]
                        )[:, 0]
                        .squeeze(1)
                        .reshape(-1, *layer_past[1].shape[1:]),
                    )
                    for layer_past in past_key_values
                ]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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

    def _reorder_cache(
        self,
        past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        beam_idx: torch.LongTensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(
            past, batch_size=len(beam_idx)
        )

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device)
            for layer_past in past
            for past_state in layer_past
            if past_state is not None
        }
        reordered_past = [
            [
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device])
                if layer_past[0] is not None
                else None,
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device])
                if layer_past[1] is not None
                else None,
                layer_past[2],
                layer_past[3],
            ]
            for layer_past in standardized_past
        ]
        return self._convert_to_bloom_cache(reordered_past)
