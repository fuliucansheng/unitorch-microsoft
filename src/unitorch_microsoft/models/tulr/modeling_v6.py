# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import SiLUActivation, gelu, gelu_new
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
)
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertOutput,
    BertPooler,
    BertSelfOutput,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaLMHead,
)
from unitorch.utils import pop_value, nested_dict_value
from unitorch.models import GenericModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, LossOutputs
from unitorch_microsoft import cached_path
from unitorch_microsoft.models.tulr import pretrained_tulr_infos


class TULRV6Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.TULRv5Model`.
    It is used to instantiate an TULRv5 model according to the specified arguments, defining the model
    architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        vocab_size (:obj:`int`, optional, defaults to 500002):
            Vocabulary size of the TULRv5 model. Defines the different tokens that
            can be represented by the `inputs_ids` passed to the forward method of :class:`~transformers.BertModel`.
        hidden_size (:obj:`int`, optional, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, optional, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, optional, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, optional, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, optional, defaults to "gelu"):
            The non-linear activation function (function or string) in the encoder and pooler.
            If string, "gelu", "relu", "swish" and "gelu_new" are supported.
        hidden_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, optional, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, optional, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, optional, defaults to 2):
            The vocabulary size of the `token_type_ids` passed into :class:`~transformers.TULRv5Model`.
        initializer_range (:obj:`float`, optional, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, optional, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        rel_pos_bins (:obj:`int`, optional, defaults to 32):
            No. of buckets used in relative position bias.
        max_rel_pos (:obj:`int`, optional, defaults to 128):
            Max relative position length supported.
    """
    model_type = "tulrv6"

    def __init__(
        self,
        vocab_size=500002,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        pad_token_id: Optional[int] = 1,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        expand_qk_dim=64,
        rel_pos_bins=32,
        max_rel_pos=128,
        use_key_bias: bool = False,
        ignore_gru_gate: bool = False,
        initialize_token_type_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.expand_qk_dim = expand_qk_dim
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.use_key_bias = use_key_bias
        self.ignore_gru_gate = ignore_gru_gate
        self.initialize_token_type_embeddings = initialize_token_type_embeddings


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "swish": SiLUActivation,
    "gelu_new": gelu_new,
    "mish": mish,
}

BertLayerNorm = torch.nn.LayerNorm


def relative_position_bucket(
    relative_position, bidirectional=True, num_buckets=32, max_distance=128
):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1)
    )

    ret += torch.where(is_small, n, val_if_large)
    return ret


class TULRV6SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.expand_qk_dim = config.expand_qk_dim

        self.query = nn.Linear(
            config.hidden_size, config.expand_qk_dim * self.num_attention_heads
        )
        if config.use_key_bias:
            self.key = nn.Linear(
                config.hidden_size,
                config.expand_qk_dim * self.num_attention_heads,
                bias=True,
            )
        else:
            self.key = nn.Linear(
                config.hidden_size,
                config.expand_qk_dim * self.num_attention_heads,
                bias=False,
            )
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.ignore_gru_gate = config.ignore_gru_gate
        if not self.ignore_gru_gate:
            self.gate_ur_linear = nn.Linear(config.expand_qk_dim, 8)
            self.eco_a = nn.Parameter(torch.ones(1, self.num_attention_heads, 1, 1))

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x, expand_qk=False):
        head_size = self.attention_head_size
        if expand_qk:
            head_size = self.expand_qk_dim
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        rel_pos=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, expand_qk=True)
        key_layer = self.transpose_for_scores(mixed_key_layer, expand_qk=True)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # logger.info('attention_scores {}'.format(torch.mean(attention_scores).cpu()))

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in TULRV6Model forward() function)
            attention_scores = attention_scores + attention_mask

        if rel_pos is not None and not self.ignore_gru_gate:
            # Applying Gated Relative Position Bias
            _B, _H, _L, __ = query_layer.size()
            # (B, H, L, D) -> (B, H, L, 1)
            gate_u, gate_r = torch.sigmoid(
                self.gate_ur_linear(query_layer)
                .view(_B, _H, _L, 2, 4)
                .sum(-1, keepdim=False)
            ).chunk(2, dim=-1)
            gate_u_1 = gate_u * (gate_r * self.eco_a - 1.0) + 2.0
            rel_pos_bias = gate_u_1 * rel_pos
            attention_scores = attention_scores + rel_pos_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs)
            if self.output_attentions
            else (context_layer,)
        )
        return outputs


class TULRV6Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = TULRV6SelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = (
            set(heads) - self.pruned_heads
        )  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        rel_pos=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            rel_pos=rel_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class TULRV6Layer(nn.Module):
    def __init__(self, config, layer_idx=-1):
        super().__init__()
        self.attention = TULRV6Attention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = TULRV6Attention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        rel_pos=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, rel_pos=rel_pos
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class TULRV6Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [TULRV6Layer(config, _) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        rel_pos=None,
        return_dict: Optional[bool] = False,
    ):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                rel_pos=rel_pos,
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        if not return_dict:
            return outputs  # last-layer hidden state, (all hidden states), (all attentions)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class TULRV6PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = TULRV6Config
    base_model_prefix = "tulrv6"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def create_position_ids_from_input_ids(
    input_ids, padding_idx, past_key_values_length=0
):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (
        torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
    ) * mask
    return incremental_indices.long() + padding_idx


class TULRV6Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        # This part taken from RobertaEmbeddings
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=self.padding_idx
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=self.padding_idx,
        )

        # This part taken from BertEmbeddings
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # This part is technically not necessary, but clearly shows the difference from BertEmbeddings
        self.token_type_embeddings = None

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        # This is taken from RobertaEmbedding
        """We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1,
            sequence_length + self.padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        # This is taken from RobertaEmbeddings
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds
                )
        # This part comes from BertEmbeddings
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        assert (
            token_type_ids is None
        ), "token_type_ids should be None for TULRV6Embeddings"

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TULRV6Model(TULRV6PreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        if self.config.initialize_token_type_embeddings:
            self.embeddings = RobertaEmbeddings(config)
        else:
            self.embeddings = TULRV6Embeddings(config)
        self.encoder = TULRV6Encoder(config)
        if add_pooling_layer:
            self.pooler = BertPooler(config)
        else:
            self.pooler = None

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(
                self.config.rel_pos_bins, config.num_attention_heads, bias=False
            )

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during pre-training.

                This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.

                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
                heads.

        Examples::

            from transformers import BertModel, BertTokenizer
            import torch

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids)

            last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = (
                    seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                    <= seq_ids[None, :, None]
                )
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = (
                    causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[
                    :, None, None, :
                ]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (
                1.0 - encoder_extended_attention_mask
            ) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1
                )
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if self.config.rel_pos_bins > 0:
            position_ids = torch.arange(input_ids.size(1), dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.size())
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos_mat = rel_pos_mat.type_as(input_ids)
            min_distance = rel_pos_mat.min()
            max_distance = rel_pos_mat.max()
            all_distance = torch.arange(
                start=min_distance,
                end=max_distance + 1,
                dtype=rel_pos_mat.dtype,
                device=rel_pos_mat.device,
            )
            rel_position_ids = relative_position_bucket(
                all_distance,
                num_buckets=self.config.rel_pos_bins,
                max_distance=self.config.max_rel_pos,
            )
            rel_pos_one_hot = F.one_hot(
                rel_position_ids, num_classes=self.config.rel_pos_bins
            ).type_as(embedding_output)
            rel_pos_embedding = torch.mm(
                self.rel_pos_bias.weight, rel_pos_one_hot.transpose(-1, -2)
            )
            rel_pos_embedding = rel_pos_embedding.unsqueeze(0).unsqueeze(-2)
            batch_size, seq_len, _ = rel_pos_mat.size()
            rel_pos_embedding = rel_pos_embedding.expand(batch_size, -1, seq_len, -1)
            num_attention_head = rel_pos_embedding.size()[1]
            rel_pos_mat = (
                (rel_pos_mat - min_distance)
                .unsqueeze(1)
                .expand(-1, num_attention_head, -1, -1)
            )
            rel_pos = torch.gather(rel_pos_embedding, dim=-1, index=rel_pos_mat)
        else:
            rel_pos = None

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            rel_pos=rel_pos,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )
        if not return_dict:
            outputs = (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[
                1:
            ]  # add hidden_states and attentions if they are here
            return (
                outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
            )
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


@register_model("microsoft/model/classification/tulr/v6")
class TULRV6ForClassification(GenericModel):
    def __init__(
        self,
        config_path: str,
        num_classes: Optional[int] = 1,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.tulrv6 = TULRV6Model(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/classification/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/classification/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "default-tulrv6")
        config_path = config.getoption("config_path", None)
        num_classes = config.getoption("num_classes", 1)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(config_path, num_classes, gradient_checkpointing)

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.tulrv6(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return ClassificationOutputs(outputs=logits)


@register_model("microsoft/model/pretrain/tulr/v6")
class TULRV6ForPretrain(GenericModel):
    def __init__(
        self,
        config_path: str,
        use_nsp_loss: Optional[bool] = True,
        tied_word_embeddings: Optional[bool] = True,
        gradient_checkpointing: Optional[bool] = False,
    ):
        super().__init__()
        self.config = TULRV6Config.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.tulrv6 = TULRV6Model(self.config, add_pooling_layer=use_nsp_loss)
        self.lm_head = RobertaLMHead(self.config)
        if tied_word_embeddings:
            self.lm_head.decoder.weight = self.tulrv6.embeddings.word_embeddings.weight
        self.classifier = (
            nn.Linear(self.config.hidden_size, 2) if use_nsp_loss else None
        )
        self.mlm_loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.nsp_loss_fn = (
            nn.CrossEntropyLoss(reduction="none") if use_nsp_loss else None
        )
        self.init_weights()

    @classmethod
    @add_default_section_for_init("microsoft/model/pretrain/tulr/v6")
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section("microsoft/model/pretrain/tulr/v6")
        pretrained_name = config.getoption("pretrained_name", "default-tulrv6")
        config_path = config.getoption("config_path", None)

        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)

        use_nsp_loss = config.getoption("use_nsp_loss", True)
        tied_word_embeddings = config.getoption("tied_word_embeddings", True)
        gradient_checkpointing = config.getoption("gradient_checkpointing", False)

        inst = cls(
            config_path,
            use_nsp_loss=use_nsp_loss,
            tied_word_embeddings=tied_word_embeddings,
            gradient_checkpointing=gradient_checkpointing,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_tulr_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)

        return inst

    @autocast()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mlm_label: torch.Tensor,
        mlm_label_mask: torch.Tensor,
        nsp_label: Optional[torch.Tensor] = None,
    ):
        outputs = self.tulrv6(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        mlm_logits = self.lm_head(outputs[0])
        batch_size, seq_len, vocab_size = mlm_logits.size()
        masked_lm_loss = self.mlm_loss_fn(
            mlm_logits.view(-1, vocab_size), mlm_label.view(-1)
        ) * mlm_label_mask.view(-1)
        masked_lm_loss = masked_lm_loss.view(batch_size, seq_len).sum(1) / torch.max(
            mlm_label_mask.view(batch_size, seq_len).sum(1),
            torch.ones(batch_size).to(mlm_label_mask.device),
        )
        loss = masked_lm_loss.mean()

        if self.nsp_loss_fn is not None and nsp_label is not None:
            nsp_logits = self.classifier(outputs[1])
            loss += self.nsp_loss_fn(nsp_logits.view(-1, 2), nsp_label.view(-1)).mean()

        return LossOutputs(loss=loss)
