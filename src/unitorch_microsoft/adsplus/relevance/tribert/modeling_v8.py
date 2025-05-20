# Copyright (c) MICROSOFT.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertPreTrainedModel,
)
from unitorch.models import GenericModel
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs
from unitorch_microsoft import cached_path


class TriEmbeddings(nn.Module):
    """Comparing with TriletterEmbeddingsSimple, this one has position encoding, name is a little bit ugly, but not breaking anything"""

    def __init__(self, config):
        super().__init__()
        self.max_n_letters = config.max_n_letters  # 20, so 20 triletters
        self.tri_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0,
        )  # triletter embedding
        self.pos_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            padding_idx=0,
        )

    def forward(
        self,
        input_ids,
        position_ids=None,
        **kwargs,
    ):
        max_seq_length = input_ids.shape[1] // self.max_n_letters
        if position_ids is None:
            position_ids = torch.arange(
                max_seq_length,
                device=input_ids.device,
            )
        pos_embeddings = self.pos_embeddings(position_ids)
        embeddings = self.tri_embeddings(input_ids)  # [N, 12*[20], hidden_size]
        embeddings = embeddings.view(
            -1,
            max_seq_length,
            self.max_n_letters,
            embeddings.shape[-1],
        )  # [N, 12, 20, hidden_size]
        embeddings = embeddings.sum(dim=2).view(
            -1,
            max_seq_length,
            embeddings.shape[-1],
        )
        embeddings = embeddings + pos_embeddings
        return embeddings


class TriTwinBertModel(BertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=False):
        super().__init__(config)

        self.embeddings = TriEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
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
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape,
                    device=device,
                )

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
            input_ids=input_ids, position_ids=position_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        outputs = (
            sequence_output,
            pooled_output,
        ) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class TriTwinBertEncoder(nn.Module):
    def __init__(
        self,
        config,
        max_n_letters=20,
        add_pooling_layer=False,
    ):
        super().__init__()
        setattr(config, "max_n_letters", max_n_letters)
        config.output_attentions = True
        config.output_hidden_states = True
        self.bert = TriTwinBertModel(config, add_pooling_layer=add_pooling_layer)
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(
        self,
        token_ids,
        token_mask,
        pos_ids=None,
    ):
        outputs = self.bert(
            token_ids,
            token_mask,
            pos_ids,
        )
        return outputs


class TriPostLayer(nn.Module):
    def __init__(
        self,
        num_classes=1,
        sim4score="cosine",
        pooltype="bert",
        hidden_size=512,
        hidden_downscale_size=128,
        reslayer_use_bn=True,
        reslayer_downscale_size=128,
        reslayer_hidden_size=64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sim4score = sim4score
        self.pooltype = pooltype
        self.hidden_size = hidden_size

        if self.sim4score in ["reslayer"]:
            self.res_bn1 = (
                nn.BatchNorm1d(reslayer_hidden_size) if reslayer_use_bn else None
            )
            self.res_bn2 = (
                nn.BatchNorm1d(reslayer_downscale_size * 3) if reslayer_use_bn else None
            )
            self.res_fc0 = nn.Linear(self.hidden_size, reslayer_downscale_size)
            self.res_fc1 = nn.Linear(reslayer_downscale_size * 3, reslayer_hidden_size)
            self.res_fc1.weight.data.normal_(mean=0.0, std=0.02)
            self.res_fc1.bias.data.zero_()

            self.res_fc2 = nn.Linear(reslayer_hidden_size, reslayer_downscale_size * 3)
            self.res_fc2.weight.data.normal_(mean=0.0, std=0.02)
            self.res_fc2.bias.data.zero_()

            self.res_fc3 = nn.Linear(reslayer_downscale_size * 3, reslayer_hidden_size)
            self.res_fc3.weight.data.normal_(mean=0.0, std=0.02)
            self.res_fc3.bias.data.zero_()

            self.res_fc4 = nn.Linear(reslayer_hidden_size, reslayer_downscale_size * 3)
            self.res_fc4.weight.data.normal_(mean=0.0, std=0.02)
            self.res_fc4.bias.data.zero_()

            self.fc = nn.Linear(reslayer_downscale_size * 3, self.num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.02)
            self.fc.bias.data.zero_()
        elif self.sim4score in ["maxfeedforward", "sumfeedforward", "avgfeedforward"]:
            self.max_fc0 = nn.Linear(self.hidden_size, self.hidden_size)
            self.max_fc1 = nn.Linear(self.hidden_size, self.num_classes)
            self.max_fc0.weight.data.normal_(mean=0.0, std=0.02)
            self.max_fc0.bias.data.zero_()
            self.max_fc1.weight.data.normal_(mean=0.0, std=0.02)
            self.max_fc1.bias.data.zero_()
        else:
            self.fc = nn.Linear(self.hidden_size * 3, self.num_classes)
            self.fc.weight.data.normal_(mean=0.0, std=0.02)
            self.fc.bias.data.zero_()

    def _quantization(self, tensor):
        tensor = torch.round(
            (tensor + 1) / (2 / 255)
        )  # 2/256, it is 2/255 in production
        tensor = tensor * (2 / 255) - 1
        return tensor

    def res_layer(self, res_fc1, res_fc2, x):
        identity = x
        out = res_fc1(x)
        if self.res_bn1 is not None:
            out = self.res_bn1(out)
        out = res_fc2(torch.relu(out))
        if self.res_bn2 is not None:
            out = self.res_bn2(out)
        out += identity
        return torch.relu(out)

    def forward(
        self,
        q_out: torch.Tensor = None,
        d_out: torch.Tensor = None,
        ads_out: torch.Tensor = None,
    ):
        if self.sim4score in ["reslayer"]:
            q_out = self.res_fc0(torch.relu(q_out))
            d_out = self.res_fc0(torch.relu(d_out))
            ads_out = self.res_fc0(torch.relu(ads_out))
            output = torch.cat([q_out, d_out, ads_out], dim=-1)
            output = self.res_layer(self.res_fc1, self.res_fc2, output)
            output = self.res_layer(self.res_fc3, self.res_fc4, output)
            output = self.fc(output)
        elif self.sim4score in ["maxfeedforward"]:
            output = torch.max(torch.max(q_out, d_out), ads_out)
            output = self.max_fc1(torch.relu(self.max_fc0(output)) + output)
        elif self.sim4score in ["sumfeedforward"]:
            output = torch.sum((q_out, d_out, ads_out), dim=-1, keepdim=True)
            output = self.max_fc1(torch.relu(self.max_fc0(output)) + output)
        elif self.sim4score in ["avgfeedforward"]:
            output = (q_out + d_out + ads_out) / 3
            output = self.max_fc1(torch.relu(self.max_fc0(output)) + output)
        else:
            output = torch.cat([q_out, d_out, ads_out], dim=-1)
            output = self.fc(output)
        return output


class MultiTriPostLayer(nn.Module):
    def __init__(
        self,
        num_tasks: int = 1,
        num_classes: int = 1,
        sim4score: str = "cosine",
        pooltype: str = "bert",
        hidden_size: int = 512,
        hidden_downscale_size: int = 128,
        reslayer_use_bn: bool = True,
        reslayer_downscale_size: int = 128,
        reslayer_hidden_size: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sim4score = sim4score
        self.pooltype = pooltype
        self.hidden_size = hidden_size
        self.hidden_downscale_size = hidden_downscale_size
        self.reslayer_use_bn = reslayer_use_bn
        self.reslayer_downscale_size = reslayer_downscale_size
        self.reslayer_hidden_size = reslayer_hidden_size

        self.fc1 = nn.Linear(hidden_downscale_size, self.hidden_size)
        self.num_tasks = num_tasks
        self.mtfc = nn.ModuleDict(
            {
                str(k): TriPostLayer(
                    num_classes=self.num_classes,
                    sim4score=self.sim4score,
                    pooltype=self.pooltype,
                    hidden_size=self.hidden_size,
                    hidden_downscale_size=self.hidden_downscale_size,
                    reslayer_use_bn=self.reslayer_use_bn,
                    reslayer_downscale_size=self.reslayer_downscale_size,
                    reslayer_hidden_size=self.reslayer_hidden_size,
                )
                for k in range(self.num_tasks)
            }
        )

    def forward(
        self,
        task: torch.Tensor = None,
        q_out: torch.Tensor = None,
        d_out: torch.Tensor = None,
        ads_out: torch.Tensor = None,
    ):
        q_out = self.fc1(q_out)
        d_out = self.fc1(d_out)
        ads_out = self.fc1(ads_out)
        out = torch.empty(q_out.shape[0], self.num_classes)
        out = torch.zeros_like(out).to(q_out).float()
        for i in range(self.num_tasks):
            out += self.mtfc[str(i)](q_out, d_out, ads_out) * (
                task == i
            ).float().unsqueeze(-1)
        return out


@register_model("microsoft/adsplus/relevance/classification/tribert/v8")
class TribertForClassification(GenericModel):
    prefix_keys_in_state_dict = {}

    def __init__(
        self,
        config_path,
        num_tasks: int = 1,
        num_classes: int = 1,
        sim4score: str = "cosine",
        pooltype: str = "bert",
        hidden_size: int = 512,
        max_n_letters: int = 20,
        hidden_downscale_size: int = 128,
        reslayer_use_bn: bool = True,
        reslayer_downscale_size: int = 128,
        reslayer_hidden_size: int = 64,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = False
        self.config.hidden_size = hidden_size
        self.query_encoder = TriTwinBertEncoder(
            self.config,
            max_n_letters=max_n_letters,
            add_pooling_layer=pooltype == "bert",
        )

        self.doc_encoder = TriTwinBertEncoder(
            self.config,
            max_n_letters=max_n_letters,
            add_pooling_layer=pooltype == "bert",
        )

        self.ads_encoder = TriTwinBertEncoder(
            self.config,
            max_n_letters=max_n_letters,
        )

        self.postlayer = MultiTriPostLayer(
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_size=self.config.hidden_size,
            hidden_downscale_size=hidden_downscale_size,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
        )

        self.query_encoder.bert.init_weights()
        self.doc_encoder.bert.init_weights()
        self.ads_encoder.bert.init_weights()

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("encoder.bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_" + _key] = _value
            state_dict["doc_" + _key] = _value
            state_dict["ads_" + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/relevance/classification/tribert/v8"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/relevance/classification/tribert/v8"
        )
        config_path = config.getoption(
            "config_path",
            "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/relevance/tribert/config.json",
        )
        config_path = cached_path(config_path)
        num_tasks = config.getoption("num_tasks", 1)
        num_classes = config.getoption("num_classes", 1)
        sim4score = config.getoption("sim4score", "cosine")
        pooltype = config.getoption("pooltype", "bert")
        hidden_size = config.getoption("hidden_size", 512)
        hidden_downscale_size = config.getoption("hidden_downscale_size", 128)
        max_n_letters = config.getoption("max_n_letters", 20)
        reslayer_use_bn = config.getoption("reslayer_use_bn", True)
        reslayer_downscale_size = config.getoption("reslayer_downscale_size", 128)
        reslayer_hidden_size = config.getoption("reslayer_hidden_size", 64)
        inst = cls(
            config_path=config_path,
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_size=hidden_size,
            hidden_downscale_size=hidden_downscale_size,
            max_n_letters=max_n_letters,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
        )

        weight_path = config.getoption(
            "pretrained_weight_path",
            "https://huggingface.co/datasets/fuliucansheng/unitorchblobfuse/resolve/main/models/adsplus/relevance/tribert/pytorch_model.v2.bin",
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)
        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        task,
        query_input_ids,
        query_attention_mask,
        ads_input_ids,
        ads_attention_mask,
        doc_input_ids,
        doc_attention_mask,
    ):
        query_outputs = self.query_encoder(query_input_ids, query_attention_mask)
        doc_outputs = self.doc_encoder(doc_input_ids, doc_attention_mask)
        ads_outputs = self.ads_encoder(ads_input_ids, ads_attention_mask)
        outputs = self.postlayer(
            task=task,
            qseq_in=query_outputs[0],
            qpool_in=query_outputs[1],
            q_mask=query_attention_mask,
            dseq_in=doc_outputs[0],
            dpool_in=doc_outputs[1],
            d_mask=doc_attention_mask,
            adsseq_in=ads_outputs[0],
            adspool_in=ads_outputs[1],
            ads_mask=ads_attention_mask,
        )
        return ClassificationOutputs(outputs=outputs)
