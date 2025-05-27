# Copyright (c) MICROSOFT.
# Licensed under the MIT License.
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPooler,
    BertModel,
    BertPreTrainedModel,
)
from unitorch_microsoft.adsplus.relevance.tribert.modeling_v8 import MultiTriPostLayer
from unitorch.models import GenericModel
from unitorch.utils import pop_value, nested_dict_value
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
    register_model,
)
from unitorch.cli.models import ClassificationOutputs, EmbeddingOutputs
from unitorch.cli.models.bert import pretrained_bert_infos
from unitorch_microsoft import cached_path


class TriTwinBertEncoder(nn.Module):
    def __init__(self, config, add_pooling_layer):
        super().__init__()
        self.bert = BertModel(
            config,
            add_pooling_layer=add_pooling_layer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        return outputs

    def from_pretrained(self, weight_path, prefix):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict[prefix + _key] = _value


@register_model("microsoft/adsplus/relevance/classification/tribert/v9")
class TribertForClassification(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}
    prefix_keys_in_state_dict = {}

    def __init__(
        self,
        config_path,
        num_tasks: int = 1,
        num_classes: int = 1,
        sim4score: str = "cosine",
        pooltype: str = "bert",
        hidden_downscale_size: int = 128,
        reslayer_use_bn: bool = True,
        reslayer_downscale_size: int = 128,
        reslayer_hidden_size: int = 64,
        output: str = "query",  # query, doc, ads
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = False
        self.pooltype = pooltype
        self.hidden_size = self.config.hidden_size
        self.query_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.doc_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.ads_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        if self.pooltype in ["weight"]:
            self.attn = nn.Linear(self.hidden_size, 1, bias=False)

        self.fc0 = nn.Linear(self.hidden_size, hidden_downscale_size)
        self.output = output

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

        self.init_weights()

    def from_pretrained(self, weight_path):
        weight_path = cached_path(weight_path)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder." + _key] = _value
            state_dict["doc_encoder." + _key] = _value
            state_dict["ads_encoder." + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/relevance/classification/tribert/v9"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/relevance/classification/tribert/v9"
        )
        pretrained_name = config.getoption("pretrained_name", "bert-base-uncased")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        num_tasks = config.getoption("num_tasks", 1)
        num_classes = config.getoption("num_classes", 1)
        sim4score = config.getoption("sim4score", "cosine")
        pooltype = config.getoption("pooltype", "bert")
        hidden_downscale_size = config.getoption("hidden_downscale_size", 128)
        reslayer_use_bn = config.getoption("reslayer_use_bn", True)
        reslayer_downscale_size = config.getoption("reslayer_downscale_size", 128)
        reslayer_hidden_size = config.getoption("reslayer_hidden_size", 64)
        output = config.getoption("output", None)
        inst = cls(
            config_path=config_path,
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_downscale_size=hidden_downscale_size,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
            output=output,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            weight_path = cached_path(weight_path)
            inst.from_pretrained(weight_path)
        return inst

    def _attn(self, d_in, d_mask):
        att = self.attn(d_in).squeeze(dim=-1)
        d_mask = d_mask.to(att)
        att = att + (1 - d_mask) * -10000
        att = F.softmax(att, dim=-1)
        return torch.bmm(d_in.transpose(1, 2), att.unsqueeze(dim=-1)).squeeze(dim=-1)

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        task,
        query_input_ids,
        query_attention_mask,
        query_token_type_ids,
        query_position_ids,
        doc_input_ids,
        doc_attention_mask,
        doc_token_type_ids,
        doc_position_ids,
        ads_input_ids,
        ads_attention_mask,
        ads_token_type_ids,
        ads_position_ids,
    ):
        query_outputs = self.query_encoder(
            query_input_ids,
            query_attention_mask,
            query_token_type_ids,
            query_position_ids,
        )
        doc_outputs = self.doc_encoder(
            doc_input_ids, doc_attention_mask, doc_token_type_ids, doc_position_ids
        )
        ads_outputs = self.ads_encoder(
            ads_input_ids, ads_attention_mask, ads_token_type_ids, ads_position_ids
        )

        if self.pooltype in ["weight"]:
            q_in = self._attn(query_outputs.last_hidden_state, query_attention_mask)
            d_in = self._attn(doc_outputs.last_hidden_state, doc_attention_mask)
            ads_in = self._attn(ads_outputs.last_hidden_state, ads_attention_mask)
        else:
            q_in = query_outputs.pooler_output
            d_in = doc_outputs.pooler_output
            ads_in = ads_outputs.pooler_output

        if self.output == "query":
            q_out = torch.tanh(self.fc0(q_in))
            return EmbeddingOutputs(embedding=q_out)
        elif self.output == "doc":
            d_out = torch.tanh(self.fc0(d_in))
            return EmbeddingOutputs(embedding=d_out)
        elif self.output == "ads":
            ads_out = torch.tanh(self.fc0(ads_in))
            return EmbeddingOutputs(embedding=ads_out)
        elif self.output == "all":
            q_out = torch.tanh(self.fc0(q_in))
            d_out = torch.tanh(self.fc0(d_in))
            ads_out = torch.tanh(self.fc0(ads_in))
            return EmbeddingOutputs(
                embedding=q_out, embedding1=d_out, embedding2=ads_out
            )

        q_out = torch.tanh(self.fc0(q_in))
        d_out = torch.tanh(self.fc0(d_in))
        ads_out = torch.tanh(self.fc0(ads_in))
        outputs = self.postlayer(
            task=task,
            q_out=q_out,
            d_out=d_out,
            ads_out=ads_out,
        )
        return ClassificationOutputs(outputs=outputs)


@register_model(
    "microsoft/adsplus/relevance/classification/tribert/v9/diffOnlineOffline"
)
class TribertForClassification_V2(GenericModel):
    replace_keys_in_state_dict = {"gamma": "weight", "beta": "bias"}
    prefix_keys_in_state_dict = {}

    def __init__(
        self,
        config_path,
        offline_config_path,
        online_model_freeze: bool = True,
        num_tasks: int = 1,
        num_classes: int = 1,
        sim4score: str = "cosine",
        pooltype: str = "bert",
        hidden_downscale_size: int = 128,
        reslayer_use_bn: bool = True,
        reslayer_downscale_size: int = 128,
        reslayer_hidden_size: int = 64,
    ):
        super().__init__()
        self.config = BertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = False

        self.offline_config = BertConfig.from_json_file(offline_config_path)
        self.offline_config.gradient_checkpointing = False

        self.query_encoder = TriTwinBertEncoder(
            self.config,
            add_pooling_layer=pooltype == "bert",
        )

        self.doc_encoder = TriTwinBertEncoder(
            self.offline_config,
            add_pooling_layer=pooltype == "bert",
        )

        self.ads_encoder = TriTwinBertEncoder(
            self.offline_config,
            add_pooling_layer=pooltype == "bert",
        )

        self.postlayer = MultiTriPostLayerV2(
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_size=self.config.hidden_size,
            offline_hidden_size=self.offline_config.hidden_size,
            hidden_downscale_size=hidden_downscale_size,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
        )

        self.init_weights()
        if online_model_freeze:
            for p in self.query_encoder.parameters():
                p.requires_grad = False

    def from_pretrained(self, weight_path):
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        _keys = [key for key in state_dict.keys() if key.startswith("bert")]
        for _key in _keys:
            _value = state_dict.pop(_key)
            state_dict["query_encoder." + _key] = _value
            state_dict["doc_encoder." + _key] = _value
            state_dict["ads_encoder." + _key] = _value

        super().from_pretrained(state_dict=state_dict)

    @classmethod
    @add_default_section_for_init(
        "microsoft/adsplus/relevance/classification/tribert/v9/diffOnlineOffline"
    )
    def from_core_configure(cls, config, **kwargs):
        config.set_default_section(
            "microsoft/adsplus/relevance/classification/tribert/v9/diffOnlineOffline"
        )
        pretrained_name = config.getoption("pretrained_name", "bert-base-uncased")
        config_path = config.getoption("config_path", None)
        config_path = pop_value(
            config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        config_path = cached_path(config_path)
        online_model_freeze = config.getoption("online_model_freeze", True)

        offline_config_path = config.getoption("offline_config_path", None)
        offline_config_path = pop_value(
            offline_config_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "config"),
        )
        offline_config_path = cached_path(offline_config_path)
        offline_pretrained_weight = config.getoption("offline_pretrained_weight", None)

        num_tasks = config.getoption("num_tasks", 1)
        num_classes = config.getoption("num_classes", 1)
        sim4score = config.getoption("sim4score", "cosine")
        pooltype = config.getoption("pooltype", "bert")
        hidden_downscale_size = config.getoption("hidden_downscale_size", 128)
        reslayer_use_bn = config.getoption("reslayer_use_bn", True)
        reslayer_downscale_size = config.getoption("reslayer_downscale_size", 128)
        reslayer_hidden_size = config.getoption("reslayer_hidden_size", 64)
        inst = cls(
            config_path=config_path,
            offline_config_path=offline_config_path,
            online_model_freeze=online_model_freeze,
            num_tasks=num_tasks,
            num_classes=num_classes,
            sim4score=sim4score,
            pooltype=pooltype,
            hidden_downscale_size=hidden_downscale_size,
            reslayer_use_bn=reslayer_use_bn,
            reslayer_downscale_size=reslayer_downscale_size,
            reslayer_hidden_size=reslayer_hidden_size,
        )

        weight_path = config.getoption("pretrained_weight_path", None)
        weight_path = pop_value(
            weight_path,
            nested_dict_value(pretrained_bert_infos, pretrained_name, "weight"),
            check_none=False,
        )
        if weight_path is not None:
            inst.from_pretrained(weight_path)
        if offline_pretrained_weight is not None:
            offline_pretrained_weight = cached_path(offline_pretrained_weight)
            inst.doc_encoder.from_pretrained(offline_pretrained_weight, "doc_encoder.")
            inst.ads_encoder.from_pretrained(offline_pretrained_weight, "ads_encoder.")

        return inst

    @autocast(device_type=("cuda" if torch.cuda.is_available() else "cpu"))
    def forward(
        self,
        task,
        query_input_ids,
        query_attention_mask,
        query_token_type_ids,
        query_position_ids,
        doc_input_ids,
        doc_attention_mask,
        doc_token_type_ids,
        doc_position_ids,
        ads_input_ids,
        ads_attention_mask,
        ads_token_type_ids,
        ads_position_ids,
    ):
        query_outputs = self.query_encoder(
            query_input_ids,
            query_attention_mask,
            query_token_type_ids,
            query_position_ids,
        )
        doc_outputs = self.doc_encoder(
            doc_input_ids, doc_attention_mask, doc_token_type_ids, doc_position_ids
        )
        ads_outputs = self.ads_encoder(
            ads_input_ids, ads_attention_mask, ads_token_type_ids, ads_position_ids
        )
        outputs = self.postlayer(
            task=task,
            qseq_in=query_outputs.last_hidden_state,
            qpool_in=query_outputs.pooler_output,
            q_mask=query_attention_mask,
            dseq_in=doc_outputs.last_hidden_state,
            dpool_in=doc_outputs.pooler_output,
            d_mask=doc_attention_mask,
            adsseq_in=ads_outputs.last_hidden_state,
            adspool_in=ads_outputs.pooler_output,
            ads_mask=ads_attention_mask,
        )
        return ClassificationOutputs(outputs=outputs)


class TriPostLayerV2(nn.Module):
    def __init__(
        self,
        num_classes=1,
        sim4score="cosine",
        pooltype="bert",
        hidden_size=512,
        offline_hidden_size=512,
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
        self.offline_hidden_size = offline_hidden_size
        self.fc0 = nn.Linear(self.hidden_size, hidden_downscale_size)
        self.fc0_offline = nn.Linear(self.offline_hidden_size, hidden_downscale_size)
        self.fc1 = nn.Linear(hidden_downscale_size, self.hidden_size)

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
        q_in: torch.Tensor = None,
        d_in: torch.Tensor = None,
        ads_in: torch.Tensor = None,
    ):
        q_out = torch.tanh(self.fc0(q_in))
        d_out = torch.tanh(self.fc0_offline(d_in))
        ads_out = torch.tanh(self.fc0_offline(ads_in))

        q_out = self.fc1(q_out)
        d_out = self.fc1(d_out)
        ads_out = self.fc1(ads_out)

        if not self.training:
            q_out = self._quantization(q_out)
            d_out = self._quantization(d_out)
            ads_out = self._quantization(ads_out)

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


class MultiTriPostLayerV2(nn.Module):
    def __init__(
        self,
        num_tasks: int = 1,
        num_classes: int = 1,
        sim4score: str = "cosine",
        pooltype: str = "bert",
        hidden_size: int = 512,
        offline_hidden_size: int = 768,
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
        self.offline_hidden_size = offline_hidden_size
        self.hidden_downscale_size = hidden_downscale_size
        self.reslayer_use_bn = reslayer_use_bn
        self.reslayer_downscale_size = reslayer_downscale_size
        self.reslayer_hidden_size = reslayer_hidden_size

        if self.pooltype in ["weight"]:
            self.attn = nn.Linear(self.hidden_size, 1, bias=False)
            self.offline_attn = nn.Linear(self.offline_hidden_size, 1, bias=False)

        self.num_tasks = num_tasks
        self.mtfc = nn.ModuleDict(
            {
                str(k): TriPostLayerV2(
                    num_classes=self.num_classes,
                    sim4score=self.sim4score,
                    pooltype=self.pooltype,
                    hidden_size=self.hidden_size,
                    offline_hidden_size=self.offline_hidden_size,
                    hidden_downscale_size=self.hidden_downscale_size,
                    reslayer_use_bn=self.reslayer_use_bn,
                    reslayer_downscale_size=self.reslayer_downscale_size,
                    reslayer_hidden_size=self.reslayer_hidden_size,
                )
                for k in range(self.num_tasks)
            }
        )

    def _attn(self, d_in, d_mask):
        att = self.attn(d_in).squeeze(dim=-1)
        d_mask = d_mask.to(att)
        att = att + (1 - d_mask) * -10000
        att = F.softmax(att, dim=-1)
        return torch.bmm(d_in.transpose(1, 2), att.unsqueeze(dim=-1)).squeeze(dim=-1)

    def _offline_attn(self, d_in, d_mask):
        att = self.offline_attn(d_in).squeeze(dim=-1)
        d_mask = d_mask.to(att)
        att = att + (1 - d_mask) * -10000
        att = F.softmax(att, dim=-1)
        return torch.bmm(d_in.transpose(1, 2), att.unsqueeze(dim=-1)).squeeze(dim=-1)

    def forward(
        self,
        task: torch.Tensor = None,
        qseq_in: torch.Tensor = None,
        qpool_in: torch.Tensor = None,
        q_mask: torch.Tensor = None,
        dseq_in: torch.Tensor = None,
        dpool_in: torch.Tensor = None,
        d_mask: torch.Tensor = None,
        adsseq_in: torch.Tensor = None,
        adspool_in: torch.Tensor = None,
        ads_mask: torch.Tensor = None,
    ):
        if dseq_in is None:
            q_out = (
                self._attn(qseq_in, q_mask) if self.pooltype in ["weight"] else qpool_in
            )
            q_out = (
                torch.tanh(self.fc0(q_out)) if self.sim4score in ["reslayer"] else q_out
            )
            return q_out

        if self.pooltype in ["weight"]:
            q_out = self._attn(qseq_in, q_mask)
            d_out = self._offline_attn(dseq_in, d_mask)
            ads_out = self._offline_attn(adsseq_in, ads_mask)
        else:
            q_out = qpool_in
            d_out = dpool_in
            ads_out = adspool_in

        out = torch.empty(q_out.shape[0], self.num_classes)
        out = torch.zeros_like(out).to(q_out).float()
        for i in range(self.num_tasks):
            out += self.mtfc[str(i)](q_out, d_out, ads_out) * (
                task == i
            ).float().unsqueeze(-1)
        return out
