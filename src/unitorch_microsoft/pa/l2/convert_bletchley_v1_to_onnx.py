# %% [markdown]
# # Bletchley v1 6 Layers ONNX Model

# %%
# convert icev2
import torch
import pandas as pd

iceterm = "/disk/lixin/to_onnx/MMDNNv4_2/ICETerm.txt"
df = pd.read_csv(iceterm, sep="\t", header=None, names=["iceid", "new_iceid"])
df.loc[3180] = [3180, 10000]

origin_model_weight = torch.load("/disk/lixin/to_onnx/MMDNNv4_2/mm_pytorch_model.bin")
ice_embed = origin_model_weight["ice_embedding.weight"]

convert_maxtrix = torch.from_numpy(df["new_iceid"].values - 10000).to(ice_embed)
new_embed = torch.zeros([4000, 288]).to(ice_embed)
for i in range(3181):
    new_embed[int(convert_maxtrix[i]), :] = ice_embed[i, :]

origin_model_weight["ice_embedding.weight"] = new_embed
torch.save(
    origin_model_weight, "/disk/lixin/to_onnx/MMDNNv4_2/convert_mm_model_weight.bin"
)

# %%
import torch
import torch.nn as nn

from transformers.activations import quick_gelu
from unitorch_microsoft.pa.bletchley_v1 import (
    BletchleyTextEncoder,
    get_bletchley_text_config,
)


class MMDNNBletchleyV1ForQuery(nn.Module):
    def __init__(
        self,
        config_type: str,
        projection_dim=288,
        num_ice=4000,
        output_hidden_dim=64,
        padding_idx: int = -1,
        weight_path=None,
    ):
        super().__init__()
        config_type = get_bletchley_text_config(config_type, False)

        self.padding_idx = padding_idx
        self.text_embed_dim = config_type.hidden_size

        self.text_encoder = BletchleyTextEncoder(
            config_type, add_projection_layer=False
        )

        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
            bias=False,
        )

        self.text_layer_norm = nn.LayerNorm(projection_dim)
        self.ice_embedding = nn.Embedding(num_ice, projection_dim)
        self.ice_layer_norm = nn.LayerNorm(projection_dim)

        self.final_text_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )

        self.from_checkpoint(weight_path)

    def from_checkpoint(self, weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward(self, query_input_ids=None, ice_ids=None):
        query_input_ids = query_input_ids + 1
        query_input_ids = torch.cat(
            [torch.tensor([0], dtype=torch.int64).reshape(1, -1), query_input_ids],
            dim=0,
        )
        query_input_ids = torch.cat(
            [query_input_ids, torch.tensor([2], dtype=torch.int64).reshape(1, -1)],
            dim=0,
        )
        query_input_ids = query_input_ids.reshape(1, -1)
        attention_mask = torch.ones([1]).reshape(1, -1).to(query_input_ids)

        query_input_ids[query_input_ids > 250001] = 250001
        query_input_ids[query_input_ids < 0] = 0
        text_outputs = self.text_encoder(
            input_ids=query_input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        ice_ids = ice_ids.reshape(1, -1)
        zero_tensor = torch.tensor(0).to(query_input_ids)
        ice_ids = torch.sub(ice_ids, 10000)
        ice_mask = ice_ids.ne(self.padding_idx)
        ice_ids = torch.where(ice_ids < 0, zero_tensor, ice_ids)
        ice_ids = torch.where(ice_ids > 3999, zero_tensor, ice_ids)
        ice_embeds = self.ice_embedding(ice_ids)
        ice_embeds = ice_embeds * ice_mask.unsqueeze(-1)
        if ice_embeds.dim() == 3:
            ice_embeds = ice_embeds.sum(dim=1)

        text_embeds = self.text_layer_norm(quick_gelu(text_embeds))
        ice_embeds = self.ice_layer_norm(ice_embeds)
        text_embeds = torch.cat([text_embeds, ice_embeds], dim=-1)

        text_embeds = self.final_text_projection(quick_gelu(text_embeds))
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds.view(1, 64), text_embeds[:, 0].view(1).squeeze()


model_file = "/disk/lixin/to_onnx/MMDNNv4_2/convert_mm_model_weight.bin"
model = MMDNNBletchleyV1ForQuery(config_type="0.3B", weight_path=model_file)
model.eval()

# %%
query = torch.tensor([160, 13, 101089, 90]).unsqueeze(-1)
qice = torch.tensor([13000, -1, -1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)

torch.set_printoptions(profile="full")
torch.onnx.export(
    model,
    (query, qice),
    f="/home/lixin/model.onnx",
    input_names=["query", "qice"],
    output_names=["qvec", "dummy_score"],
    export_params=True,
    dynamic_axes={"query": {0: "querylen"}, "qice": {0: "qicelen"}},
    do_constant_folding=True,
    verbose=False,
    opset_version=13,
)

# %%
import torch
import onnxruntime as ort

query = torch.tensor([161, 14, 101090, 91]).unsqueeze(-1)
qice = torch.tensor(
    [13063, 13061, 12184, 12213, 12693, 13311, 10886, 11386, 13059, 13074]
).unsqueeze(0)

ort_session = ort.InferenceSession("/home/lixin/model.onnx")
outputs = ort_session.run(
    None,
    {"query": query.cpu().numpy(), "qice": qice.cpu().numpy()},
)
print("onnx output: ", outputs)
embedding, dummy_score = model(query, qice)
print(embedding)
print(dummy_score)

# %%
for x in ort_session.get_inputs():
    print(x)
for x in ort_session.get_outputs():
    print(x)

# %%
import onnxruntime.transformers.optimizer as optimizer
from onnxruntime.transformers.fusion_options import FusionOptions
import onnxruntime.quantization as quantization

input_path_to_unoptimized_onnx_model = "/home/lixin/model.onnx"
output_path_to_optimized_model = "model.optimized.onnx"
output_path_to_quantized_model = "model.optimized.quantized.onnx"

optimized_model = optimizer.optimize_model(input_path_to_unoptimized_onnx_model)
optimized_model.save_model_to_file(output_path_to_optimized_model)

quantization.quantize_dynamic(
    output_path_to_optimized_model, output_path_to_quantized_model
)

# %% [markdown]
# # Bletchley v1 3 Layers ONNX Model

# %%
import torch
import torch.nn as nn

from transformers.activations import quick_gelu
from unitorch_microsoft.pa.bletchley_v1 import (
    BletchleyTextEncoder,
    get_bletchley_text_config,
)


class MMDNNBletchleyV1Layers3ForQuery(nn.Module):
    def __init__(
        self,
        config_type: str,
        num_query_layers=3,
        projection_dim=288,
        num_ice=4000,
        output_hidden_dim=64,
        padding_idx: int = -1,
        weight_path=None,
    ):
        super().__init__()
        config_type = get_bletchley_text_config(config_type, False)

        self.padding_idx = padding_idx
        self.text_embed_dim = config_type.hidden_size
        config_type.num_hidden_layers = num_query_layers

        self.text_encoder = BletchleyTextEncoder(
            config_type, add_projection_layer=False
        )

        self.text_projection = nn.Linear(
            self.text_embed_dim,
            projection_dim,
            bias=False,
        )

        self.text_layer_norm = nn.LayerNorm(projection_dim)
        self.ice_embedding = nn.Embedding(num_ice, projection_dim)
        self.ice_layer_norm = nn.LayerNorm(projection_dim)

        self.final_text_projection = nn.Linear(
            projection_dim + projection_dim,
            output_hidden_dim,
        )

        # self.from_checkpoint(weight_path)

    def from_checkpoint(self, weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward(self, query_input_ids=None, ice_ids=None):
        query_input_ids = query_input_ids + 1
        query_input_ids = torch.cat(
            [torch.tensor([0], dtype=torch.int64).reshape(1, -1), query_input_ids],
            dim=0,
        )
        query_input_ids = torch.cat(
            [query_input_ids, torch.tensor([2], dtype=torch.int64).reshape(1, -1)],
            dim=0,
        )
        query_input_ids = query_input_ids.reshape(1, -1)
        attention_mask = torch.ones([1]).reshape(1, -1).to(query_input_ids)

        query_input_ids[query_input_ids > 250001] = 250001
        query_input_ids[query_input_ids < 0] = 0
        text_outputs = self.text_encoder(
            input_ids=query_input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = text_outputs[:, 0]
        text_embeds = self.text_projection(text_embeds)

        ice_ids = ice_ids.reshape(1, -1)
        zero_tensor = torch.tensor(0).to(query_input_ids)
        ice_ids = torch.sub(ice_ids, 10000)
        ice_mask = ice_ids.ne(self.padding_idx)
        ice_ids = torch.where(ice_ids < 0, zero_tensor, ice_ids)
        ice_ids = torch.where(ice_ids > 3999, zero_tensor, ice_ids)
        ice_embeds = self.ice_embedding(ice_ids)
        ice_embeds = ice_embeds * ice_mask.unsqueeze(-1)
        if ice_embeds.dim() == 3:
            ice_embeds = ice_embeds.sum(dim=1)

        text_embeds = self.text_layer_norm(quick_gelu(text_embeds))
        ice_embeds = self.ice_layer_norm(ice_embeds)
        text_embeds = torch.cat([text_embeds, ice_embeds], dim=-1)

        text_embeds = self.final_text_projection(quick_gelu(text_embeds))
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        return text_embeds.view(1, 64), text_embeds[:, 0].view(1).squeeze()


model_file = "/disk/lixin/to_onnx/MMDNNv4_2/convert_mm_model_weight.bin"
model = MMDNNBletchleyV1Layers3ForQuery(config_type="0.3B", weight_path=model_file)
model.eval()

# %%
query = torch.tensor([160, 13, 101089, 90]).unsqueeze(-1)
qice = torch.tensor([13000, -1, -1, -1, -1, -1, -1, -1, -1, -1]).unsqueeze(0)

torch.set_printoptions(profile="full")
torch.onnx.export(
    model,
    (query, qice),
    f="/home/lixin/model_3layer.onnx",
    input_names=["query", "qice"],
    output_names=["qvec", "dummy_score"],
    export_params=True,
    dynamic_axes={"query": {0: "querylen"}},
    do_constant_folding=True,
    verbose=False,
    opset_version=13,
)

# %%
import torch
import onnxruntime as ort

query = torch.tensor([161, 14, 101090, 91]).unsqueeze(-1)
qice = torch.tensor(
    [13063, 13061, 12184, 12213, 12693, 13311, 10886, 11386, 13059, 13074]
).unsqueeze(0)

ort_session = ort.InferenceSession("/home/lixin/model_3layer.onnx")
outputs = ort_session.run(
    None,
    {"query": query.cpu().numpy(), "qice": qice.cpu().numpy()},
)
print("onnx output: ", outputs)
embedding, dummy_score = model(query, qice)
print(embedding)
print(dummy_score)

# %%
for x in ort_session.get_inputs():
    print(x)
for x in ort_session.get_outputs():
    print(x)


