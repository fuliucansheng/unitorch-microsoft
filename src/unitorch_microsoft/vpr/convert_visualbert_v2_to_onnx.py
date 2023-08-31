# %% [markdown]
# # Visualbert v2 ONNX Model

# %%
from unitorch.cli import cached_path
from transformers import BertTokenizer

def get_bert_tokenizer(
    vocab_path,
    do_lower_case = True,
    do_basic_tokenize = True,
    special_input_ids = dict(),
):
    tokenizer = BertTokenizer(
        vocab_path,
        do_lower_case=do_lower_case,
        do_basic_tokenize=do_basic_tokenize,
    )
    for token, _id in special_input_ids.items():
        tokenizer.added_tokens_encoder[token] = _id
        tokenizer.unique_no_split_tokens.append(token)
        tokenizer.added_tokens_decoder[_id] = token
        tokenizer.add_tokens(token, special_tokens=True)
    return tokenizer

vocab_path = cached_path("https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt")
tokenizer = get_bert_tokenizer(vocab_path = vocab_path)
tokens = tokenizer.tokenize("clothing shoes jewelry women clothing coats jackets vests")
tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
print(tokens)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)

# %%
from unitorch_microsoft.vpr.processing import VisualBertProcessor
from unitorch.cli import cached_path

processor = VisualBertProcessor(
    vocab_path=cached_path(
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
    ),
    max_seq_length=36,
)
tokens = processor._classification(
    "clothing shoes jewelry women clothing coats jackets vests"
)
print(tokens.__tensors__)

# %%
import torch
import torch.nn as nn
from unitorch.cli import cached_path

from transformers.models.visual_bert.modeling_visual_bert import (
    VisualBertConfig,
    VisualBertModel,
)

class VisualBertV2ForQuery(nn.Module):
    def __init__(
        self,
        config_path,
        image_embed_dim = 100,
        projection_dim = 100,
        gradient_checkpointing = False,
        weight_path = None
    ):
        super().__init__()
        self.config = VisualBertConfig.from_json_file(config_path)
        self.config.gradient_checkpointing = gradient_checkpointing
        self.config.visual_embedding_dim = self.config.hidden_size
        self.image_embed_dim = image_embed_dim

        self.image_conv = nn.Linear(image_embed_dim, self.config.hidden_size)
        self.query_bert = VisualBertModel(self.config, add_pooling_layer=False)
        self.projection_dim = projection_dim
        self.query_embed_dim = self.config.hidden_size
        self.query_projection = nn.Linear(self.query_embed_dim, self.projection_dim)

        self.from_checkpoint(weight_path)
    
    def from_checkpoint(self, weight_path):
        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        query_input_ids=None,
        query_image_embeds=None,
    ):
        query_input_ids = query_input_ids.reshape(1, -1)
        query_input_ids[query_input_ids > 30521] = 30521
        query_input_ids[query_input_ids < 0] = 0

        query_image_embeds = query_image_embeds.reshape(1, 1, 100)
        query_image_embeds = self.image_conv(query_image_embeds)
        query_outputs = self.query_bert(
            query_input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            visual_embeds=query_image_embeds,
            visual_attention_mask=None,
            visual_token_type_ids=None,
        )
        query_embeds = self.query_projection(query_outputs[0][:, 0])
        query_embeds = query_embeds / query_embeds.norm(dim=-1, keepdim=True)
        
        return query_embeds.view(1,100), query_embeds[:, 0].view(1).squeeze()


model_file = "/disk/lixin/ckpt/visualbert/pytorch_model.bin"
model = VisualBertV2ForQuery(
    config_path=cached_path("./configs/visualbert.argus.json"), weight_path=model_file
)
model.eval()

# %%
query = torch.tensor(
    [101, 5929, 6007, 11912, 2308, 5929, 15695, 17764, 17447, 2015, 102]
).unsqueeze(-1)
img_embed = "-0.04616177 -0.1159673 -0.08528523 0.1905098 0.03709327 -0.01312277 0.06673015 -0.08224729 -0.01040809 0.04598344 -0.1462961 -0.08063745 0.0884358 -0.01628013 0.148392 0.1082236 -0.1765878 0.09306743 0.0122325 -0.08560283 -0.05029535 0.03084217 0.01596232 -0.1859483 -0.002774475 0.07493474 -0.1663142 0.05123898 0.09619494 -0.1338701 0.08091179 -0.0679644 -0.03450328 0.1885101 0.2552213 0.1810764 -0.06981172 -0.02060077 0.09823886 0.03618161 -0.02384998 0.05091225 0.1621212 -0.01318149 -0.01524111 -0.08737237 -0.08761846 -0.1222196 0.1103951 0.06298082 0.05506718 0.04463443 -0.1053156 0.05489326 0.1009745 -0.06218047 -0.2151767 0.2009955 0.07995496 -0.1002232 -0.03015113 0.01651702 -0.0504215 0.05088477 -0.1110381 -0.08662394 0.24704 -0.06810931 -0.142163 0.02408212 -0.06323168 -0.1578443 0.04513462 -0.05154346 -0.04670211 -0.1431078 -0.06747935 0.07811068 0.09118298 -0.0676966 -0.001526211 -0.01333729 0.08748315 -0.1061855 -0.005310548 -0.03976621 -0.03471386 -0.06432562 -0.09493478 -0.1295277 -0.04983081 -0.00206829 0.1023707 0.02492245 0.203907 -0.09249474 -0.01149014 -0.003467969 -0.1368538 -0.07677495"
img_embed_list = list(map(float, img_embed.split(" ")))
image_embeds = torch.tensor(img_embed_list).unsqueeze(0)

torch.set_printoptions(profile="full")
torch.onnx.export(model, (query, image_embeds), 
    f = './visualbertv2.onnx',
    input_names = ['query', 'AVEV9Str'],
    output_names = ['qvec', 'dummy_score'],
    export_params = True,
    dynamic_axes = {'query' : {0: 'querylen'}},
    do_constant_folding=True,
    verbose=False,
    opset_version=13,
)

# %%
import torch
import onnxruntime as ort

query = torch.tensor(
    [101, 5929, 6007, 11912, 2308, 5929, 15695, 17764, 17447, 2015, 102]
).unsqueeze(-1)
img_embed = "-0.04616177 -0.1159673 -0.08528523 0.1905098 0.03709327 -0.01312277 0.06673015 -0.08224729 -0.01040809 0.04598344 -0.1462961 -0.08063745 0.0884358 -0.01628013 0.148392 0.1082236 -0.1765878 0.09306743 0.0122325 -0.08560283 -0.05029535 0.03084217 0.01596232 -0.1859483 -0.002774475 0.07493474 -0.1663142 0.05123898 0.09619494 -0.1338701 0.08091179 -0.0679644 -0.03450328 0.1885101 0.2552213 0.1810764 -0.06981172 -0.02060077 0.09823886 0.03618161 -0.02384998 0.05091225 0.1621212 -0.01318149 -0.01524111 -0.08737237 -0.08761846 -0.1222196 0.1103951 0.06298082 0.05506718 0.04463443 -0.1053156 0.05489326 0.1009745 -0.06218047 -0.2151767 0.2009955 0.07995496 -0.1002232 -0.03015113 0.01651702 -0.0504215 0.05088477 -0.1110381 -0.08662394 0.24704 -0.06810931 -0.142163 0.02408212 -0.06323168 -0.1578443 0.04513462 -0.05154346 -0.04670211 -0.1431078 -0.06747935 0.07811068 0.09118298 -0.0676966 -0.001526211 -0.01333729 0.08748315 -0.1061855 -0.005310548 -0.03976621 -0.03471386 -0.06432562 -0.09493478 -0.1295277 -0.04983081 -0.00206829 0.1023707 0.02492245 0.203907 -0.09249474 -0.01149014 -0.003467969 -0.1368538 -0.07677495"
img_embed_list = list(map(float, img_embed.split(" ")))
image_embeds = torch.tensor(img_embed_list).unsqueeze(0)

ort_session = ort.InferenceSession("./visualbertv2.onnx")
outputs = ort_session.run(
    None,
    {"query": query.cpu().numpy(), "AVEV9Str": image_embeds.cpu().numpy()},
)
print('onnx output: ', outputs)
embedding, dummy_score = model(query, image_embeds)
print(embedding)
print(dummy_score)


# %%
import torch
import onnxruntime as ort

query = torch.tensor([101, 8145, 7028, 2015, 2986, 2396, 2998, 7008, 102]).unsqueeze(-1)
img_embed = "0.06872461 0.06707684 -0.1004234 0.157478 0.005788147 0.05468951 -0.1689698 -0.2161552 0.005107108 -0.1413067 -0.2213147 -0.02213422 0.09381612 0.01865214 0.04586374 -0.0002564177 0.03646244 0.05179495 -0.05045513 0.03800458 -0.1236463 -0.09200428 -0.008652093 0.1573422 0.1308466 0.04061916 0.04233519 0.126397 -0.1020023 0.01449885 -0.03855991 0.009280446 -0.02262367 -0.02973268 0.0554627 -0.101397 0.2015726 0.01195515 0.04804528 0.1362682 0.08004167 -0.1898625 0.00134357 -0.04880193 0.1572406 -0.1457732 -0.06192927 -0.0002384908 -0.04287158 -0.1308655 0.0839516 0.08932887 -0.02052006 -0.06518085 -0.02623975 -0.1054462 0.06900226 0.1614499 0.1670368 -0.2113939 0.06217249 0.01754874 -0.1132064 -0.01737586 0.00583668 0.0410508 0.02069465 0.03145327 -0.09015376 0.05931452 -0.06863946 0.04490919 -0.2106095 0.1168717 -0.04938331 0.00816984 0.07793544 0.2658825 -0.03320502 0.1367838 -0.00196413 -0.000946969 -0.05130554 -0.1619273 -0.0199189 0.1331647 -0.04231874 0.2170963 0.1372149 0.01060467 0.1185931 0.0985478 0.02126219 -0.09441158 0.05213621 -0.0914111 -0.06110685 0.0133629 -0.04279166 0.07691804"
img_embed_list = list(map(float, img_embed.split(" ")))
image_embeds = torch.tensor(img_embed_list).unsqueeze(0)

ort_session = ort.InferenceSession("./visualbertv2.onnx")
outputs = ort_session.run(
    None,
    {"query": query.cpu().numpy(), "AVEV9Str": image_embeds.cpu().numpy()},
)
print('onnx output: ', outputs)
embedding, dummy_score = model(query, image_embeds)
print(embedding)
print(dummy_score)


# %%
for x in ort_session.get_inputs():
    print(x)
for x in ort_session.get_outputs():
    print(x)

# %%
import onnxruntime.quantization as quantization

quantization.quantize_dynamic(
    "/disk/lixin/ckpt/visualbert/visualbertv2.onnx",
    "/disk/lixin/ckpt/visualbert/visualbertv2_int8.onnx",
)

# %%
query = torch.tensor([101, 8145, 7028, 2015, 2986, 2396, 2998, 7008, 102]).unsqueeze(-1)
img_embed = "0.06872461 0.06707684 -0.1004234 0.157478 0.005788147 0.05468951 -0.1689698 -0.2161552 0.005107108 -0.1413067 -0.2213147 -0.02213422 0.09381612 0.01865214 0.04586374 -0.0002564177 0.03646244 0.05179495 -0.05045513 0.03800458 -0.1236463 -0.09200428 -0.008652093 0.1573422 0.1308466 0.04061916 0.04233519 0.126397 -0.1020023 0.01449885 -0.03855991 0.009280446 -0.02262367 -0.02973268 0.0554627 -0.101397 0.2015726 0.01195515 0.04804528 0.1362682 0.08004167 -0.1898625 0.00134357 -0.04880193 0.1572406 -0.1457732 -0.06192927 -0.0002384908 -0.04287158 -0.1308655 0.0839516 0.08932887 -0.02052006 -0.06518085 -0.02623975 -0.1054462 0.06900226 0.1614499 0.1670368 -0.2113939 0.06217249 0.01754874 -0.1132064 -0.01737586 0.00583668 0.0410508 0.02069465 0.03145327 -0.09015376 0.05931452 -0.06863946 0.04490919 -0.2106095 0.1168717 -0.04938331 0.00816984 0.07793544 0.2658825 -0.03320502 0.1367838 -0.00196413 -0.000946969 -0.05130554 -0.1619273 -0.0199189 0.1331647 -0.04231874 0.2170963 0.1372149 0.01060467 0.1185931 0.0985478 0.02126219 -0.09441158 0.05213621 -0.0914111 -0.06110685 0.0133629 -0.04279166 0.07691804"
img_embed_list = list(map(float, img_embed.split(" ")))
image_embeds = torch.tensor(img_embed_list).unsqueeze(0)

ort_session = ort.InferenceSession("/disk/lixin/ckpt/visualbert/visualbertv2.onnx")
outputs = ort_session.run(
    None,
    {"query": query.cpu().numpy(), "AVEV9Str": image_embeds.cpu().numpy()},
)
print("onnx output: ", outputs)

ort_session = ort.InferenceSession("/disk/lixin/ckpt/visualbert/visualbertv2_int8.onnx")
outputs = ort_session.run(
    None,
    {"query": query.cpu().numpy(), "AVEV9Str": image_embeds.cpu().numpy()},
)
print("onnx output: ", outputs)
